#!/usr/bin/env python3

import math
import time
from enum import Enum
from obstacle_detector.msg import Obstacles
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener, TransformException


class State(Enum):
    NORMAL = 0
    ESCAPING = 1
    STOPPED_NO_SPACE = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FreeSpaceSupervisor(Node):

    def __init__(self):
        super().__init__("freespace_supervisor")

        self.declare_parameter("mppi_cmd_topic", "/cmd_vel_smoothed_disabled")
        self.declare_parameter("output_cmd_topic", "/cmd_vel")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("control_rate", 20.0)

        self.declare_parameter("robot_radius", 0.24)
        self.declare_parameter("safety_margin", 0.08)
        self.declare_parameter("lethal_cost", 75)
        self.declare_parameter("allow_unknown", False)

        self.declare_parameter("trigger_x_min", 0.05)
        self.declare_parameter("trigger_x_max", 1.10)
        self.declare_parameter("trigger_y_abs", 0.38)

        self.declare_parameter(
            "candidate_offsets",
            "0.00,0.45;0.00,-0.45;"
            "0.20,0.55;0.20,-0.55;"
            "0.45,0.55;0.45,-0.55;"
            "-0.15,0.45;-0.15,-0.45"
        )

        self.declare_parameter("target_tolerance", 0.12)
        self.declare_parameter("min_escape_time", 0.35)
        self.declare_parameter("max_escape_time", 1.2)

        self.declare_parameter("kp_x", 0.9)
        self.declare_parameter("kp_y", 1.2)
        self.declare_parameter("kp_yaw", 0.8)

        self.declare_parameter("max_escape_vx", 0.35)
        self.declare_parameter("max_escape_vy", 0.25)
        self.declare_parameter("max_escape_wz", 0.8)

        self.declare_parameter("max_output_vx", 0.75)
        self.declare_parameter("max_output_vy", 0.75)
        self.declare_parameter("max_output_wz", 2.0)

        self.declare_parameter("mppi_timeout", 0.5)
        self.declare_parameter("release_clear_cycles", 4)

        self.nav_active = False

        self.declare_parameter("tracked_obstacles_topic", "/tracked_obstacles")

        self.declare_parameter("dynamic_speed_threshold", 0.12)
        self.declare_parameter("prediction_horizon", 2.0)
        self.declare_parameter("prediction_dt", 0.2)

        self.declare_parameter("dynamic_x_min", 0.0)
        self.declare_parameter("dynamic_x_max", 1.6)
        self.declare_parameter("dynamic_y_abs", 0.45)

        self.declare_parameter("dynamic_radius_margin", 0.12)

        self.declare_parameter("nav_status_topic", "/navigate_to_pose/_action/status")
        self.nav_status_topic = self.get_parameter("nav_status_topic").value

        self.declare_parameter("tracked_obstacles_topic", "/tracked_obstacles")

        self.declare_parameter("dynamic_speed_threshold", 0.12)
        self.declare_parameter("prediction_horizon", 2.0)
        self.declare_parameter("prediction_dt", 0.2)

        self.declare_parameter("dynamic_x_min", 0.0)
        self.declare_parameter("dynamic_x_max", 1.6)
        self.declare_parameter("dynamic_y_abs", 0.45)

        self.declare_parameter("dynamic_radius_margin", 0.12)

        self.last_obstacles = None
        self.last_obstacles_time = 0.0
        self.mppi_cmd_topic = self.get_parameter("mppi_cmd_topic").value
        self.output_cmd_topic = self.get_parameter("output_cmd_topic").value
        self.costmap_topic = self.get_parameter("costmap_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.control_rate = self.get_parameter("control_rate").value

        self.robot_radius = self.get_parameter("robot_radius").value
        self.safety_margin = self.get_parameter("safety_margin").value
        self.lethal_cost = self.get_parameter("lethal_cost").value
        self.allow_unknown = self.get_parameter("allow_unknown").value

        self.trigger_x_min = self.get_parameter("trigger_x_min").value
        self.trigger_x_max = self.get_parameter("trigger_x_max").value
        self.trigger_y_abs = self.get_parameter("trigger_y_abs").value

        self.candidates = self.parse_candidates(
            self.get_parameter("candidate_offsets").value
        )

        self.target_tolerance = self.get_parameter("target_tolerance").value
        self.min_escape_time = self.get_parameter("min_escape_time").value
        self.max_escape_time = self.get_parameter("max_escape_time").value

        self.kp_x = self.get_parameter("kp_x").value
        self.kp_y = self.get_parameter("kp_y").value
        self.kp_yaw = self.get_parameter("kp_yaw").value

        self.max_escape_vx = self.get_parameter("max_escape_vx").value
        self.max_escape_vy = self.get_parameter("max_escape_vy").value
        self.max_escape_wz = self.get_parameter("max_escape_wz").value

        self.max_output_vx = self.get_parameter("max_output_vx").value
        self.max_output_vy = self.get_parameter("max_output_vy").value
        self.max_output_wz = self.get_parameter("max_output_wz").value

        self.mppi_timeout = self.get_parameter("mppi_timeout").value
        self.release_clear_cycles = self.get_parameter("release_clear_cycles").value

        self.costmap = None
        self.odom = None
        self.last_mppi_cmd = Twist()
        self.last_mppi_time = 0.0

        self.state = State.NORMAL
        self.escape_target = None
        self.escape_start_time = 0.0
        self.clear_count = 0

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Twist,
            self.mppi_cmd_topic,
            self.mppi_cmd_callback,
            20
        )

        self.create_subscription(
            OccupancyGrid,
            self.costmap_topic,
            self.costmap_callback,
            10
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        self.create_subscription(
            GoalStatusArray,
            self.nav_status_topic,
            self.nav_status_callback,
            10
        )

        self.create_subscription(
            Obstacles,
            self.tracked_obstacles_topic,
            self.obstacles_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.output_cmd_topic,
            20
        )

        self.state_pub = self.create_publisher(
            String,
            "/freespace_supervisor/state",
            10
        )

        self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info(
            f"Free-space supervisor running: {self.mppi_cmd_topic} -> {self.output_cmd_topic}"
        )


    def obstacles_callback(self, msg):
        self.last_obstacles = msg
        self.last_obstacles_time = time.time()

    def transform_point_to_costmap_frame(self, x, y, source_frame):
        target_frame = self.costmap.header.frame_id

        if source_frame == target_frame:
            return x, y

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.03)
            )

            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            yaw = yaw_from_quat(tf.transform.rotation)

            c = math.cos(yaw)
            s = math.sin(yaw)

            wx = tx + c * x - s * y
            wy = ty + s * x + c * y

            return wx, wy

        except TransformException:
            return None
    

    def transform_velocity_to_costmap_frame(self, vx, vy, source_frame):
        target_frame = self.costmap.header.frame_id

        if source_frame == target_frame:
            return vx, vy

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.03)
            )

            yaw = yaw_from_quat(tf.transform.rotation)

            c = math.cos(yaw)
            s = math.sin(yaw)

            out_vx = c * vx - s * vy
            out_vy = s * vx + c * vy

            return out_vx, out_vy

        except TransformException:
            return None
    
    def velocity_world_to_base(self, robot, vx, vy):
        c = math.cos(robot["yaw"])
        s = math.sin(robot["yaw"])

        bx = c * vx + s * vy
        by = -s * vx + c * vy

        return bx, by
    
    def dynamic_obstacle_risk(self, robot):
        if self.last_obstacles is None:
            return {
                "active": False,
                "preferred_side": 0,
                "reason": "no_obstacles"
            }

        if time.time() - self.last_obstacles_time > 0.5:
            return {
                "active": False,
                "preferred_side": 0,
                "reason": "stale_obstacles"
            }

        source_frame = self.last_obstacles.header.frame_id

        best_risk = None

        # Use circles first. They are better for dynamic obstacle reaction.
        for circle in self.last_obstacles.circles:
            obs_speed = math.hypot(circle.velocity.x, circle.velocity.y)

            # This prevents static walls / static furniture from triggering escape.
            if obs_speed < self.dynamic_speed_threshold:
                continue

            p = self.transform_point_to_costmap_frame(
                circle.center.x,
                circle.center.y,
                source_frame
            )

            v = self.transform_velocity_to_costmap_frame(
                circle.velocity.x,
                circle.velocity.y,
                source_frame
            )

            if p is None or v is None:
                continue

            ox, oy = p
            ovx, ovy = v

            bx, by = self.world_to_base_error(robot, ox, oy)
            bvx, bvy = self.velocity_world_to_base(robot, ovx, ovy)

            # Relative velocity in robot frame.
            # MPPI command is robot velocity in base_link.
            rvx = bvx - self.last_mppi_cmd.linear.x
            rvy = bvy - self.last_mppi_cmd.linear.y

            radius = circle.radius + self.robot_radius + self.dynamic_radius_margin

            t = 0.0
            while t <= self.prediction_horizon:
                px = bx + rvx * t
                py = by + rvy * t

                in_front = self.dynamic_x_min <= px <= self.dynamic_x_max
                in_corridor = abs(py) <= self.dynamic_y_abs + radius

                if in_front and in_corridor:
                    # If obstacle is on left, prefer escaping right.
                    # If obstacle is on right, prefer escaping left.
                    preferred_side = -1 if py > 0.0 else 1

                    risk = {
                        "active": True,
                        "preferred_side": preferred_side,
                        "time": t,
                        "x": px,
                        "y": py,
                        "speed": obs_speed,
                        "uid": getattr(circle, "uid", -1),
                        "reason": "circle_prediction"
                    }

                    if best_risk is None or risk["time"] < best_risk["time"]:
                        best_risk = risk

                    break

                t += self.prediction_dt

        if best_risk is not None:
            return best_risk

        return {
            "active": False,
            "preferred_side": 0,
            "reason": "no_predicted_collision"
        }

    def parse_candidates(self, text):
        result = []
        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue
            x, y = item.split(",")
            result.append((float(x), float(y)))
        return result

    def mppi_cmd_callback(self, msg):
        self.last_mppi_cmd = msg
        self.last_mppi_time = time.time()

    def costmap_callback(self, msg):
        self.costmap = msg

    def odom_callback(self, msg):
        self.odom = msg

    def nav_status_callback(self, msg):
        self.nav_active = False

        for status in msg.status_list:
            # 1 = ACCEPTED
            # 2 = EXECUTING
            # 3 = CANCELING
            if status.status in [1, 2, 3]:
                self.nav_active = True
                return
            
    def mppi_is_waiting(self):
        if time.time() - self.last_mppi_time > self.mppi_timeout:
            return False

        vx = self.last_mppi_cmd.linear.x
        vy = self.last_mppi_cmd.linear.y
        wz = self.last_mppi_cmd.angular.z

        speed = math.sqrt(vx * vx + vy * vy)

        # MPPI is effectively stopped / waiting.
        return speed < 0.04 and abs(wz) < 0.08
    
    def choose_escape_side(self, robot, preferred_side=0):
        left_free = self.side_is_free(robot, 1)
        right_free = self.side_is_free(robot, -1)

        if preferred_side == 1 and left_free:
            return 1

        if preferred_side == -1 and right_free:
            return -1

        if left_free and not right_free:
            return 1

        if right_free and not left_free:
            return -1

        if left_free and right_free:
            left_score = self.side_clearance_score(robot, 1)
            right_score = self.side_clearance_score(robot, -1)

            if left_score >= right_score:
                return 1
            else:
                return -1

        return 0
    
    def side_is_free(self, robot, side):
        # side = +1 means left, -1 means right
        # Check a lateral escape corridor, not just one point.

        lateral_distance = 0.55
        forward_span = 0.35

        samples = [
            (0.00, side * 0.25),
            (0.00, side * 0.40),
            (0.00, side * lateral_distance),
            (0.20, side * 0.35),
            (0.20, side * lateral_distance),
            (-0.10, side * 0.35),
        ]

        for bx, by in samples:
            wx, wy = self.base_to_world(robot, bx, by)

            if not self.footprint_safe(wx, wy):
                return False

            if not self.segment_safe(robot["x"], robot["y"], wx, wy):
                return False

        return True
    
    def side_clearance_score(self, robot, side):
        samples = [
            (0.00, side * 0.35),
            (0.00, side * 0.55),
            (0.20, side * 0.55),
        ]

        score = 0.0

        for bx, by in samples:
            wx, wy = self.base_to_world(robot, bx, by)
            score += self.clearance_score(wx, wy)

        return score
    
    def publish_escape_strafe(self, side):
        cmd = Twist()

        # Do not command forward. Just shift sideways out of blocked path.
        cmd.linear.x = 0.0
        cmd.linear.y = side * self.max_escape_vy
        cmd.angular.z = 0.0

        self.cmd_pub.publish(self.limit_cmd(cmd))

    def control_loop(self):
        self.state_pub.publish(String(data=self.state.name))

        if not self.nav_active:
            self.state = State.NORMAL
            self.escape_target = None
            self.clear_count = 0
            self.publish_stop()
            return

        if self.costmap is None:
            self.publish_mppi_or_stop()
            return

        robot = self.get_robot_pose()

        dynamic_risk = self.dynamic_obstacle_risk(robot)
        front_blocked = self.front_blocked(robot)

        if dynamic_risk["active"]:
            self.clear_count = 0
        else:
            self.clear_count += 1
        if robot is None:
            self.publish_mppi_or_stop()
            return

        front_blocked = self.front_blocked(robot)
        # mppi_waiting = self.mppi_is_waiting()

        if front_blocked:
            self.clear_count = 0
        else:
            self.clear_count += 1

        # NORMAL MODE:
        # Do not interfere with MPPI unless MPPI itself is waiting/stopped
        # because of an obstacle in front.
        if self.state == State.NORMAL:
            if dynamic_risk["active"]:
                side = self.choose_escape_side(
                    robot,
                    preferred_side=dynamic_risk["preferred_side"]
                )

                if side == 0:
                    self.state = State.STOPPED_NO_SPACE
                    self.publish_stop()
                    return

                self.escape_target = {
                    "side": side,
                    "start_time": time.time(),
                    "risk": dynamic_risk,
                }

                self.escape_start_time = time.time()
                self.state = State.ESCAPING

                self.get_logger().warn(
                    f"Dynamic obstacle predicted. uid={dynamic_risk.get('uid', -1)}, "
                    f"ttc={dynamic_risk.get('time', -1):.2f}, "
                    f"pos=({dynamic_risk.get('x', 0):.2f}, {dynamic_risk.get('y', 0):.2f}), "
                    f"escape_side={side}"
                )

                self.publish_escape_strafe(side)
                return

            self.publish_mppi_or_stop()
            return

        # STOPPED MODE:
        # Try again only if obstacle is still there and MPPI is still waiting.
        if self.state == State.STOPPED_NO_SPACE:
            if not dynamic_risk["active"]:
                self.release_to_mppi()
                self.publish_mppi_or_stop()
                return

            side = self.choose_escape_side(
                robot,
                preferred_side=dynamic_risk["preferred_side"]
            )

            if side != 0:
                self.escape_target = {
                    "side": side,
                    "start_time": time.time(),
                    "risk": dynamic_risk,
                }
                self.escape_start_time = time.time()
                self.state = State.ESCAPING
                self.publish_escape_strafe(side)
                return

            self.publish_stop()
            return

        # ESCAPING MODE:
        if self.state == State.ESCAPING:
            elapsed = time.time() - self.escape_start_time
            side = self.escape_target["side"]

            # Release only when predicted dynamic risk is gone.
            if self.clear_count >= self.release_clear_cycles:
                self.release_to_mppi()
                self.publish_mppi_or_stop()
                return

            if elapsed > self.max_escape_time:
                self.release_to_mppi()
                self.publish_mppi_or_stop()
                return

            if not self.side_is_free(robot, side):
                other_side = -side

                if self.side_is_free(robot, other_side):
                    self.escape_target["side"] = other_side
                    self.escape_start_time = time.time()
                    self.publish_escape_strafe(other_side)
                    return

                self.state = State.STOPPED_NO_SPACE
                self.publish_stop()
                return

            self.publish_escape_strafe(side)
            return

    def get_robot_pose(self):
        frame = self.costmap.header.frame_id

        try:
            tf = self.tf_buffer.lookup_transform(
                frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.03)
            )

            q = tf.transform.rotation

            return {
                "x": tf.transform.translation.x,
                "y": tf.transform.translation.y,
                "yaw": yaw_from_quat(q),
                "frame": frame,
            }

        except TransformException:
            pass

        if self.odom is None:
            return None

        q = self.odom.pose.pose.orientation

        return {
            "x": self.odom.pose.pose.position.x,
            "y": self.odom.pose.pose.position.y,
            "yaw": yaw_from_quat(q),
            "frame": self.odom.header.frame_id,
        }

    def base_to_world(self, robot, bx, by):
        c = math.cos(robot["yaw"])
        s = math.sin(robot["yaw"])

        wx = robot["x"] + c * bx - s * by
        wy = robot["y"] + s * bx + c * by

        return wx, wy

    def world_to_base_error(self, robot, wx, wy):
        dx = wx - robot["x"]
        dy = wy - robot["y"]

        c = math.cos(robot["yaw"])
        s = math.sin(robot["yaw"])

        bx = c * dx + s * dy
        by = -s * dx + c * dy

        return bx, by

    def world_to_grid(self, x, y):
        info = self.costmap.info

        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution

        gx = int((x - ox) / res)
        gy = int((y - oy) / res)

        if gx < 0 or gy < 0:
            return None

        if gx >= info.width or gy >= info.height:
            return None

        return gx, gy

    def cost_at_world(self, x, y):
        ij = self.world_to_grid(x, y)

        if ij is None:
            return 100

        gx, gy = ij
        idx = gy * self.costmap.info.width + gx

        return self.costmap.data[idx]

    def occupied(self, x, y):
        c = self.cost_at_world(x, y)

        if c < 0:
            return not self.allow_unknown

        return c >= self.lethal_cost

    def circle_safe(self, x, y, radius):
        res = self.costmap.info.resolution
        steps = int(math.ceil(radius / res))

        center = self.world_to_grid(x, y)
        if center is None:
            return False

        cx, cy = center

        for ix in range(cx - steps, cx + steps + 1):
            for iy in range(cy - steps, cy + steps + 1):

                wx = self.costmap.info.origin.position.x + ix * res
                wy = self.costmap.info.origin.position.y + iy * res

                if dist((x, y), (wx, wy)) <= radius:
                    if self.occupied(wx, wy):
                        return False

        return True

    def footprint_safe(self, x, y):
        return self.circle_safe(
            x,
            y,
            self.robot_radius + self.safety_margin
        )

    def segment_safe(self, x1, y1, x2, y2):
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length / 0.04))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            if not self.footprint_safe(x, y):
                return False

        return True

    def front_blocked(self, robot):
        step = 0.05

        x = self.trigger_x_min
        while x <= self.trigger_x_max:
            y = -self.trigger_y_abs

            while y <= self.trigger_y_abs:
                wx, wy = self.base_to_world(robot, x, y)

                if self.occupied(wx, wy):
                    return True

                y += step

            x += step

        return False

    def clearance_score(self, x, y):
        search_radius = 1.0
        res = self.costmap.info.resolution
        steps = int(search_radius / res)

        center = self.world_to_grid(x, y)
        if center is None:
            return 0.0

        cx, cy = center
        best = search_radius

        for ix in range(cx - steps, cx + steps + 1):
            for iy in range(cy - steps, cy + steps + 1):
                wx = self.costmap.info.origin.position.x + ix * res
                wy = self.costmap.info.origin.position.y + iy * res

                if self.occupied(wx, wy):
                    d = dist((x, y), (wx, wy))
                    best = min(best, d)

        return best

    def find_escape_target(self, robot):
        best = None
        best_score = -999.0

        for bx, by in self.candidates:
            wx, wy = self.base_to_world(robot, bx, by)

            if not self.footprint_safe(wx, wy):
                continue

            if not self.segment_safe(robot["x"], robot["y"], wx, wy):
                continue

            clearance = self.clearance_score(wx, wy)

            score = 0.0
            score += 4.0 * clearance
            score += 1.5 * abs(by)
            score += 0.3 * max(0.0, bx)
            score -= 0.8 * max(0.0, -bx)

            if score > best_score:
                best_score = score
                best = {
                    "x": wx,
                    "y": wy,
                    "yaw": robot["yaw"],
                    "offset_x": bx,
                    "offset_y": by,
                    "score": score,
                }

        return best

    def publish_escape_cmd(self, robot):
        if self.escape_target is None:
            self.publish_stop()
            return

        ex, ey = self.world_to_base_error(
            robot,
            self.escape_target["x"],
            self.escape_target["y"]
        )

        cmd = Twist()

        cmd.linear.x = clamp(
            self.kp_x * ex,
            -self.max_escape_vx,
            self.max_escape_vx
        )

        cmd.linear.y = clamp(
            self.kp_y * ey,
            -self.max_escape_vy,
            self.max_escape_vy
        )

        cmd.angular.z = 0.0

        cmd = self.limit_cmd(cmd)

        self.cmd_pub.publish(cmd)

    def publish_mppi_or_stop(self):
        if time.time() - self.last_mppi_time > self.mppi_timeout:
            self.publish_stop()
            return

        self.cmd_pub.publish(
            self.limit_cmd(self.last_mppi_cmd)
        )

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def release_to_mppi(self):
        self.get_logger().info("Releasing control back to MPPI")
        self.state = State.NORMAL
        self.escape_target = None
        self.clear_count = 0

    def limit_cmd(self, cmd):
        out = Twist()

        out.linear.x = clamp(
            cmd.linear.x,
            -self.max_output_vx,
            self.max_output_vx
        )

        out.linear.y = clamp(
            cmd.linear.y,
            -self.max_output_vy,
            self.max_output_vy
        )

        out.angular.z = clamp(
            cmd.angular.z,
            -self.max_output_wz,
            self.max_output_wz
        )

        return out


def main():
    rclpy.init()
    node = FreeSpaceSupervisor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()