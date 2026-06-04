import math
import numpy as np
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import os
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
import tf2_ros
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from rclpy.time import Time


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    # geometry_msgs Quaternion: x, y, z, w
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class RLLocalController(Node):
    def __init__(self):
        super().__init__("rl_local_controller")
        pkg_share = get_package_share_directory('m3_ros2')
        self.declare_parameter("onnx_path", os.path.join(pkg_share, 'rl_policies', 'policy.onnx'))
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/plan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("num_path_points", 8)
        self.declare_parameter("path_step", 8)
        self.declare_parameter("num_scan_rays", 72)

        self.declare_parameter("max_vx", 0.5)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_wz", 1.0)

        self.onnx_path = self.get_parameter("onnx_path").value
        if not self.onnx_path:
            raise RuntimeError("onnx_path parameter is empty")

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.num_path_points = int(self.get_parameter("num_path_points").value)
        self.path_step = int(self.get_parameter("path_step").value)
        self.num_scan_rays = int(self.get_parameter("num_scan_rays").value)

        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_scan = None
        self.latest_odom = None
        self.latest_path = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self._last_tf_warn_ns = 0

        self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").value,
            self.scan_callback,
            10,
        )

        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            10,
        )

        self.create_subscription(
            Path,
            self.get_parameter("path_topic").value,
            self.path_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            10,
        )

        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info(f"Loaded ONNX policy: {self.onnx_path}")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        ranges = np.nan_to_num(
            ranges,
            nan=msg.range_max,
            posinf=msg.range_max,
            neginf=msg.range_max,
        )

        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        # Resample scan to exactly 72 rays if needed.
        if len(ranges) != self.num_scan_rays:
            old_idx = np.linspace(0, len(ranges) - 1, len(ranges))
            new_idx = np.linspace(0, len(ranges) - 1, self.num_scan_rays)
            ranges = np.interp(new_idx, old_idx, ranges).astype(np.float32)

        # IsaacLab scan was normalized by max_range.
        scan_norm = ranges / float(msg.range_max)
        scan_norm = np.clip(scan_norm, 0.0, 1.0)

        self.latest_scan = scan_norm.astype(np.float32)

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return

        pts = []
        for p in msg.poses:
            pts.append([p.pose.position.x, p.pose.position.y])

        self.latest_path = np.array(pts, dtype=np.float32)

    def build_local_path_obs(self, robot_xy, robot_yaw):
        path = self.latest_path

        if path is None or len(path) < 2:
            return (
                np.zeros(self.num_path_points * 2, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )

        d = np.linalg.norm(path - robot_xy[None, :], axis=1)
        nearest_idx = int(np.argmin(d))

        local_points = []

        for k in range(self.num_path_points):
            idx = nearest_idx + (k + 1) * self.path_step
            idx = min(idx, len(path) - 1)

            rel = path[idx] - robot_xy

            c = math.cos(-robot_yaw)
            s = math.sin(-robot_yaw)

            x_b = c * rel[0] - s * rel[1]
            y_b = s * rel[0] + c * rel[1]

            local_points.extend([x_b, y_b])

        local_path_window = np.array(local_points, dtype=np.float32)

        next_idx = min(nearest_idx + 3, len(path) - 1)
        p0 = path[nearest_idx]
        p1 = path[next_idx]

        path_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        heading_error = np.array([wrap_to_pi(path_yaw - robot_yaw)], dtype=np.float32)

        cross_track = np.array([float(np.min(d))], dtype=np.float32)

        return local_path_window, heading_error, cross_track

    def build_observation(self):
        if self.latest_scan is None or self.latest_odom is None or self.latest_path is None:
            return None

        pose_result = self.get_robot_pose_in_map()
        if pose_result is None:
            return None

        robot_xy, robot_yaw = pose_result
        odom = self.latest_odom

        local_path_window, heading_error, cross_track = self.build_local_path_obs(
            robot_xy,
            robot_yaw,
        )

        base_lin_vel = np.array(
            [
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
            ],
            dtype=np.float32,
        )

        base_ang_vel = np.array(
            [odom.twist.twist.angular.z],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                local_path_window,
                heading_error,
                cross_track,
                self.latest_scan,
                base_lin_vel,
                base_ang_vel,
                self.previous_action,
            ],
            axis=0,
        ).astype(np.float32)

        return obs
    
    def get_robot_pose_in_map(self):
        try:
            # Time() means latest available TF.
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.20),
            )

        except Exception as e:
            # Manual throttled warning because rclpy logger has no warn_throttle().
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_tf_warn_ns > int(2.0 * 1e9):
                self.get_logger().warn(
                    f"TF lookup failed {self.map_frame}->{self.base_frame}: {e}"
                )
                self._last_tf_warn_ns = now_ns

            return None

        robot_xy = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
            ],
            dtype=np.float32,
        )

        q = tf.transform.rotation
        robot_yaw = yaw_from_quat(q)

        return robot_xy, robot_yaw

    def control_loop(self):
        obs = self.build_observation()
        if obs is None:
            return

        obs_batch = obs.reshape(1, -1)

        action = self.session.run(
            [self.output_name],
            {self.input_name: obs_batch},
        )[0]

        action = np.asarray(action).reshape(-1)[:3]
        action = np.clip(action, -1.0, 1.0)

        self.previous_action = action.astype(np.float32)

        cmd = Twist()
        cmd.linear.x = float(action[0] * self.max_vx)
        cmd.linear.y = float(action[1] * self.max_vy)
        cmd.angular.z = float(action[2] * self.max_wz)

        self.cmd_pub.publish(cmd)

        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        self._debug_count += 1

        if self._debug_count % 20 == 0:
            self.get_logger().info(
                f"obs: heading={obs[16]:+.3f}, cte={obs[17]:+.3f}, "
                f"scan_min={np.min(obs[18:90]):.3f}, "
                f"raw_action=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}], "
                f"cmd=[{action[0]*self.max_vx:+.3f}, {action[1]*self.max_vy:+.3f}, {action[2]*self.max_wz:+.3f}]"
            )

def main():
    rclpy.init()
    node = RLLocalController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()