import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path as PathMsg
from rclpy.action import ActionClient
from rclpy.node import Node
import subprocess


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return qz, qw


class PathReplayNavigator(Node):
    """
    Reads one saved Nav2 global path dataset file and:
      1. publishes saved path to /dataset_path for RViz,
      2. optionally publishes initial pose,
      3. optionally sends saved goal to Nav2 NavigateToPose.
    """

    def __init__(self, args):
        super().__init__("path_replay_navigator")

        self.args = args

        data = np.load(args.path_file)

        self.start = data["start"].astype(np.float32)
        self.goal = data["goal"].astype(np.float32)
        self.path_xy = data["path_xy"].astype(np.float32)

        self.get_logger().info(f"Loaded file: {args.path_file}")
        self.get_logger().info(f"start: {self.start}")
        self.get_logger().info(f"goal : {self.goal}")
        self.get_logger().info(f"path_xy shape: {self.path_xy.shape}")

        self.path_pub = self.create_publisher(PathMsg, args.path_topic, 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            args.initial_pose_topic,
            10,
        )

        self.nav_client = ActionClient(self, NavigateToPose, args.navigate_action)

        self.path_msg = self.make_path_msg()

        self.timer = self.create_timer(args.publish_period, self.publish_saved_path)

        self.initial_pose_sent = False
        self.goal_sent = False
        self.robot_teleported = False


        self.startup_timer = self.create_timer(1.0, self.startup_sequence)

    def make_pose_stamped(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.args.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        qz, qw = yaw_to_quat(float(yaw))
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    def make_path_msg(self):
        msg = PathMsg()
        msg.header.frame_id = self.args.frame_id

        for i in range(self.path_xy.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = self.args.frame_id

            pose.pose.position.x = float(self.path_xy[i, 0])
            pose.pose.position.y = float(self.path_xy[i, 1])
            pose.pose.position.z = 0.02

            # Estimate orientation from next path point.
            if i < self.path_xy.shape[0] - 1:
                dx = float(self.path_xy[i + 1, 0] - self.path_xy[i, 0])
                dy = float(self.path_xy[i + 1, 1] - self.path_xy[i, 1])
            else:
                dx = float(self.path_xy[i, 0] - self.path_xy[i - 1, 0])
                dy = float(self.path_xy[i, 1] - self.path_xy[i - 1, 1])

            yaw = math.atan2(dy, dx)
            qz, qw = yaw_to_quat(yaw)

            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            msg.poses.append(pose)

        return msg

    def publish_saved_path(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        for pose in self.path_msg.poses:
            pose.header.stamp = self.path_msg.header.stamp

        self.path_pub.publish(self.path_msg)

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.args.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(self.start[0])
        msg.pose.pose.position.y = float(self.start[1])
        msg.pose.pose.position.z = 0.0

        qz, qw = yaw_to_quat(float(self.start[2]))
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Basic covariance for AMCL initial pose.
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685

        self.initial_pose_pub.publish(msg)

        self.get_logger().info(
            f"Published initial pose: x={self.start[0]:.3f}, y={self.start[1]:.3f}, yaw={self.start[2]:.3f}"
        )

    def send_goal(self):
        self.get_logger().info(f"Waiting for Nav2 action: {self.args.navigate_action}")
        self.nav_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.make_pose_stamped(
            self.goal[0],
            self.goal[1],
            self.goal[2],
        )

        self.get_logger().info(
            f"Sending goal: x={self.goal[0]:.3f}, y={self.goal[1]:.3f}, yaw={self.goal[2]:.3f}"
        )

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Nav2 goal rejected.")
            return

        self.get_logger().info("Nav2 goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        if self.args.print_feedback:
            self.get_logger().info(
                f"distance_remaining={feedback.distance_remaining:.3f}, "
                f"navigation_time={feedback.navigation_time.sec}s"
            )

    def result_callback(self, future):
        result = future.result()
        self.get_logger().info(f"Navigation finished with status: {result.status}")

    def startup_sequence(self):
        # Keep publishing path always.
        self.publish_saved_path()

        if self.args.teleport_robot and not self.robot_teleported:
                self.robot_teleported = self.teleport_robot_to_start()
                return

        if self.args.set_initial_pose and not self.initial_pose_sent:
            self.publish_initial_pose()
            self.initial_pose_sent = True
            return

        if self.args.send_goal and not self.goal_sent:
            self.send_goal()
            self.goal_sent = True
            return

        if (
            (not self.args.teleport_robot or self.robot_teleported)
            and (not self.args.set_initial_pose or self.initial_pose_sent)
            and (not self.args.send_goal or self.goal_sent)
        ):
            self.startup_timer.cancel()
    
    def teleport_robot_to_start(self):
        qz, qw = yaw_to_quat(float(self.start[2]))

        service_name = f"/world/{self.args.gz_world}/set_pose"

        req = (
            f'name: "{self.args.robot_entity}" '
            f'position {{ x: {float(self.start[0])} y: {float(self.start[1])} z: {self.args.robot_z} }} '
            f'orientation {{ x: 0.0 y: 0.0 z: {qz} w: {qw} }}'
        )

        cmd = [
            "gz", "service",
            "-s", service_name,
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", req,
        ]

        self.get_logger().info(f"Teleporting robot using gz service: {service_name}")
        self.get_logger().info(f"Request: {req}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            self.get_logger().error("Gazebo teleport failed.")
            self.get_logger().error(result.stderr)
            self.get_logger().error(result.stdout)
            return False

        self.get_logger().info(result.stdout)
        self.get_logger().info(
            f"Physically teleported '{self.args.robot_entity}' to "
            f"x={self.start[0]:.3f}, y={self.start[1]:.3f}, yaw={self.start[2]:.3f}"
        )
        return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_file", required=True, help="Saved .npz path file.")
    parser.add_argument("--frame_id", default="map")

    parser.add_argument("--path_topic", default="/dataset_path")
    parser.add_argument("--initial_pose_topic", default="/initialpose")
    parser.add_argument("--navigate_action", default="/navigate_to_pose")

    parser.add_argument("--publish_period", type=float, default=0.5)

    parser.add_argument("--set_initial_pose", action="store_true")
    parser.add_argument("--send_goal", action="store_true")
    parser.add_argument("--print_feedback", action="store_true")

    parser.add_argument("--teleport_robot", action="store_true")
    parser.add_argument("--gz_world", required=True)
    parser.add_argument("--robot_entity", required=True)
    parser.add_argument("--robot_z", type=float, default=0.05)

    args = parser.parse_args()

    if not Path(args.path_file).exists():
        raise FileNotFoundError(args.path_file)

    rclpy.init()
    node = PathReplayNavigator(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()