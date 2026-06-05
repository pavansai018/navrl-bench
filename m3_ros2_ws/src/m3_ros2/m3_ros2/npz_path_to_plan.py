#!/usr/bin/env python3

import math
import subprocess
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMsg


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return 0.0, 0.0, qz, qw


class NPZPathToPlan(Node):
    def __init__(self):
        super().__init__("npz_path_to_plan")

        self.declare_parameter("path_file", "")
        self.declare_parameter("path_topic", "/plan")
        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("goal_topic", "/goal_pose")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 5.0)

        self.declare_parameter("teleport_robot", True)
        self.declare_parameter("publish_initial_pose", True)
        self.declare_parameter("publish_goal_pose", True)

        self.declare_parameter("gz_world", "default")
        self.declare_parameter("robot_entity", "m3")

        self.path_file = self.get_parameter("path_file").value
        self.path_topic = self.get_parameter("path_topic").value
        self.initialpose_topic = self.get_parameter("initialpose_topic").value
        self.goal_topic = self.get_parameter("goal_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.teleport_robot = bool(self.get_parameter("teleport_robot").value)
        self.publish_initial_pose = bool(self.get_parameter("publish_initial_pose").value)
        self.publish_goal_pose = bool(self.get_parameter("publish_goal_pose").value)

        self.gz_world = self.get_parameter("gz_world").value
        self.robot_entity = self.get_parameter("robot_entity").value

        if not self.path_file:
            raise RuntimeError("path_file parameter is empty")

        self.path_file = str(Path(self.path_file).expanduser().resolve())

        data = np.load(self.path_file)

        self.start = data["start"].astype(np.float32)   # [x, y, yaw]
        self.goal = data["goal"].astype(np.float32)     # [x, y, yaw]
        self.path_xy = data["path_xy"].astype(np.float32)

        if self.path_xy.shape[0] < 2:
            raise RuntimeError("path_xy has fewer than 2 points")

        self.get_logger().info(f"Loaded: {self.path_file}")
        self.get_logger().info(
            f"start: x={self.start[0]:.3f}, y={self.start[1]:.3f}, yaw={self.start[2]:.3f}"
        )
        self.get_logger().info(
            f"goal : x={self.goal[0]:.3f}, y={self.goal[1]:.3f}, yaw={self.goal[2]:.3f}"
        )
        self.get_logger().info(f"path_xy shape: {self.path_xy.shape}")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.path_pub = self.create_publisher(PathMsg, self.path_topic, qos)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.initialpose_topic,
            qos,
        )
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, qos)

        self.path_msg = self.build_path_msg()
        self.initialpose_msg = self.build_initialpose_msg()
        self.goal_msg = self.build_goal_msg()

        if self.teleport_robot:
            self.teleport_robot_to_start()

        if self.publish_initial_pose:
            self.initialpose_pub.publish(self.initialpose_msg)
            self.get_logger().info(f"Published initial pose on {self.initialpose_topic}")

        if self.publish_goal_pose:
            self.goal_pub.publish(self.goal_msg)
            self.get_logger().info(f"Published goal pose on {self.goal_topic}")

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.publish_path)

    def build_path_msg(self):
        msg = PathMsg()
        msg.header.frame_id = self.frame_id

        for xy in self.path_xy:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(xy[0])
            pose.pose.position.y = float(xy[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

    def build_initialpose_msg(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id

        x, y, yaw = float(self.start[0]), float(self.start[1]), float(self.start[2])
        qx, qy, qz, qw = yaw_to_quat(yaw)

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Reasonable covariance for AMCL initialpose.
        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.05

        return msg

    def build_goal_msg(self):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id

        x, y, yaw = float(self.goal[0]), float(self.goal[1]), float(self.goal[2])
        qx, qy, qz, qw = yaw_to_quat(yaw)

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    def teleport_robot_to_start(self):
        x, y, yaw = float(self.start[0]), float(self.start[1]), float(self.start[2])
        qx, qy, qz, qw = yaw_to_quat(yaw)

        service_name = f"/world/{self.gz_world}/set_pose"

        req = (
            f'name: "{self.robot_entity}" '
            f'position {{ x: {x} y: {y} z: 0.05 }} '
            f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
        )

        cmd = [
            "gz",
            "service",
            "-s",
            service_name,
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req,
        ]

        self.get_logger().info(f"Teleporting robot '{self.robot_entity}' using {service_name}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
            )

            if result.returncode != 0:
                self.get_logger().error("Gazebo teleport failed")
                self.get_logger().error(result.stderr)
            else:
                self.get_logger().info("Gazebo teleport command sent")
                if result.stdout.strip():
                    self.get_logger().info(result.stdout.strip())

        except Exception as e:
            self.get_logger().error(f"Gazebo teleport exception: {e}")

    def publish_path(self):
        now = self.get_clock().now().to_msg()

        self.path_msg.header.stamp = now
        for pose in self.path_msg.poses:
            pose.header.stamp = now

        self.path_pub.publish(self.path_msg)

        if self.publish_goal_pose:
            self.goal_msg.header.stamp = now
            self.goal_pub.publish(self.goal_msg)


def main():
    rclpy.init()
    node = NPZPathToPlan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()