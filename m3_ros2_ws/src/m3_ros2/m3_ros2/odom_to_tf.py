#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTF(Node):
    def __init__(self):
        super().__init__("odom_to_tf")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link")

        self.odom_topic = self.get_parameter("odom_topic").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value

        self.br = TransformBroadcaster(self)

        self.sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20,
        )

        self.get_logger().info(
            f"Publishing TF from {self.odom_topic}: {self.parent_frame} -> {self.child_frame}"
        )

    def odom_callback(self, msg):
        tf = TransformStamped()

        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = msg.header.frame_id or self.parent_frame
        tf.child_frame_id = msg.child_frame_id or self.child_frame

        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z

        tf.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(tf)


def main():
    rclpy.init()
    node = OdomToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()