from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="m3_ros2",
            executable="freespace_supervisor",
            name="freespace_supervisor",
            output="screen",
            parameters=[{
                "mppi_cmd_topic": "/cmd_vel_mppi",
                "output_cmd_topic": "/cmd_vel",
                "costmap_topic": "/local_costmap/costmap",
                "odom_topic": "/odom",

                "robot_radius": 0.24,
                "safety_margin": 0.08,

                "trigger_x_max": 1.10,
                "trigger_y_abs": 0.38,

                "candidate_offsets":
                    "0.00,0.45;0.00,-0.45;"
                    "0.20,0.55;0.20,-0.55;"
                    "0.45,0.55;0.45,-0.55",

                "max_escape_vx": 0.35,
                "max_escape_vy": 0.45,
                "max_escape_wz": 0.8,
            }]
        )
    ])