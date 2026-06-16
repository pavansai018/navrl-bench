from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument("map"),
        DeclareLaunchArgument("params_file"),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_map_to_base_link",
            arguments=[
                "0", "0", "0",
                "0", "0", "0",
                "map",
                "base_link",
            ],
            output="screen",
        ),

        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[
                params_file,
                {"yaml_filename": map_yaml},
            ],
        ),

        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[params_file],
        ),

        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_planning",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart": True,
                "node_names": [
                    "map_server",
                    "planner_server",
                ],
            }],
        ),
    ])