import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(get_package_share_directory('laser_merge'), 'config', 'laserscan_merge.yaml')
    return LaunchDescription([
            Node(
                package='laser_merge',
                executable='lasers_merger',
                name='laserscan_multi_merger',
                output='screen',
                parameters=[config_file],
            ),
    ])


