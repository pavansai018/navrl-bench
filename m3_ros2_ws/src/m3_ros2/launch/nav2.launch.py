import os
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('m3_ros2')
    # command = "source /home/pavan/Downloads/SUTD/Project/VCN/tbot_ws/install/setup.bash"

    # subprocess.run(command, shell=True, executable='/bin/bash')
    carbase_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'car_base.launch.py'))
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory('m3_ros2'),
                    'launch',
                    'navigation_bringup_launch.py'
                )
            ]
        ),
        launch_arguments={
            'map': os.path.join(pkg_share, 'maps', 'no_roof_warehouse.yaml'),
            'params_file': os.path.join(pkg_share, 'config', 'M3_nav2_params.yaml'),
            'use_sim_time': 'True',
            'slam': 'False',
        }.items(),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'nav2.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    ld = LaunchDescription()
    ld.add_action(carbase_launch)
    ld.add_action(
        TimerAction(
            period=5.0,
            actions=[nav2_launch]
        )
    )

    ld.add_action(
        TimerAction(
            period=7.0,
            actions=[rviz_node]
        )
    )
    # ld.add_action(nav2_launch)
    # ld.add_action(rviz_node)
    return ld