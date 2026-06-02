# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    LASER_TYPE = 'dual' # os.environ['LASER_TYPE'] # get lidar type
    CAMERA_TYPE = 'as_hp60c' # os.environ['CAMERA_TYPE'] # get camera type
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_prefix_cmd = DeclareLaunchArgument(
            'prefix',
            default_value='',
            description='multi-robot setup',
        )
    
    # Get the launch directory
    aws_small_warehouse_dir = get_package_share_directory('m3_ros2')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    current_model_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    local_model_path = os.path.join(get_package_share_directory('m3_ros2'), 'models')
    # local_model_path = get_package_share_directory('m3_ros2')
    pkg_share = get_package_share_directory('m3_ros2')
    pkg_share_parent = os.path.dirname(pkg_share)
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            local_model_path,
             ':',
             current_model_path,
             ':',
             pkg_share_parent
            ]
    )
    prefix = LaunchConfiguration('prefix')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('m3_ros2'), 'urdf', 'ROSMASTER-M3.xacro']
            ),
            ' ',
            'prefix:=',prefix,
            ' ',
            'laser_type:=',LASER_TYPE,
            ' ',
            'camera_type:=',CAMERA_TYPE
        ]
    )
    # Include the gz sim launch file  
    gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(aws_small_warehouse_dir, 'worlds', 'no_roof_small_warehouse', 'no_roof_small_warehouse.world')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_sim_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args' :  f'-r {world_file}' #'-r empty.sdf'
        }.items()
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "/robot_description",
            "-name", "m3",
            "-allow_renaming", "true",
            "-x", "1.0",
            "-y", "1.0",
            "-z", "0.1",
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description_content, 'use_sim_time': True,}],
    )

    # Wrap robot_state_publisher in a TimerAction
    delayed_rsp = TimerAction(
        period=3.0,   # Wait for Gazebo to publish /clock first
        actions=[robot_state_publisher_node]
    )

    # rviz_config_file= os.path.join(pkg_share, 'rviz', 'laser_view.rviz')
    # bringup_laser_launch=ExecuteProcess(
    #             cmd=['ros2', 'launch', 'm3_ros2','bringup_laser.launch.py'],
    #             output='screen'
    #         )
    bringup_slam_launch=ExecuteProcess(
                cmd=['ros2', 'launch', 'm3_ros2','slam_toolbox_vanilla.launch.py'],
                output='screen'
            )
    bringup_nav2_launch=ExecuteProcess(
            cmd=['ros2', 'launch', 'm3_ros2','nav2.launch.py'],
            output='screen'
        )
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', rviz_config_file],
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}],
    # )
    # spawner_joint_state = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     arguments=['joint_state_broadcaster'],
    #     parameters=[{'use_sim_time': use_sim_time}],
    # )

    # spawner_mecanum = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     arguments=['mecanum_drive_controller'],
    #     parameters=[{'use_sim_time': use_sim_time}],
    # )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            # ROS -> GZ
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            # GZ -> ROS
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            # "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            # "/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/scan0@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan1@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/imu/data_raw@sensor_msgs/msg/Imu[gz.msgs.IMU",
            '/camera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        ],
        parameters=[
            {'use_sim_time': use_sim_time},
            ],
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_prefix_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(set_gazebo_model_path)
    ld.add_action(gz_sim)
    # ld.add_action(robot_state_publisher_node)
    ld.add_action(delayed_rsp)
    ld.add_action(gz_spawn_entity)
    ld.add_action(gz_ros2_bridge)
    # ld.add_action(bringup_laser_launch)
    # ld.add_action(bringup_slam_launch)
    ld.add_action(bringup_nav2_launch)
    # ld.add_action(rviz_node)
    # ld.add_action(spawner_joint_state)
    # ld.add_action(spawner_mecanum)

    return ld