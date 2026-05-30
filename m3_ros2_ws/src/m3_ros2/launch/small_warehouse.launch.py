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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    aws_small_warehouse_dir = get_package_share_directory('m3_ros2')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    current_model_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    local_model_path = os.path.join(get_package_share_directory('m3_ros2'), 'models')
    # local_model_path = get_package_share_directory('m3_ros2')

    set_gazebo_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            local_model_path,
             ':',
             current_model_path
            ]
    )
    # Include the gz sim launch file  
    gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(aws_small_warehouse_dir, 'worlds', 'small_warehouse', 'small_warehouse.world')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_sim_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args' :  f'{world_file}' #'-r empty.sdf'
        }.items()
    )



    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(set_gazebo_model_path)
    ld.add_action(gz_sim)

    return ld