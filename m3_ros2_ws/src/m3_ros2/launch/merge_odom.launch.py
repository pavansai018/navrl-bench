import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,LogInfo
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from colorama import Fore,init
init(autoreset=True) 

def generate_launch_description()->LaunchDescription:
    localization_config_file=os.path.join(get_package_share_directory('m3_ros2'), 'config', 'robot_localization.yaml')
  
    # 声明变量Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Add namespace添加命名空间",
        )
    )

    # 初始化变量Initialize Arguments 
    namespace=LaunchConfiguration("namespace")

    return LaunchDescription([
        *declared_arguments,
        LogInfo(
                msg=[Fore.GREEN+'====================Merge_Odom Localization Start====================\n',Fore.RESET]),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            namespace=namespace,
            output='screen',
            parameters=[localization_config_file, {'use_sim_time': True}],
            remappings=[('/odometry/filtered','/odom')]
           ),
])

