import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,LogInfo
from launch.substitutions import LaunchConfiguration,Command
from launch_ros.actions import Node
from launch.conditions import IfCondition,UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from colorama import Fore,init
init(autoreset=True) 

def generate_launch_description():
    LASER_TYPE = 'dual' # os.environ['LASER_TYPE'] # get lidar type
    CAMERA_TYPE = 'as_hp60c' # os.environ['CAMERA_TYPE'] # get camera type



    # 声明变量Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "gui",
            default_value="false",
            description="Start Rviz2 and Joint State Publisher gui automatically",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value="",
            description="multi-robot setup",
        )
    )
    
    # 初始化变量Initialize Arguments
    gui = LaunchConfiguration("gui")
    prefix = LaunchConfiguration("prefix")

    rviz_config_file = os.path.join(get_package_share_directory('m3_ros2'),'rviz','display.rviz')
    pkg_share = get_package_share_directory('m3_ros2')
    xacro_file = os.path.join(pkg_share, 'urdf', 'ROSMASTER-M3.xacro')

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
    
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(gui),
    )
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        condition=UnlessCondition(gui),
    )
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description_content}],
    )
    rviz_ndoe=  Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        condition=IfCondition(gui),
    )
    log=LogInfo(
        msg=[Fore.GREEN+'====================Robot model ROSMASTER-M3 start====================\n',
                f'                      ==========================LASER_TYPE:{LASER_TYPE}================================\n',
             Fore.RESET])

    nodes=[
            log,
            joint_state_publisher_gui_node,
            joint_state_publisher_node,
            robot_state_publisher_node,
            rviz_ndoe,
        ]

    return LaunchDescription(declared_arguments+nodes)


