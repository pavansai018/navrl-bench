import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription,LogInfo,TimerAction
from launch_ros.actions import  Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration,PythonExpression,PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument,LogInfo
from colorama import Fore,init
init(autoreset=True) 

def generate_launch_description():
    map_folder_dir = os.path.join(os.path.expanduser('~'), 'map')
    # 声明变量Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
        "gui",
        default_value='True',
        description="Start Rviz2 automatically ?",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
                "slam_mode",
                default_value="async", # originally author used sync
                description="建图模式选择:sync/async",
        )  
    )
    declared_arguments.append(
        DeclareLaunchArgument(
                "pose_graph",
                default_value="",
                description="传入的位姿图,可以进行继续建图，如果为空，则重新开始建图\
                            The passed pose graph can be continued to build. If it is empty, it will be restarted.",
        )  
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # 初始化变量Initialize Arguments
    gui=LaunchConfiguration('gui')
    slam_mode=LaunchConfiguration('slam_mode')
    pose_graph=PathJoinSubstitution([map_folder_dir, LaunchConfiguration("pose_graph")])
    configdir=os.path.join(get_package_share_directory('m3_ros2'),'config','slam_toolbox_config')
    sync_slam_params_file=os.path.join(configdir, 'mapper_params_online_sync.yaml')
    async_slam_params_file=os.path.join(configdir, "mapper_params_online_async.yaml")

    carbase_launch=IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("m3_ros2"),
                                        'launch',
                                        'car_base.launch.py')))

    
    #同步模式建图节点Synchronous mode mapping node
    sync_slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        parameters=[sync_slam_params_file,
                    {'map_file_name': pose_graph},
                    {'use_sim_time': use_sim_time},
                    ],
        name='sync_slam_toolbox',
        output='screen',
        condition=IfCondition(PythonExpression(["'", slam_mode, "' == 'sync'"]))
        )
    
    #异步模式建图节点Asynchronous mode map building node
    async_slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        parameters=[async_slam_params_file,
                    {'map_file_name': pose_graph},
                    {'use_sim_time': use_sim_time},
                    ],
        name='async_slam_toolbox',
        output='screen',
        condition=IfCondition(PythonExpression(["'", slam_mode, "' == 'async'"]))
        )
    rviz_node=Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('m3_ros2'), 'rviz', 'slam_toolbox_rviz.rviz')],
        condition=IfCondition(gui),
    )

    return LaunchDescription([
            *declared_arguments,            # 声明变量
            declare_use_sim_time_cmd,
            rviz_node,                      #rviz
            carbase_launch,                 #carbase启动文件
            # sync_slam_toolbox_node,         #同步模式建图节点
            # async_slam_toolbox_node,        #异步模式建图节点
            TimerAction(
                period=5.0,           # Wait for Gazebo + sensors
                actions=[
                    sync_slam_toolbox_node,
                    async_slam_toolbox_node,
                ]
            ),
            LogInfo(msg=[Fore.GREEN+'================================SLAM Toolbox Start================================',Fore.RESET])
    ])
