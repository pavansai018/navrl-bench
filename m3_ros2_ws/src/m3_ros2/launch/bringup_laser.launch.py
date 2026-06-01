import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,GroupAction,IncludeLaunchDescription,LogInfo,TimerAction
from launch_ros.actions import  Node, PushRosNamespace,ComposableNodeContainer
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration,PythonExpression
from launch_ros.descriptions import ComposableNode
from colorama import Fore


def generate_launch_description()->LaunchDescription:
    LASER_TYPE = 'dual' #os.environ.get('LASER_TYPE', 'single')     

    # 根据LASER_TYPE设置不同的激光雷达话题Different lidar topics are set according to LASER_TYPE.
    if LASER_TYPE == 'dual':
        laserscan_topics = '/scan0 /scan1'
        multi_frame='lasersim_frame'
    elif LASER_TYPE == 'single':
        laserscan_topics = '/scan2 /scan3'
        multi_frame='laser23_frame'
    else:
        raise ValueError(f"LASER_TYPE must be one of 'single' or 'dual', but got '{LASER_TYPE}'")

    # 声明变量Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Add namespace添加命名空间",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
                'use_composition',
                default_value="False",
                description='Whether to use composed bringup是否使用容器的方式启动',
            )
    )

    # 初始化变量Initialize Arguments 
    namespace=LaunchConfiguration("namespace")
    use_composition=LaunchConfiguration("use_composition")
    config_file = os.path.join(get_package_share_directory('m3_ros2'), 'config', 'common_param.yaml')

    #话题重映射Topic Remapping
    remappings=[('/scan_filtered', [namespace, '/scan']), #滤波后的激光雷达话题The topic of LiDAR after filtering
                ('/scan', [namespace, '/scan_multi'])]    #滤波前激光雷达话题LiDAR topic before filtering
    
    #机器人状态发布节点，雷达融合和过滤必须先发布机器人模型tf状态Robot state publishing node, radar fusion and filtering must first publish the robot model tf state
    robot_display=IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('m3_ros2'),
            'launch',
            'rviz.display.launch.py'
            )),
        launch_arguments={
            'prefix': namespace,
            'gui': 'False'  #仅限用户单独调试时设置为True，一般情况下设置为False Set to True only when the user is debugging alone; otherwise, set to False.
        }.items(),
    )
    
    #独立方式启动雷达处理节点Start the radar processing node in standalone mode
    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(['not ', use_composition])),
        actions=[
            # 激光雷达数据融合LiDAR data fusion
            PushRosNamespace(namespace=namespace),
            Node(
                package='laser_merge',
                executable='lasers_merger',
                name='laserscan_multi_merger',
                parameters=[config_file,
                            {'laserscan_topics': laserscan_topics},
                            {'destination_frame': multi_frame}
                           ],
            ),
            # 激光雷达数据过滤LiDAR data filtering
            Node(
                package="laser_filters",
                executable="scan_to_scan_filter_chain",
                name='laser_filter',
                parameters=[config_file],
                remappings=remappings
            ),
            #日志打印Log Printing
            LogInfo(
                    msg=[Fore.GREEN+'LASER start sucefully!,Start by independent node',Fore.RESET]
                )
        ]
    )

    #以容器方式启动雷达处理节点Start the radar processing node in a container manner
    load_composable_nodes=GroupAction(
        condition=IfCondition(use_composition),
        actions=[
            # PushRosNamespace(namespace=namespace),
            ComposableNodeContainer(
                    name='laser_process_container',
                    package='rclcpp_components',
                    namespace=namespace,
                    executable='component_container_isolated',
                    composable_node_descriptions=[
                        # 激光雷达数据融合LiDAR data fusion
                        ComposableNode(
                            package='laser_merge',
                            plugin='laserscanmerger::LaserscanMerger',
                            name='laserscan_multi_merger',
                            parameters=[config_file,
                                        {'laserscan_topics': laserscan_topics},
                                        {'destination_frame': multi_frame}
                                    ],
                            extra_arguments=[{'use_intra_process_comms': True}],
                        ),
                        # 激光雷达数据过滤LiDAR data filtering
                        ComposableNode(
                            package='laser_filters',
                            plugin='ScanToScanFilterChain',
                            name='laser_filter',
                            parameters=[config_file],
                            remappings=remappings,
                            extra_arguments=[{'use_intra_process_comms': True}],
                        ),
                    ]
                ),

        #日志打印Log Printing
        LogInfo(msg=[Fore.GREEN+'LASER Start Sucefully!,Start Via Container',Fore.RESET])
        ]
    )


    delay_start_composable_nodes= TimerAction(
        period=3.0,
        actions=[load_composable_nodes]
     )

    ld = []
    ld.extend(declared_arguments)
    ld.append(robot_display)
    ld.append(load_nodes) 
    ld.append(delay_start_composable_nodes)

    return LaunchDescription(ld)