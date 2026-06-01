
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription,LogInfo,TimerAction,ExecuteProcess
from launch.actions import DeclareLaunchArgument,GroupAction,IncludeLaunchDescription,LogInfo,RegisterEventHandler
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from colorama import Fore,init
init(autoreset=True) 

def generate_launch_description():


    #激光雷达融合和过滤启动文件Laser radar fusion and filtering startup file
    bringup_laser_launch=IncludeLaunchDescription(PathJoinSubstitution([FindPackageShare('m3_ros2'),'launch', 'bringup_laser.launch.py']))
    #多里程计融合定位启动文件Multi-mileage fusion positioning startup file
    merge_odom_launch=IncludeLaunchDescription(PathJoinSubstitution([FindPackageShare('m3_ros2'),'launch','merge_odom.launch.py']))
    

    return LaunchDescription([
            bringup_laser_launch,      # 激光雷达融合和过滤启动文件Laser radar fusion and filtering startup file
            TimerAction(
            period=2.0,
            actions=[
                merge_odom_launch,        
            ]
         ),
            LogInfo(msg=[Fore.GREEN+'================================Robot Base Start================================',Fore.RESET]),
    ])
