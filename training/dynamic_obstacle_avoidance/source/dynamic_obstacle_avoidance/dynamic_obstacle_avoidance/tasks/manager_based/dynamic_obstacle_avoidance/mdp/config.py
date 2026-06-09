

domain_randomization_stages: list = [
  'lidar_rays',
  'scan_noise',
  'scan_dropout',
  'action_delay',
  'motor_strength',
  'mass',
  'com_shift',
  'wheel_radius',
  'wheel_slip',
  'combined_strong',
]

obstacle_stages: list = [
  'side_stationary_tiny',
  'side_stationary_small',
  # 'center_stationary_tiny',
  # 'center_stationary_small',
  # 'center_stationary_medium',
  # 'slow_crossing_far',
  # 'slow_crossing_near',
  # 'medium_crossing_far',
  # 'medium_crossing_near',
  # 'fast_crossing_far',
  # 'same_lane_slow',
  # 'same_lane_medium',
  # 'reverse_same_lane_slow',
  # 'center_stationary_large',
  # 'two_crossing_combo',
]

nav2_path_dataset_dir: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/nav_rl_bridge/rl_path_dataset/aws_warehouse"
nav2_map_yaml_path: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/m3_ros2/maps/no_roof_warehouse.yaml"

ACTIONS: dict = {
    'wheel_joint_names': [
        "lwheel1_Joint",
        "lwheel2_Joint",
        "rwheel1_Joint",
        "rwheel2_Joint",
    ],
    'wheel_radius': 0.035,
    'wheel_base_x': 0.0795,
    'wheel_base_y': 0.09775,
    'max_vx': 0.75,
    'max_vy': 0.75,
    'max_wz': 2.0,
    'max_delta_vx': 0.04,
    'max_delta_vy': 0.04,
    'max_delta_wz': 0.12
}

OBSERVATIONS: dict = {
    'actor': {
        'local_path_window': {
            'params': {'num_points': 8, 'step': 8,},
        },
        'nav2_heading_error': {},
        'nav2_cross_track_error': {},
        'combined_scan': {
            'params': {'num_rays': 144, 'max_range': 4.0, 'step_size': 0.10,},
        },
        'dynamic_obstacles': {
            'params': {"num_obstacles": 4, "max_range": 4.0},
        },
        'path_blocked': {
            'params': {"lookahead_points": 32, "path_radius": 0.35},
        },
        'time_to_closest_approach': {
            'params': {"num_obstacles": 2, "max_range": 4.0, "horizon_s": 3.0},
        },
        'base_lin_vel': {},
        'base_angle_vel': {},
        'previous_action': {},

    },

    'critic': {
        'local_path_window': {
            'params': {'num_points': 8, 'step': 8,},
        },
        'nav2_heading_error': {},
        'nav2_cross_track_error': {},
        'combined_scan': {
            'params': {'num_rays': 144, 'max_range': 4.0, 'step_size': 0.10,},
        },
        'dynamic_obstacles': {
            'params': {"num_obstacles": 4, "max_range": 4.0},
        },
        'path_blocked': {
            'params': {"lookahead_points": 32, "path_radius": 0.35},
        },
        'time_to_closest_approach': {
            'params': {"num_obstacles": 2, "max_range": 4.0, "horizon_s": 3.0},
        },
        'base_lin_vel': {},
        'base_angle_vel': {},
        'previous_action': {},
        'distance_to_goal': {},
        'progress_fraction': {},
        'map_collision': {},
        'dynamic_collision': {},

    },
}

EVENTS: dict = {
    'reset_nav2_path': {'max_path_points': 600,},
    'reset_dynamic_obstacles': {'max_path_points': 600,},
    'update_dynamic_obstacles': {'interval_range_s': (0.03, 0.03),},
    'draw_nav2_debug': {
        'interval_range_s': (0.03, 0.03,),
        'params': {
            "map_stride": 2,
            "max_map_points": 6000,
            "path_stride": 4,
            "num_rays": 72,
            "max_range": 4.0,
            "step_size": 0.05,
        },
    },
    'log_curriculum_progress': {'interval_range_s': (1.0, 1.0), },
    'log_map_collision_directions': {'interval_range_s': (1.0, 1.0), },
}

REWARDS: dict = {
    'progress': {
        'weight': 35.0,
        'max_step_progress': 0.05,
    },
    'goal_approach':{
        'weight': 15.0,
        'max_step_progress': 0.08,
    },
    'cross_track': {
        'weight': -1.5,
        'max_error': 1.0,
    },
    'path_rejoin': {
        'weight': 5.0,
        'active_threshold': 0.2,
    },
    'heading_alignment': {
        'weight': 8.0,
        'lookahead_index_offset': 4,
    },
    'dynamic_collision': {
        'weight': -100.0,
        'robot_radius': 0.22,
    },
    'dynamic_clearance': {
        'weight': -5.0,
        'robot_radius': 0.22,
        'clearance': 0.25,
    },
    'dynamic_ttc': {
        'weight': -1.0,
        'robot_radius': 0.22,
        'horizon_s': 1.0,
    },
    'lateral_oscillation': {
        'weight': -0.35,

    },
    'map_collision': {
        'weight': -150.0,
        'radius': 0.22,
    },
    'final_goal': {
        'weight': 120.0,
        'threshold': 0.30,
    },
    'action_smoothness': {
        'weight': -0.06,

    },
    'yaw_rate': {
        'weight': -0.05,

    },
    'path_velocity': {
        'weight': 6.0,

    },
    'time': {
        'weight': -0.06,
    },
    'no_wait': {
        'weight': -5.0,
        'speed_threshold': 0.10,
    },
    'static_velocity_clearance': {
        'weight': -5.0,
        'safe_distance': 0.30,
        'max_range': 4.0,
        'num_rays': 144,
        'sector_half_angle_rad': 0.785398,
        'min_speed': 0.05,
    }
}

TERMINATIONS: dict = {
    'final_goal_reached': {
        'threshold': 0.3,
    },
    'map_collision': {
        'radius': 0.22,
    },
    'dynamic_collision': {
        'robot_radius': 0.22,
    },
    'stuck': {
        'speed_threshold': 0.02,
        'time_window_s': 2.0,
        'grace_period_s': 2.0,
    }
}