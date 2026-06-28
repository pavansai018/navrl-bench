# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
from . import mdp
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import actions as custom_actions
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import rewards as custom_rewards
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import events as custom_events
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import terminations as custom_terminations
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import observations as custom_observations
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp import config as config

##
# Pre-defined configs
##

from dynamic_obstacle_avoidance.assets.m3 import M3_CFG

##
# Scene definition
##


@configclass
class DynamicObstacleAvoidanceSceneCfg(InteractiveSceneCfg):
    """Configuration for an obstacle avoidance scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot will be injected from another config file
    robot: ArticulationCfg = MISSING


    # Final goal marker: visual only, not an obstacle
    final_goal_marker = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/FinalGoalMarker',
        spawn=sim_utils.SphereCfg(
            radius=0.16,

            # Kinematic object so we can move it during reset/update.
            # It should not be controlled by physics.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),

            # IMPORTANT:
            # Do not add collision_props here.
            # This marker must not behave like an obstacle.
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.9, 0.25),
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16),
        ),
    )



    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    Policy action should be base velocity:
    action=[vx, vy, wz]

    The custom MDP action term must convert this into mecanum wheel velocities

    """

    base_velocity = custom_actions.KinematicMecanumActionCfg(
        asset_name="robot",
        wheel_joint_names=config.ACTIONS['wheel_joint_names'],
        wheel_radius=config.ACTIONS['wheel_radius'],
        wheel_base_x=config.ACTIONS['wheel_base_x'],
        wheel_base_y=config.ACTIONS['wheel_base_y'],
        max_vx=config.ACTIONS['max_vx'],
        max_vy=config.ACTIONS['max_vy'],
        max_wz=config.ACTIONS['max_wz'],
        max_delta_vx=config.ACTIONS['max_delta_vx'],
        max_delta_vy=config.ACTIONS['max_delta_vy'],
        max_delta_wz=config.ACTIONS['max_delta_wz'],
    )

@configclass
class ObservationsCfg:
    """Observation specifications for dynamic obstacle avoidance."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        Observation should describe local navigation, not full global navigation.
        """
        local_path_window = ObsTerm(
            func=custom_observations.local_path_window,
            params=config.OBSERVATIONS['actor']['local_path_window']['params']
        )

        nav2_heading_error = ObsTerm(
            func=custom_observations.nav2_path_heading_error,
        )
        nav2_cross_track_error = ObsTerm(
            func=custom_observations.nav2_cross_track_error,
        )
        # combined_scan = ObsTerm(
        #     func=custom_observations.combined_static_dynamic_scan,
        #     params=config.OBSERVATIONS['actor']['combined_scan']['params'],
        # )
        scan_history = ObsTerm(
            func=custom_observations.scan_history,
            params=config.OBSERVATIONS['actor']['scan_history']['params'],
        )
        # dynamic_obstacles = ObsTerm(
        #     func=custom_observations.dynamic_obstacle_states,
        #     params=config.OBSERVATIONS['actor']['dynamic_obstacles']['params'],
        # )

        # path_blocked = ObsTerm(
        #     func=custom_observations.dynamic_path_blockage,
        #     params=config.OBSERVATIONS['actor']['path_blocked']['params'],
        # )

        # time_to_closest_approach = ObsTerm(
        #     func=custom_observations.time_to_closest_approach,
        #     params=config.OBSERVATIONS['actor']['time_to_closest_approach']['params'],
        # )

        # Robot motion
        base_lin_vel = ObsTerm(func=custom_observations.base_lin_vel)
        base_angle_vel = ObsTerm(func=custom_observations.base_ang_vel)
        # Previous action for smoother policy behavior
        previous_action = ObsTerm(func=custom_observations.previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
        
    @configclass
    class CriticCfg(ObsGroup):
        local_path_window = ObsTerm(
            func=custom_observations.local_path_window,
            params=config.OBSERVATIONS['critic']['local_path_window']['params']
        )

        nav2_heading_error = ObsTerm(
            func=custom_observations.nav2_path_heading_error,
        )
        nav2_cross_track_error = ObsTerm(
            func=custom_observations.nav2_cross_track_error,
        )

        # combined_scan = ObsTerm(
        #     func=custom_observations.combined_static_dynamic_scan,
        #     params=config.OBSERVATIONS['critic']['combined_scan']['params'],
        # )

        scan_history = ObsTerm(
            func=custom_observations.scan_history,
            params=config.OBSERVATIONS['critic']['scan_history']['params'],
        )

        dynamic_obstacles = ObsTerm(
            func=custom_observations.dynamic_obstacle_states,
            params=config.OBSERVATIONS['critic']['dynamic_obstacles']['params'],
        )
        # path_blocked = ObsTerm(
        #     func=custom_observations.dynamic_path_blockage,
        #     params=config.OBSERVATIONS['critic']['path_blocked']['params'],
        # )
        path_blocked = ObsTerm(
            func=custom_observations.path_blocked_now_or_future_1s,
            params=config.OBSERVATIONS['critic']['path_blocked_now_or_future_1s']['params'],
        )

        time_to_closest_approach = ObsTerm(
            func=custom_observations.time_to_closest_approach,
            params=config.OBSERVATIONS['critic']['time_to_closest_approach']['params'],
        )

        base_lin_vel = ObsTerm(func=custom_observations.base_lin_vel)
        base_ang_vel = ObsTerm(func=custom_observations.base_ang_vel)
        previous_actions = ObsTerm(func=custom_observations.previous_action)

        distance_to_goal = ObsTerm(func=custom_observations.distance_to_final_goal)
        progress_fraction = ObsTerm(func=custom_observations.nav2_path_progress_fraction)
        map_collision = ObsTerm(func=custom_observations.map_collision_observation)
        dynamic_collision = ObsTerm(func=custom_observations.dynamic_collision_observation)

        # future_path_blocked_1s = ObsTerm(
        #     func=custom_observations.future_path_blocked_1s,
        #     params=config.OBSERVATIONS['critic']['future_path_blocked_1s']['params'],
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_nav2_path = EventTerm(
        func=custom_events.reset_nav2_path_and_debug_validate,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "final_goal_marker_cfg": SceneEntityCfg("final_goal_marker"),
            "max_path_points": config.EVENTS['reset_nav2_path']['max_path_points'],
        },
    )

    reset_dynamic_obstacles = EventTerm(
        func=custom_events.reset_dynamic_obstacles_tensor,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_path_points": config.EVENTS['reset_dynamic_obstacles']['max_path_points'],
        },
    )

    update_dynamic_obstacles = EventTerm(
        func=custom_events.update_dynamic_obstacles_tensor,
        mode="interval",
        interval_range_s=config.EVENTS['update_dynamic_obstacles']['interval_range_s'],
    )

    # draw_nav2_debug = EventTerm(
    #     func=custom_events.draw_nav2_map_path_scan_debug,
    #     mode="interval",
    #     interval_range_s=(0.03, 0.030),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"), 
    #         'map_stride': config.EVENTS['draw_nav2_debug']['params']['map_stride'],
    #         "max_map_points": config.EVENTS['draw_nav2_debug']['params']['max_map_points'],
    #         "path_stride": config.EVENTS['draw_nav2_debug']['params']['path_stride'],
    #         "num_rays": config.EVENTS['draw_nav2_debug']['params']['num_rays'],
    #         "max_range": config.EVENTS['draw_nav2_debug']['params']['max_range'],
    #         "step_size": config.EVENTS['draw_nav2_debug']['params']['step_size'],
    #     }
    # )

    reset_stuck_buffers = EventTerm(
        func=custom_terminations.reset_stuck_buffers,
        mode="reset",
    )

    # print_training_metrics = EventTerm(
    #     func=custom_events.print_curriculum_training_metrics,
    #     mode="interval",
    #     interval_range_s=(10.0, 12.0),
    # )

    # print_dr_metrics = EventTerm(
    #     func=custom_events.print_domain_randomization_metrics,
    #     mode="interval",
    #     interval_range_s=(10.0, 12.0),
    # )

    log_curriculum_progress = EventTerm(
        func=custom_events.log_curriculum_progress,
        mode="interval",
        interval_range_s=config.EVENTS['log_curriculum_progress']['interval_range_s'],
    )

    log_map_collision_directions = EventTerm(
        func=custom_events.log_map_collision_directions,
        mode="interval",
        interval_range_s=config.EVENTS['log_map_collision_directions']['interval_range_s'],
    )

@configclass
class RewardsCfg:
    """
    Task rewards for adaptive dynamic obstacle avoidance.

    No raw vy reward. Mecanum usage is expected to emerge from scenarios where
    lateral/diagonal motion maintains progress while avoiding dynamic obstacles.
    """
    progress = RewTerm(
        func=custom_rewards.progress_along_nav2_path,
        weight=config.REWARDS['progress']['weight'],
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'max_step_progress': config.REWARDS['progress']['max_step_progress'],
        },
    )
    goal_approach = RewTerm(
        func=custom_rewards.goal_approach_reward,
        weight=config.REWARDS['goal_approach']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "max_step_progress": config.REWARDS['goal_approach']['max_step_progress']},
    )

    cross_track = RewTerm(
        func=custom_rewards.nav2_cross_track_penalty,
        weight=config.REWARDS['cross_track']['weight'],
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'max_error': config.REWARDS['cross_track']['max_error'],
        },
    )

    path_rejoin = RewTerm(
        func=custom_rewards.path_rejoin_reward,
        weight=config.REWARDS['path_rejoin']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "active_threshold": config.REWARDS['path_rejoin']['active_threshold']},
    )


    heading_alignment = RewTerm(
        func=custom_rewards.nav2_heading_alignment_reward,
        weight=config.REWARDS['heading_alignment']['weight'],#1.5,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'lookahead_index_offset': config.REWARDS['heading_alignment']['lookahead_index_offset'],
        },
    )

    dynamic_collision = RewTerm(
        func=custom_rewards.dynamic_obstacle_collision_penalty,
        weight=config.REWARDS['dynamic_collision']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": config.REWARDS['dynamic_collision']['robot_radius']},
    )

    dynamic_clearance = RewTerm(
        func=custom_rewards.dynamic_obstacle_clearance_penalty,
        weight=config.REWARDS['dynamic_clearance']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": config.REWARDS['dynamic_clearance']['robot_radius'], "clearance": config.REWARDS['dynamic_clearance']['clearance']},
    )

    dynamic_ttc = RewTerm(
        func=custom_rewards.dynamic_time_to_collision_penalty,
        weight=config.REWARDS['dynamic_ttc']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": config.REWARDS['dynamic_ttc']['robot_radius'], "horizon_s": config.REWARDS['dynamic_ttc']['horizon_s']},
    )

    lateral_oscillation = RewTerm(
        func=custom_rewards.lateral_oscillation_penalty,
        weight=config.REWARDS['lateral_oscillation']['weight'], #-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    map_collision = RewTerm(
        func=custom_rewards.map_collision_penalty,
        weight=config.REWARDS['map_collision']['weight'],
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'radius': config.REWARDS['map_collision']['radius'],
        },
    )

    final_goal = RewTerm(
        func=custom_rewards.final_goal_reward,
        weight=config.REWARDS['final_goal']['weight'],
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'threshold': config.REWARDS['final_goal']['threshold'],
        },
    )

    action_smoothness = RewTerm(
        func=custom_rewards.action_smoothness_penalty,
        weight=config.REWARDS['action_smoothness']['weight'],
    )

    yaw_rate = RewTerm(
        func=custom_rewards.yaw_rate_penalty,
        weight=config.REWARDS['yaw_rate']['weight'], #-0.02,
    )

    path_velocity = RewTerm(
        func=custom_rewards.path_velocity_reward,
        weight=config.REWARDS['path_velocity']['weight'], #3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    time = RewTerm(
        func=custom_rewards.time_penalty,
        weight=config.REWARDS['time']['weight'], #-0.03,
    )

    no_wait = RewTerm(
        func=custom_rewards.no_wait_penalty,
        weight=config.REWARDS['no_wait']['weight'],
        params={"asset_cfg": SceneEntityCfg("robot"), "speed_threshold": config.REWARDS['no_wait']['speed_threshold']},
    )

    static_velocity_clearance = RewTerm(
        func=custom_rewards.static_velocity_clearance_penalty,
        weight=config.REWARDS['static_velocity_clearance']['weight'],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "safe_distance": config.REWARDS['static_velocity_clearance']['safe_distance'],
            "max_range": config.REWARDS['static_velocity_clearance']['max_range'],
            "num_rays": config.REWARDS['static_velocity_clearance']['num_rays'],
            "sector_half_angle_rad": config.REWARDS['static_velocity_clearance']['sector_half_angle_rad'],
            "min_speed": config.REWARDS['static_velocity_clearance']['min_speed'],
        },
    )
    start_speed = RewTerm(
        func=custom_rewards.start_speed_penalty,
        weight=config.REWARDS['start_speed']['weight'],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "warmup_s": config.REWARDS['start_speed']['warmup_s'],
        },
    )

    timeout_failure = RewTerm(
        func=custom_rewards.timeout_penalty,
        weight=config.REWARDS['timeout_failure']['weight'],
    )

    adaptive_future_path_tracking = RewTerm(
        func=custom_rewards.adaptive_future_aware_path_tracking_penalty,
        weight=config.REWARDS['adaptive_future_path_tracking']['weight'],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "horizon_s": config.REWARDS['adaptive_future_path_tracking']['params']['horizon_s'],
            "lookahead_m": config.REWARDS['adaptive_future_path_tracking']['params']['lookahead_m'],
            "corridor_half_width": config.REWARDS['adaptive_future_path_tracking']['params']['corridor_half_width'],
            "robot_radius": config.REWARDS['adaptive_future_path_tracking']['params']['robot_radius'],
            "track_free_cte": config.REWARDS['adaptive_future_path_tracking']['params']['track_free_cte'],
            "track_max_cte": config.REWARDS['adaptive_future_path_tracking']['params']['track_max_cte'],
            "detour_free_cte": config.REWARDS['adaptive_future_path_tracking']['params']['detour_free_cte'],
            "detour_max_cte": config.REWARDS['adaptive_future_path_tracking']['params']['detour_max_cte'],
            "max_lateral_speed": config.ACTIONS['max_vx'],
        },
    )

    # future_clear_no_lateral_bypass = RewTerm(
    #     func=custom_rewards.future_clear_no_lateral_bypass_penalty,
    #     weight=config.REWARDS['future_clear_no_lateral_bypass']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "horizon_s": config.REWARDS['future_clear_no_lateral_bypass']['params']['horizon_s'],
    #         "lookahead_m": config.REWARDS['future_clear_no_lateral_bypass']['params']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['future_clear_no_lateral_bypass']['params']['corridor_half_width'],
    #         "free_cte": config.REWARDS['future_clear_no_lateral_bypass']['params']['free_cte'],
    #         "max_cte": config.REWARDS['future_clear_no_lateral_bypass']['params']['max_cte'],
    #         "max_vy": config.ACTIONS['max_vx'],
    #     },
    # )

    # dynamic_corridor_closing = RewTerm(
    #     func=custom_rewards.dynamic_corridor_closing_penalty,
    #     weight=config.REWARDS['dynamic_corridor_closing']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "lookahead_m": config.REWARDS['dynamic_corridor_closing']['params']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['dynamic_corridor_closing']['params']['corridor_half_width'],
    #         "danger_clearance": config.REWARDS['dynamic_corridor_closing']['params']['danger_clearance'],
    #         "max_closing_speed": config.ACTIONS['max_vx'],
    #         'robot_radius': config.REWARDS['dynamic_corridor_closing']['params']['robot_radius'],
    #     },
    # )

    # gated_detour = RewTerm(
    #     func=custom_rewards.gated_detour_reward,
    #     weight=config.REWARDS['gated_detour']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "danger_clearance": config.REWARDS['gated_detour']['params']['danger_clearance'],
    #         "max_cte": config.REWARDS['gated_detour']['params']['max_cte'],
    #     },
    # )

    # lateral_bypass = RewTerm(
    #     func=custom_rewards.lateral_bypass_reward,
    #     weight=config.REWARDS['lateral_bypass']['weight'],
    #     params={"asset_cfg": SceneEntityCfg("robot"), "max_cte": config.REWARDS['lateral_bypass']['max_cte']},
    # )
    # mppi_imitation = RewTerm(
    #     func=custom_rewards.mppi_teacher_imitation_reward,
    #     weight=config.REWARDS['mppi_imitation']['weight'],
    # )

    # wall_aware_dynamic_bypass = RewTerm(
    #     func=custom_rewards.wall_aware_dynamic_bypass_reward,
    #     weight=config.REWARDS['wall_aware_dynamic_bypass']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "lookahead_m": config.REWARDS['wall_aware_dynamic_bypass']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['wall_aware_dynamic_bypass']['corridor_half_width'],
    #         "min_side_clearance": config.REWARDS['wall_aware_dynamic_bypass']['min_side_clearance'],
    #         "max_lateral_speed": config.ACTIONS['max_vy'],
    #     },
    # )

    # dynamic_yield_no_side_space = RewTerm(
    #     func=custom_rewards.dynamic_yield_when_no_side_space_reward,
    #     weight=config.REWARDS['dynamic_yield_no_side_space']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "lookahead_m": config.REWARDS['dynamic_yield_no_side_space']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['dynamic_yield_no_side_space']['corridor_half_width'],
    #         "min_side_clearance": config.REWARDS['dynamic_yield_no_side_space']['min_side_clearance'],
    #         "max_vx": config.ACTIONS['max_vx'],
    #         "max_vy": config.ACTIONS['max_vy'],
    #     },
    # )

    # dynamic_forward_blocked = RewTerm(
    #     func=custom_rewards.dynamic_forward_blocked_penalty,
    #     weight=config.REWARDS['dynamic_forward_blocked']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "lookahead_m": config.REWARDS['dynamic_forward_blocked']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['dynamic_forward_blocked']['corridor_half_width'],
    #         "max_vx": config.ACTIONS['max_vx'],
    #     },
    # )

    # dynamic_bad_lateral = RewTerm(
    #     func=custom_rewards.dynamic_bad_lateral_penalty,
    #     weight=config.REWARDS['dynamic_bad_lateral']['weight'],
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "lookahead_m": config.REWARDS['dynamic_bad_lateral']['lookahead_m'],
    #         "corridor_half_width": config.REWARDS['dynamic_bad_lateral']['corridor_half_width'],
    #         "min_side_clearance": config.REWARDS['dynamic_bad_lateral']['min_side_clearance'],
    #         "max_vy": config.ACTIONS['max_vy'],
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Goal reached
    final_goal_reached = DoneTerm(
        func=custom_terminations.final_goal_reached,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'threshold': config.TERMINATIONS['final_goal_reached']['threshold'],
        },
    )

    map_collision = DoneTerm(
        func=custom_terminations.map_collision_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "radius": config.TERMINATIONS['map_collision']['radius'],
        },
    )

    dynamic_collision = DoneTerm(
        func=custom_terminations.dynamic_obstacle_collision_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": config.TERMINATIONS['dynamic_collision']['robot_radius']},
    )

    stuck = DoneTerm(
        func=custom_terminations.stuck_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_threshold": config.TERMINATIONS['stuck']['speed_threshold'],
            "time_window_s": config.TERMINATIONS['stuck']['time_window_s'],
            "grace_period_s": config.TERMINATIONS['stuck']['grace_period_s'],
        },
    )

##
# Environment configuration
##


@configclass
class DynamicObstacleAvoidanceEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DynamicObstacleAvoidanceSceneCfg = DynamicObstacleAvoidanceSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    nav2_path_dataset_dir: str = config.nav2_path_dataset_dir
    nav2_map_yaml_path: str = config.nav2_map_yaml_path
    mppi_teacher: dict = config.MPPI_TEACHER

    # Tensor dynamic obstacle/curriculum configuration.
    max_dynamic_obstacles: int = 20
    dynamic_obstacle_radius_min: float = 0.18
    dynamic_obstacle_radius_max: float = 0.32
    dynamic_obstacle_deactivate_range: float = 6.0

    # Small robustness noise only; dynamic obstacles create the actual avoidance problem.
    reset_lateral_noise_m: float = 0.0 #0.10
    reset_yaw_noise_rad: float = 0.05 #0.10
    path_window_normalization_m: float = 4.0

    # Debug controls.
    debug_validate_nav2_path: bool = False
    debug_draw_nav2: bool = False
    debug_draw_lidar: bool = False
    debug_draw_map: bool = False
    debug_draw_path: bool = False
    debug_draw_dynamic_obstacles: bool = False
    debug_draw_max_envs: int = 4

    # -----------------------------
    # Domain Randomization V1
    # -----------------------------
    dr_enable: bool = True

    com_z_m: float = 0.005


    curriculum_max_level: int = -1 #len(custom_events.read_config()['domain_randomization_stages'] + custom_events.read_config()['obstacle_stages'])
    fixed_curriculum_level: int = -1
    curriculum_perf_window: int = 100000
    curriculum_min_samples: int = 90000
    sample_curriculum_levels: bool = True
    curriculum_sample_min_level: int = 0
    curriculum_sample_max_level: int = 10

    curriculum_success_promote: float = 0.70
    curriculum_map_collision_promote: float = 0.20
    curriculum_dynamic_collision_promote: float = 0.20
    curriculum_timeout_promote: float = 0.15

    curriculum_success_demote: float = 0.10
    curriculum_map_collision_demote: float = 0.50
    curriculum_dynamic_collision_demote: float = 0.5
    curriculum_timeout_demote: float = 0.50

    enable_static_action_shield: bool = True
    shield_robot_radius: float = 0.22
    shield_num_points: int = 16

    curriculum_order: str = "obstacles_first"  # "obstacles_first" or "dr_first"

    # DR logging
    dr_log_every_steps: int = 5000
    curriculum_log_every_steps: int = 5000
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # PhysX settings
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 2
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
