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
        wheel_joint_names=[
            "lwheel1_Joint",
            "lwheel2_Joint",
            "rwheel1_Joint",
            "rwheel2_Joint",
        ],
        wheel_radius=0.035,
        wheel_base_x=0.0795,
        wheel_base_y=0.09775,
        max_vx=0.75,
        max_vy=0.75,
        max_wz=2.0,
        max_delta_vx=0.04,
        max_delta_vy=0.04,
        max_delta_wz=0.12,
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
            params={
                'num_points': 8,
                'step': 8,
            },
        )

        nav2_heading_error = ObsTerm(
            func=custom_observations.nav2_path_heading_error,
        )
        nav2_cross_track_error = ObsTerm(
            func=custom_observations.nav2_cross_track_error,
        )
        combined_scan = ObsTerm(
            func=custom_observations.combined_static_dynamic_scan,
            params={"num_rays": 144, "max_range": 4.0, "step_size": 0.10},
        )

        # dynamic_obstacles = ObsTerm(
        #     func=custom_observations.dynamic_obstacle_states,
        #     params={"num_obstacles": 4, "max_range": 4.0},
        # )

        # path_blocked = ObsTerm(
        #     func=custom_observations.dynamic_path_blockage,
        #     params={"lookahead_points": 32, "path_radius": 0.35},
        # )

        # time_to_closest_approach = ObsTerm(
        #     func=custom_observations.time_to_closest_approach,
        #     params={"num_obstacles": 2, "max_range": 4.0, "horizon_s": 3.0},
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
            params={
                'num_points': 8,
                'step': 8,
            },
        )

        nav2_heading_error = ObsTerm(
            func=custom_observations.nav2_path_heading_error,
        )
        nav2_cross_track_error = ObsTerm(
            func=custom_observations.nav2_cross_track_error,
        )

        combined_scan = ObsTerm(
            func=custom_observations.combined_static_dynamic_scan,
            params={"num_rays": 144, "max_range": 4.0, "step_size": 0.10},
        )
        dynamic_obstacles = ObsTerm(
            func=custom_observations.dynamic_obstacle_states,
            params={"num_obstacles": 4, "max_range": 4.0},
        )
        path_blocked = ObsTerm(
            func=custom_observations.dynamic_path_blockage,
            params={"lookahead_points": 32, "path_radius": 0.35},
        )

        time_to_closest_approach = ObsTerm(
            func=custom_observations.time_to_closest_approach,
            params={"num_obstacles": 2, "max_range": 4.0, "horizon_s": 3.0},
        )

        base_lin_vel = ObsTerm(func=custom_observations.base_lin_vel)
        base_ang_vel = ObsTerm(func=custom_observations.base_ang_vel)
        previous_actions = ObsTerm(func=custom_observations.previous_action)

        distance_to_goal = ObsTerm(func=custom_observations.distance_to_final_goal)
        progress_fraction = ObsTerm(func=custom_observations.nav2_path_progress_fraction)
        map_collision = ObsTerm(func=custom_observations.map_collision_observation)
        dynamic_collision = ObsTerm(func=custom_observations.dynamic_collision_observation)

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
            "max_path_points": 600,
        },
    )

    reset_dynamic_obstacles = EventTerm(
        func=custom_events.reset_dynamic_obstacles_tensor,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_path_points": 600,
        },
    )

    update_dynamic_obstacles = EventTerm(
        func=custom_events.update_dynamic_obstacles_tensor,
        mode="interval",
        interval_range_s=(0.03, 0.03),
    )

    draw_nav2_debug = EventTerm(
        func=custom_events.draw_nav2_map_path_scan_debug,
        mode="interval",
        interval_range_s=(0.03, 0.030),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "map_stride": 2,
            "max_map_points": 6000,
            "path_stride": 4,
            "num_rays": 72,
            "max_range": 4.0,
            "step_size": 0.05,
        },
    )

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
        interval_range_s=(1.0, 1.0),
    )

    log_map_collision_directions = EventTerm(
        func=custom_events.log_map_collision_directions,
        mode="interval",
        interval_range_s=(1.0, 1.0),
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
        weight=35.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'max_step_progress': 0.05,
        },
    )
    goal_approach = RewTerm(
        func=custom_rewards.goal_approach_reward,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_step_progress": 0.08},
    )

    cross_track = RewTerm(
        func=custom_rewards.nav2_cross_track_penalty,
        weight=-1.5,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'max_error': 1.0,
        },
    )

    path_rejoin = RewTerm(
        func=custom_rewards.path_rejoin_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "active_threshold": 0.20},
    )


    heading_alignment = RewTerm(
        func=custom_rewards.nav2_heading_alignment_reward,
        weight=1.5,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'lookahead_index_offset': 4,
        },
    )

    dynamic_collision = RewTerm(
        func=custom_rewards.dynamic_obstacle_collision_penalty,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": 0.22},
    )

    dynamic_clearance = RewTerm(
        func=custom_rewards.dynamic_obstacle_clearance_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": 0.22, "clearance": 0.25},
    )

    dynamic_ttc = RewTerm(
        func=custom_rewards.dynamic_time_to_collision_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": 0.22, "horizon_s": 1.0},
    )

    lateral_oscillation = RewTerm(
        func=custom_rewards.lateral_oscillation_penalty,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    map_collision = RewTerm(
        func=custom_rewards.map_collision_penalty,
        weight=-150.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'radius': 0.22,
        },
    )

    final_goal = RewTerm(
        func=custom_rewards.final_goal_reward,
        weight=120.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'threshold': 0.30,
        },
    )

    action_smoothness = RewTerm(
        func=custom_rewards.action_smoothness_penalty,
        weight=-0.06,
    )

    yaw_rate = RewTerm(
        func=custom_rewards.yaw_rate_penalty,
        weight=-0.02,
    )

    path_velocity = RewTerm(
        func=custom_rewards.path_velocity_reward,
        weight=6.0, #3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    time = RewTerm(
        func=custom_rewards.time_penalty,
        weight=-0.06, #-0.03,
    )

    no_wait = RewTerm(
        func=custom_rewards.no_wait_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "speed_threshold": 0.10},
    )

    # static_velocity_clearance = RewTerm(
    #     func=custom_rewards.static_velocity_clearance_penalty,
    #     weight=-5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "safe_distance": 0.30,
    #         "max_range": 4.0,
    #         "num_rays": 144,
    #         "sector_half_angle_rad": 0.785398,
    #         "min_speed": 0.05,
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
            'threshold': 0.30,
        },
    )

    map_collision = DoneTerm(
        func=custom_terminations.map_collision_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "radius": 0.22,
        },
    )

    dynamic_collision = DoneTerm(
        func=custom_terminations.dynamic_obstacle_collision_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "robot_radius": 0.22},
    )

    stuck = DoneTerm(
        func=custom_terminations.stuck_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_threshold": 0.02,
            "time_window_s": 2.0,
            "grace_period_s": 2.0,
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
    nav2_path_dataset_dir: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/nav_rl_bridge/rl_path_dataset/aws_warehouse"
    nav2_map_yaml_path: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/m3_ros2/maps/no_roof_warehouse.yaml"

    # Tensor dynamic obstacle/curriculum configuration.
    max_dynamic_obstacles: int = 6
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

    # LiDAR curriculum: output observation always stays fixed at 360 rays
    lidar_max_rays: int = 144
    lidar_level_0_rays: int = 72
    lidar_level_1_rays: int = 72
    lidar_level_2_rays: int = 144
    lidar_level_3_rays: int = 144

    scan_noise_level_0: float = 0.0
    scan_noise_level_1: float = 0.005
    scan_noise_level_2: float = 0.01
    scan_noise_level_3: float = 0.02

    scan_dropout_level_0: float = 0.0
    scan_dropout_level_1: float = 0.01
    scan_dropout_level_2: float = 0.02
    scan_dropout_level_3: float = 0.03

    # Battery / motor strength randomization
    battery_scale_level_0_min: float = 1.00
    battery_scale_level_0_max: float = 1.00
    battery_scale_level_1_min: float = 0.95
    battery_scale_level_1_max: float = 1.00
    battery_scale_level_2_min: float = 0.90
    battery_scale_level_2_max: float = 1.00
    battery_scale_level_3_min: float = 0.80
    battery_scale_level_3_max: float = 1.00

    # Action delay randomization
    action_delay_level_0: int = 0
    action_delay_level_1: int = 1
    action_delay_level_2: int = 2
    action_delay_level_3: int = 3
    # Physics domain randomization
    mass_level_0_min: float = 1.0
    mass_level_0_max: float = 1.0
    mass_level_1_min: float = 0.95
    mass_level_1_max: float = 1.05
    mass_level_2_min: float = 0.90
    mass_level_2_max: float = 1.10
    mass_level_3_min: float = 0.85
    mass_level_3_max: float = 1.15

    com_level_0_xy_m: float = 0.0
    com_level_1_xy_m: float = 0.005
    com_level_2_xy_m: float = 0.010
    com_level_3_xy_m: float = 0.020
    com_z_m: float = 0.005

    # curriculum_resets_per_level: int = 200
    curriculum_max_level: int = 23
    fixed_curriculum_level: int = -1
    curriculum_perf_window: int = 1000
    curriculum_min_samples: int = 500

    curriculum_success_promote: float = 0.75
    curriculum_map_collision_promote: float = 0.10
    curriculum_dynamic_collision_promote: float = 0.10
    curriculum_timeout_promote: float = 0.15

    curriculum_success_demote: float = 0.40
    curriculum_map_collision_demote: float = 0.20
    curriculum_dynamic_collision_demote: float = 0.35
    curriculum_timeout_demote: float = 0.40

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
