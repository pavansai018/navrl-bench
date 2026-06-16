# local_planner/base_env_cfg.py

from __future__ import annotations

import isaaclab.sim as sim_utils

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class LocalPlannerSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = MISSING

    final_goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FinalGoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.16,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.9, 0.25),
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )


@configclass
class ActionsCfg:
    local_path = mdp.LocalPlannerOffsetActionCfg(
        asset_name="robot",
        num_path_points=8,
        max_offset=0.8,
    )

    # tracker = mdp.LocalPathTrackerActionCfg(
    #     asset_name="robot",
    #     wheel_joint_names=[
    #         "lwheel1_Joint",
    #         "lwheel2_Joint",
    #         "rwheel1_Joint",
    #         "rwheel2_Joint",
    #     ],
    #     wheel_radius=0.035,
    #     wheel_base_x=0.0795,
    #     wheel_base_y=0.09775,
    #     max_vx=1.5,
    #     max_vy=1.5,
    #     max_wz=2.0,
    #     max_delta_vx=0.08,
    #     max_delta_vy=0.08,
    #     max_delta_wz=0.15,
    #     num_path_points=8,
    #     step=4,
    #     target_point_index=1,
    #     kx=1.4,
    #     ky=1.8,
    #     kyaw=2.2,
    # )
    tracker = mdp.FrozenPolicyTrackerActionCfg(
        asset_name="robot",
        policy_path="/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/training/dynamic_obstacle_avoidance/logs/rsl_rl/m3_obstacle_avoidance/2026-06-10_09-48-59_trailv8/exported/policy.pt",

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
        max_wz=1.0,
        max_delta_vx=0.06,
        max_delta_vy=0.06,
        max_delta_wz=0.15,

        num_path_points=8,
        step=4,
        path_norm_m=4.0,
        target_point_index=5,
        obs_dim=168,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        local_path_window = ObsTerm(
            func=mdp.local_path_window,
            params={"num_points": 8, "step": 8},
        )

        combined_scan = ObsTerm(
            func=mdp.combined_static_dynamic_scan,
            params={"num_rays": 144, "max_range": 4.0, "step_size": 0.10},
        )

        dynamic_obstacles = ObsTerm(
            func=mdp.dynamic_obstacle_states,
            params={"num_obstacles": 4, "max_range": 4.0},
        )

        path_blocked = ObsTerm(
            func=mdp.dynamic_path_blockage,
            params={"lookahead_points": 32, "path_radius": 0.35},
        )

        previous_local_offsets = ObsTerm(
            func=mdp.previous_local_offsets,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        local_path_window = ObsTerm(
            func=mdp.local_path_window,
            params={"num_points": 8, "step": 8},
        )

        combined_scan = ObsTerm(
            func=mdp.combined_static_dynamic_scan,
            params={"num_rays": 144, "max_range": 4.0, "step_size": 0.10},
        )

        dynamic_obstacles = ObsTerm(
            func=mdp.dynamic_obstacle_states,
            params={"num_obstacles": 4, "max_range": 4.0},
        )

        path_blocked = ObsTerm(
            func=mdp.dynamic_path_blockage,
            params={"lookahead_points": 32, "path_radius": 0.35},
        )

        previous_local_offsets = ObsTerm(
            func=mdp.previous_local_offsets,
        )

        local_path_collision_risk = ObsTerm(
            func=mdp.local_path_collision_risk_observation,
            params={"num_points": 8, "safe_distance": 0.35},
        )

        map_collision = ObsTerm(
            func=mdp.map_collision_observation,
        )

        dynamic_collision = ObsTerm(
            func=mdp.dynamic_collision_observation,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # local_path_dynamic_clearance = RewTerm(
    #     func=mdp.local_path_dynamic_clearance_penalty,
    #     weight=-20.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "num_points": 8,
    #         "safe_distance": 0.35,
    #     },
    # )

    # local_path_static_clearance = RewTerm(
    #     func=mdp.local_path_static_clearance_penalty,
    #     weight=-35.0,
    #     params={
    #         "num_points": 8,
    #         "safe_distance": 0.30,
    #     },
    # )

    local_path_smoothness = RewTerm(
        func=mdp.local_path_smoothness_penalty,
        weight=-3.0,
    )

    local_path_rejoin = RewTerm(
        func=mdp.local_path_rejoin_penalty,
        weight=-3.0,
        params={"num_points": 8},
    )

    dynamic_collision = RewTerm(
        func=mdp.dynamic_obstacle_collision_penalty,
        weight=-250.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "robot_radius": 0.22,
        },
    )

    map_collision = RewTerm(
        func=mdp.map_collision_penalty,
        weight=-300.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "radius": 0.22,
        },
    )

    progress = RewTerm(
        func=mdp.progress_along_global_path,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_step_progress": 0.05,
        },
    )

    final_goal = RewTerm(
        func=mdp.final_goal_reward,
        weight=120.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.30,
        },
    )

    stuck_penalty = RewTerm(
        func=mdp.stuck_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_threshold": 0.05,
        },
    )

    # conditional_offset = RewTerm(
    #     func=mdp.conditional_offset_penalty,
    #     weight=-8.0,
    #     params={
    #         "lookahead_points": 32,
    #         "path_radius": 0.35,
    #         "free_scale": 1.0,
    #         "blocked_scale": 0.10,
    #     },
    # )
    conditional_structured_offset = RewTerm(
        func=mdp.conditional_structured_offset_penalty,
        weight=-8.0,
        params={
            "lookahead_points": 32,
            "path_radius": 0.35,
        },
    )

    # local_path_static_body_clearance = RewTerm(
    #     func=mdp.local_path_static_body_clearance_penalty,
    #     weight=-40.0,
    #     params={
    #         "num_points": 8,
    #         "robot_radius": 0.25,
    #     },
    # )

    # local_path_segment_dynamic_clearance = RewTerm(
    #     func=mdp.local_path_segment_dynamic_clearance_penalty,
    #     weight=-30.0,
    #     params={
    #         "num_points": 8,
    #         "samples_per_segment": 5,
    #         "safe_distance": 0.35,
    #     },
    # )

    # local_path_segment_static_clearance = RewTerm(
    #     func=mdp.local_path_segment_static_clearance_penalty,
    #     weight=-35.0,
    #     params={
    #         "num_points": 8,
    #         "samples_per_segment": 5,
    #     },
    # )

    dense_path_dynamic_clearance = RewTerm(
        func=mdp.dense_path_dynamic_clearance_penalty,
        weight=-50.0,
        params={
            "clearance": 0.30,
            "samples_per_segment": 8,
        },
    )

    dense_path_static_clearance = RewTerm(
        func=mdp.dense_path_static_clearance_penalty,
        weight=-50.0,
        params={
            "clearance": 0.30,
            "samples_per_segment": 8,
            "circle_samples": 16,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    dynamic_collision = DoneTerm(
        func=mdp.dynamic_collision,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "robot_radius": 0.22,
        },
    )

    map_collision = DoneTerm(
        func=mdp.map_collision,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "radius": 0.22,
        },
    )

    final_goal_reached = DoneTerm(
        func=mdp.final_goal_reached,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.30,
        },
    )
    # stuck = DoneTerm(
    #     func=mdp.stuck,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "speed_threshold": 0.05,
    #         "time_window_s": 1.0,
    #         "grace_period_s": 2.0,
    #     },
    # )


@configclass
class EventCfg:
    reset_nav2_path = EventTerm(
        func=mdp.reset_nav2_path_and_debug_validate,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "final_goal_marker_cfg": SceneEntityCfg("final_goal_marker"),
            "max_path_points": 600,
        },
    )

    reset_dynamic_obstacles = EventTerm(
        func=mdp.reset_dynamic_obstacles_tensor,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_path_points": 600,
        },
    )

    update_dynamic_obstacles = EventTerm(
        func=mdp.update_dynamic_obstacles_tensor,
        mode="interval",
        interval_range_s=(0.03, 0.03),
    )

    draw_nav2_debug = EventTerm(
        func=mdp.draw_nav2_map_path_scan_debug,
        mode="interval",
        interval_range_s=(0.03, 0.03),
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

    log_curriculum_progress = EventTerm(
        func=mdp.log_curriculum_progress,
        mode="interval",
        interval_range_s=(1.0, 1.0),
    )


@configclass
class LocalPlannerBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(
        num_envs=512,
        env_spacing=22.0,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    nav2_path_dataset_dir: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/nav_rl_bridge/rl_path_dataset/aws_warehouse"
    nav2_map_yaml_path: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/m3_ros2/maps/no_roof_warehouse.yaml"

    max_dynamic_obstacles: int = 6
    dynamic_obstacle_deactivate_range: float = 6.0

    reset_lateral_noise_m: float = 0.0
    reset_yaw_noise_rad: float = 0.05
    path_window_normalization_m: float = 4.0

    debug_validate_nav2_path: bool = False
    debug_draw_nav2: bool = False
    debug_draw_lidar: bool = False
    debug_draw_map: bool = False
    debug_draw_path: bool = False
    debug_draw_dynamic_obstacles: bool = False
    debug_draw_rl_local_path: bool = False
    debug_draw_max_envs: int = 4

    fixed_curriculum_level: int = 3
    curriculum_max_level: int = -1 #len(custom_events.read_config()['domain_randomization_stages'] + custom_events.read_config()['obstacle_stages'])
    fixed_curriculum_level: int = -1
    curriculum_perf_window: int = 5000
    curriculum_min_samples: int = 4500

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

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0

        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 2
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15