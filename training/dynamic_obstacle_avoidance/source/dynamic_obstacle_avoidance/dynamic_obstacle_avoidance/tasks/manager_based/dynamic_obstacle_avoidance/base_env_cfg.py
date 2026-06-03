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

    # Dynamic obstacles: moving cylinders / human proxies

    dynamic_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f'dynamic_{i:02d}': RigidObjectCfg(
                prim_path=f'{{ENV_REGEX_NS}}/DynamicObstacle_{i:02d}',
                spawn=sim_utils.CylinderCfg(
                    radius=0.25,
                    height=0.65,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.58, 0.10),
                        roughness=0.7,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for i in range(8)
        }
    )

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
        wheel_radius=0.04,
        wheel_base_x=0.0795,
        wheel_base_y=0.09775,
        max_vx=0.5,
        max_vy=0.5,
        max_wz=1.0,
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
        map_scan = ObsTerm(
            func=custom_observations.map_based_scan,
            params={
                'num_rays': 72,
                'max_range': 4.0,
                'step_size': 0.05,
            },
        )

        # Robot motion
        base_lin_vel = ObsTerm(func=custom_observations.base_lin_vel)
        base_angle_vel = ObsTerm(func=custom_observations.base_ang_vel)
        # Previous action for smoother policy behavior
        previous_action = ObsTerm(func=custom_observations.previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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

    draw_nav2_debug = EventTerm(
        func=custom_events.draw_nav2_map_path_scan_debug,
        mode="interval",
        interval_range_s=(0.20, 0.20),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "map_stride": 1,
            "max_map_points": 30000,
            "path_stride": 4,
            "num_rays": 72,
            "max_range": 4.0,
            "step_size": 0.05,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for RL local-controller and dynamic obstacle avoidance."""
    pass
    

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

    debug_draw_nav2: bool = True
    debug_draw_lidar: bool = True
    debug_draw_map: bool = True
    debug_draw_path: bool = True
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
