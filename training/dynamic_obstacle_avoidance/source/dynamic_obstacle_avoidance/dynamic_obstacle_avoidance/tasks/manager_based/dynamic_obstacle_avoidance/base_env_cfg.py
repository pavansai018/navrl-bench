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

    # Reusable wall blocks for on-the-fly procedural map generation
    walls = RigidObjectCollectionCfg(
        rigid_objects={
            f'wall_{i:03d}': RigidObjectCfg(
                prim_path=f'{{ENV_REGEX_NS}}/Wall_{i:03d}',
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,

                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.72, 0.76, 0.82),
                        roughness=0.8,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10)),
            )
            for i in range(96)
        }
    )

    # static obstacles: boxes, furniture-like blocks, narrow-passge blockers
    static_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f'static_{i:02d}': RigidObjectCfg(
                prim_path=f'{{ENV_REGEX_NS}}/StaticObstacle_{i:02d}',
                spawn=sim_utils.CuboidCfg(
                    size=(0.55, 0.55, 0.45),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.42, 0.52),
                        roughness=0.8,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for i in range(12)
        }
    )

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


    # Lookahead marker: visual only, represents local Nav2-style target
    lookahead_marker = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/LookaheadMarker',
        spawn=sim_utils.SphereCfg(
            radius=0.10,

            # Kinematic visual marker.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),

            # IMPORTANT:
            # Do not add collision_props here.
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.45, 1.0),
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.10),
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

    base_velocity = custom_actions.MecanumVelocityActionCfg(
        class_type=custom_actions.MecanumVelocityAction,
        asset_name = 'robot',
        wheel_joint_names = [
            'lwheel1_Joint', # front left
            'lwheel2_Joint', # rear left
            'rwheel1_Joint', # front right
            'rwheel2_Joint', # rear right
        ],
        wheel_radius=0.04, # find correct radius
        wheel_base_x=0.0795, # from urdf wheel x offset,
        wheel_base_y=0.09775, # from urdf wheel y offset,
        max_vx=0.6,
        max_vy=0.6,
        max_wz=1.8,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for dynamic obstacle avoidance."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        Observation should describe local navigation, not full global navigation.
        """

        # Observation terms order preserved.

        # Local path / lookahead target information
        lookahead_distance = ObsTerm(func=custom_observations.lookahead_distance)
        lookahead_angle = ObsTerm(func=custom_observations.lookahead_angle)
        path_heading_error = ObsTerm(func=custom_observations.path_heading_error)
        cross_track_error = ObsTerm(func=custom_observations.cross_track_error)

        # Robot motion
        base_lin_vel = ObsTerm(func=custom_observations.base_lin_vel)
        base_angle_vel = ObsTerm(func=custom_observations.base_ang_vel)

        # Local obstacle information
        obstacle_scan = ObsTerm(
            func=custom_observations.obstacle_scan,
            params={
                'static_asset_cfg': SceneEntityCfg('static_obstacles'),
                'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
                'num_rays': 72,
                'max_range': 4.0,
            },
        )

        # Nearest dynamic obstacle features

        nearest_dynamic_obstacle = ObsTerm(
            func=custom_observations.nearest_dynamic_obstacle,
            params={
                'asset_cfg': SceneEntityCfg('dynamic_obstacles'),
                'max_range': 4.0,
            },
        )

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
    # generate map layout at reset
    reset_runtime_map = EventTerm(
        func=custom_events.reset_runtime_map,
        mode='reset',
        params={
            'wall_asset_cfg': SceneEntityCfg('walls'),
            'static_asset_cfg': SceneEntityCfg('static_obstacles'),
            'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
            # 'goal_asset_cfg': SceneEntityCfg('goal_marker'),
            'template_name': 'random',
            'num_static_obstacles': 6,
            'num_dynamic_obstacles': 4,
        },
    )

    # reset robot start pose
    reset_robot_pose = EventTerm(
        func=custom_events.reset_robot_pose,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'x_range': (-4.5, -3.5),
            'y_range': (-0.6, 0.6),
            'yaw_range': (-0.15, 0.15),
        },
    )

    
    reset_path_command = EventTerm(
        func=custom_events.reset_path_command,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot'),

            # visual markers only
            'final_goal_marker_cfg': SceneEntityCfg('final_goal_marker'),
            'lookahead_marker_cfg': SceneEntityCfg('lookahead_marker'),
            'start_x_range': (-4.5, -3.5),
            'start_y_range': (-0.6, 0.6),
            'goal_x_range': (3.5, 4.5),
            'goal_y_range': (-0.8, 0.8),
            'lookahead_distance': 0.8,
        },
    )

    # Reset dynamic obstacle speed and direction
    reset_dynamic_obstacles = EventTerm(
        func=custom_events.reset_dynamic_obstacles,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('dynamic_obstacles'),
            'min_speed': 0.35,
            'max_speed': 1.20,
            'num_active': 4,
        },
    )

    # Move dynamic obstacles during episode
    move_dynamic_obstacles = EventTerm(
        func=custom_events.move_dynamic_obstacles,
        mode='interval',
        interval_range_s=(0.02, 0.02),
        params={
            'asset_cfg': SceneEntityCfg('dynamic_obstacles'),
            'x_limit': 5.5,
            'y_limit': 3.0,
        },
    )

    update_lookahead_target = EventTerm(
        func=custom_events.update_lookahead_target,
        mode='interval',
        interval_range_s=(0.05, 0.05),
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'lookahead_marker_cfg': SceneEntityCfg('lookahead_marker'),
            'lookahead_distance': 0.8,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for RL local-controller and dynamic obstacle avoidance."""

    # Follow local path / lookahead target
    progress_along_path = RewTerm(
        func=custom_rewards.progress_along_path,
        weight=6.0,
        params={'asset_cfg': SceneEntityCfg('robot')},
    )

    lookahead_tracking = RewTerm(
        func=custom_rewards.lookahead_tracking_reward,
        weight=3.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'distance_scale': 1.0,
            'angle_scale': 1.0,
        },
    )

    path_alignment = RewTerm(
        func=custom_rewards.path_alignment_reward,
        weight=2.0,
        params={'asset_cfg': SceneEntityCfg('robot')},
    )

    cross_track_error = RewTerm(
        func=custom_rewards.cross_track_error_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Obstacle avoidance
    obstacle_clearance = RewTerm(
        func=custom_rewards.obstacle_clearance_reward,
        weight=2.0,
        params={
            'static_asset_cfg': SceneEntityCfg('static_obstacles'),
            'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
            'safe_distance': 0.35,
        },
    )

    collision = RewTerm(
        func=custom_rewards.collision_penalty,
        weight=-80.0,
        params={
            'robot_cfg': SceneEntityCfg('robot'),
            'static_asset_cfg': SceneEntityCfg('static_obstacles'),
            'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
        },
    )

    # Dynamic obstacle behavior
    dynamic_obstacle_clearance = RewTerm(
        func=custom_rewards.dynamic_obstacle_clearance_reward,
        weight=3.0,
        params={
            'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
            'safe_distance': 0.45,
        },
    )

    # Avoid freezing
    unnecessary_stop = RewTerm(
        func=custom_rewards.unnecessary_stop_penalty,
        weight=-1.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'speed_threshold': 0.04,
        },
    )

    # Smooth control
    action_smoothness = RewTerm(
        func=custom_rewards.action_smoothness_penalty,
        weight=-0.05,
    )

    # Finish route / local path task
    final_goal_reached = RewTerm(
        func=custom_rewards.final_goal_reached_reward,
        weight=50.0,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'threshold': 0.30,
        },
    )

    # Time pressure
    time_penalty = RewTerm(
        func=custom_rewards.constant_penalty,
        weight=-0.01,
    )

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
    # Collision with static or dynamic obstacles
    collision = DoneTerm(
        func=custom_terminations.collision_termination,
        params={
            'robot_cfg': SceneEntityCfg('robot'),
            'static_asset_cfg': SceneEntityCfg('static_obstacles'),
            'dynamic_asset_cfg': SceneEntityCfg('dynamic_obstacles'),
        },
    )

    # Robot leaves training arena
    out_of_bounds = DoneTerm(
        func=custom_terminations.robot_out_of_bounds,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'x_bounds': (-6.0, 6.0),
            'y_bounds': (-3.5, 3.5),
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
