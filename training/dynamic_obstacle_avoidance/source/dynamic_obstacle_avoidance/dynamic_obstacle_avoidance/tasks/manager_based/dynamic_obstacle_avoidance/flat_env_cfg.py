# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dynamic_obstacle_avoidance.assets.m3 import M3_CFG
from . import mdp


@configclass
class M3FlatSceneCfg(InteractiveSceneCfg):
    """Flat ground scene for ROSMASTER M3 movement testing."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(20.0, 20.0),
        ),
    )

    robot: ArticulationCfg = M3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=600.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


# @configclass
# class ActionsCfg:
#     """Only base velocity command."""

#     base_velocity = mdp.MecanumVelocityActionCfg(
#         class_type=mdp.MecanumVelocityAction,
#         asset_name="robot",
#         wheel_joint_names=[
#             "lwheel1_Joint",
#             "lwheel2_Joint",
#             "rwheel1_Joint",
#             "rwheel2_Joint",
#         ],
#         wheel_radius=0.04,
#         wheel_base_x=0.0795,
#         wheel_base_y=0.09775,
#         max_vx=0.5,
#         max_vy=0.5,
#         max_wz=1.0,
#     )
# @configclass
# class ActionsCfg:
#     raw_wheel_velocity = mdp.RawWheelVelocityActionCfg(
#         class_type=mdp.RawWheelVelocityAction,
#         asset_name="robot",
#         wheel_joint_names=[
#             "lwheel1_Joint",
#             "lwheel2_Joint",
#             "rwheel1_Joint",
#             "rwheel2_Joint",
#         ],
#         max_wheel_speed=5.0,
#     )

@configclass
class ActionsCfg:
    base_velocity = mdp.KinematicMecanumActionCfg(
        class_type=mdp.KinematicMecanumAction,
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
    """Minimal observations for movement test."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        previous_action = ObsTerm(func=mdp.previous_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Only reset robot pose."""

    reset_robot_pose = EventTerm(
        func=mdp.reset_robot_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """No meaningful reward. This env is for action testing only."""

    pass


@configclass
class TerminationsCfg:
    """No early termination for movement test."""

    pass


@configclass
class M3FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Flat-ground ROSMASTER M3 movement test env."""

    scene: M3FlatSceneCfg = M3FlatSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.viewer.eye = (4.0, -4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 2
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
