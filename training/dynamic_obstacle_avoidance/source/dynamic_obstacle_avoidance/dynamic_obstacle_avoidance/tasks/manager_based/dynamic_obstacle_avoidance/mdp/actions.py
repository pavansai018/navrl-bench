from __future__ import annotations
from dataclasses import MISSING
import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

@configclass
class MecanumVelocityActionCfg(ActionTermCfg):
    """
    Action config for ROSMASTER M3 mecanum velocity control.

    Policy action:
        action[0] = vx
        action[1] = vy
        action[2] = wz

    This action term converts base velocity command into four wheel velocity targets
    """

    class_type: type = None
    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING
    wheel_radius: float = MISSING
    wheel_base_x: float = 0.0795
    wheel_base_y: float = 0.09775

    max_vx: float = 0.6
    max_vy: float = 0.6
    max_wz: float = 1.8


class MecanumVelocityAction(ActionTerm):
    """
    Convert [vx, vy, wz] policy action into mecanum wheel velocity targets
    """

    cfg: MecanumVelocityActionCfg

    def __init__(self, cfg: MecanumVelocityActionCfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._wheel_velocity_targets = torch.zeros(env.num_envs, 4, device=env.device)

    @property
    def action_dim(self) -> int:
        return 3
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        actions = torch.clamp(actions, -1.0, 1.0)

        vx = actions[:, 0] * self.cfg.max_vx
        vy = actions[:, 1] * self.cfg.max_vy
        wz = actions[:, 2] * self.cfg.max_wz

        self._processed_actions[:, 0] = vx
        self._processed_actions[:, 1] = 0.0 #vy
        self._processed_actions[:, 2] = 0.0 #wz


        r = self.cfg.wheel_radius
        l = self.cfg.wheel_base_x + self.cfg.wheel_base_y

        # Joint order:
        # lwheel1_Joint = front-left
        # lwheel2_Joint = rear-left
        # rwheel1_Joint = front-right
        # rwheel2_Joint = rear-right

        w_fl = (vx - vy - l * wz) / r
        w_rl = (vx + vy - l * wz) / r
        w_fr = (vx + vy + l * wz) / r
        w_rr = (vx - vy + l * wz) / r

        # self._wheel_velocity_targets[:, 0] = w_fl
        # self._wheel_velocity_targets[:, 1] = w_rl
        # self._wheel_velocity_targets[:, 2] = w_fr
        # self._wheel_velocity_targets[:, 3] = w_rr

        self._wheel_velocity_targets[:, 0] = vx / r
        self._wheel_velocity_targets[:, 1] = vx / r
        self._wheel_velocity_targets[:, 2] = -vx / r
        self._wheel_velocity_targets[:, 3] = -vx / r


    def apply_actions(self):
        self._asset.set_joint_velocity_target(
            self._wheel_velocity_targets,
            joint_ids=self._joint_ids,
        )   


@configclass
class RawWheelVelocityActionCfg(ActionTermCfg):
    """Direct wheel velocity action for debugging wheel signs/order.

    Action:
        [w_lwheel1, w_lwheel2, w_rwheel1, w_rwheel2]
    """

    class_type: type[ActionTerm] = None

    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING
    max_wheel_speed: float = 5.0


class RawWheelVelocityAction(ActionTerm):
    """Directly commands the four wheel joint velocity targets."""

    cfg: RawWheelVelocityActionCfg

    def __init__(self, cfg: RawWheelVelocityActionCfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.wheel_joint_names)

        self._raw_actions = torch.zeros(env.num_envs, 4, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 4, device=env.device)

        print("[RAW WHEEL ACTION] joint names:", self._joint_names)
        print("[RAW WHEEL ACTION] joint ids:", self._joint_ids)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_wheel_speed

    def apply_actions(self):
        self._asset.set_joint_velocity_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )

def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


class KinematicMecanumAction(ActionTerm):
    """Direct planar base action for mecanum navigation training.

    Policy action:
        action[:, 0] = vx in robot/base frame
        action[:, 1] = vy in robot/base frame
        action[:, 2] = wz yaw rate

    This does NOT depend on mecanum wheel contact physics.
    It directly applies planar root velocity and also spins wheel joints.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.wheel_joint_names)

        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._wheel_velocity_targets = torch.zeros(env.num_envs, 4, device=env.device)

        print("[KINEMATIC MECANUM ACTION] wheel joint names:", self._joint_names)
        print("[KINEMATIC MECANUM ACTION] wheel joint ids:", self._joint_ids)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        actions = torch.clamp(actions, -1.0, 1.0)

        vx = actions[:, 0] * self.cfg.max_vx
        vy = actions[:, 1] * self.cfg.max_vy
        wz = actions[:, 2] * self.cfg.max_wz

        self._processed_actions[:, 0] = vx
        self._processed_actions[:, 1] = vy
        self._processed_actions[:, 2] = wz

        r = self.cfg.wheel_radius
        l = self.cfg.wheel_base_x + self.cfg.wheel_base_y

        # Wheel visual/kinematic spin.
        # Sign convention already fixed from your test:
        # forward vx -> [+, +, -, -]
        #
        # This is for wheel spinning consistency. Actual planar motion is applied
        # through root velocity in apply_actions().
        w_fl = (vx - vy - l * wz) / r
        w_rl = (vx + vy - l * wz) / r
        w_fr = -(vx + vy + l * wz) / r
        w_rr = -(vx - vy + l * wz) / r

        self._wheel_velocity_targets[:, 0] = w_fl
        self._wheel_velocity_targets[:, 1] = w_rl
        self._wheel_velocity_targets[:, 2] = w_fr
        self._wheel_velocity_targets[:, 3] = w_rr

    def apply_actions(self):
        # 1. Spin wheel joints
        self._asset.set_joint_velocity_target(
            self._wheel_velocity_targets,
            joint_ids=self._joint_ids,
        )

        # 2. Apply planar base velocity directly
        root_vel = self._asset.data.root_vel_w.clone()

        vx_b = self._processed_actions[:, 0]
        vy_b = self._processed_actions[:, 1]
        wz = self._processed_actions[:, 2]

        yaw = _yaw_from_quat_wxyz(self._asset.data.root_quat_w)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Convert base-frame velocity to world-frame velocity.
        vx_w = cos_yaw * vx_b - sin_yaw * vy_b
        vy_w = sin_yaw * vx_b + cos_yaw * vy_b

        root_vel[:, 0] = vx_w
        root_vel[:, 1] = vy_w
        root_vel[:, 2] = 0.0

        root_vel[:, 3] = 0.0
        root_vel[:, 4] = 0.0
        root_vel[:, 5] = wz

        self._asset.write_root_velocity_to_sim(root_vel)


@configclass
class KinematicMecanumActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = KinematicMecanumAction

    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING

    wheel_radius: float = 0.04
    wheel_base_x: float = 0.0795
    wheel_base_y: float = 0.09775

    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 1.0
