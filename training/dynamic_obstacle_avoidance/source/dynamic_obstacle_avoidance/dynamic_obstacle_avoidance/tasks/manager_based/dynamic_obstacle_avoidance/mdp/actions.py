from __future__ import annotations
from dataclasses import MISSING
import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import math

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

    max_vx: float = 0.75
    max_vy: float = 0.75
    max_wz: float = 2.0


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
        # Test-only: intentionally disabling vy and wz
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

        self._env = env
        self._action_delay_buffer = torch.zeros(env.num_envs, 4, 3, device=env.device)
        self._applied_actions = torch.zeros(env.num_envs, 3, device=env.device)

        self.max_delta_vx = float(getattr(cfg, "max_delta_vx", 0.04))
        self.max_delta_vy = float(getattr(cfg, "max_delta_vy", 0.04))
        self.max_delta_wz = float(getattr(cfg, "max_delta_wz", 0.12))
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

        if hasattr(self._env, "action_delay_steps"):
            self._action_delay_buffer = torch.roll(self._action_delay_buffer, shifts=1, dims=1)
            self._action_delay_buffer[:, 0, :] = actions

            delay = self._env.action_delay_steps.clamp(0, self._action_delay_buffer.shape[1] - 1)
            env_ids = torch.arange(actions.shape[0], device=actions.device)
            actions = self._action_delay_buffer[env_ids, delay]

        actions = torch.clamp(actions, -1.0, 1.0)

        vx = actions[:, 0] * self.cfg.max_vx
        vy = actions[:, 1] * self.cfg.max_vy
        wz = actions[:, 2] * self.cfg.max_wz

        if hasattr(self._env, "motor_strength_scale"):
            scale = self._env.motor_strength_scale
            vx = vx * scale
            vy = vy * scale
            wz = wz * scale
            
        # self._processed_actions[:, 0] = vx
        # self._processed_actions[:, 1] = vy
        # self._processed_actions[:, 2] = wz

        target = torch.stack([vx, vy, wz], dim=-1)

        max_delta = torch.tensor(
            [self.max_delta_vx, self.max_delta_vy, self.max_delta_wz],
            device=target.device,
            dtype=target.dtype,
        )

        delta = torch.clamp(
            target - self._applied_actions,
            min=-max_delta,
            max=max_delta,
        )

        self._applied_actions[:] = self._applied_actions + delta
        # self._processed_actions[:] = self._applied_actions
        if bool(getattr(self.cfg, "enable_static_action_shield", True)):
            self._processed_actions[:] = self._static_map_action_shield(self._applied_actions)
        else:
            self._processed_actions[:] = self._applied_actions

        vx = self._processed_actions[:, 0]
        vy = self._processed_actions[:, 1]
        wz = self._processed_actions[:, 2]

        if hasattr(self._env, "wheel_slip_scale"):
            slip = self._env.wheel_slip_scale  # [num_envs, 4]

            # Convert 4 wheel slip values into effective chassis degradation.
            # Mecanum lateral motion is more sensitive to wheel slip.
            vx_slip = slip.mean(dim=1)
            vy_slip = slip.mean(dim=1)

            # Yaw depends on left/right imbalance.
            left_slip = 0.5 * (slip[:, 0] + slip[:, 1])
            right_slip = 0.5 * (slip[:, 2] + slip[:, 3])
            wz_slip = 0.5 * (left_slip + right_slip)

            vx = vx * vx_slip
            vy = vy * vy_slip
            wz = wz * wz_slip

            # Wheel imbalance creates unwanted yaw drift.
            yaw_coupling = (right_slip - left_slip) * 0.3
            wz = wz + yaw_coupling * torch.abs(vx)

        if hasattr(self._env, "wheel_radius_scale"):
            radius_scale = self._env.wheel_radius_scale  # [num_envs, 4]

            # Effective wheel-radius/gain mismatch.
            radius_mean = radius_scale.mean(dim=1)

            left_radius = 0.5 * (radius_scale[:, 0] + radius_scale[:, 1])
            right_radius = 0.5 * (radius_scale[:, 2] + radius_scale[:, 3])

            vx = vx * radius_mean
            vy = vy * radius_mean

            # Left/right radius mismatch causes heading drift.
            radius_yaw_bias = (right_radius - left_radius) * 0.4
            wz = wz + radius_yaw_bias * torch.abs(vx)

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

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._applied_actions.zero_()
        else:
            self._applied_actions[env_ids] = 0.0

    def _static_map_action_shield(self, cmd: torch.Tensor) -> torch.Tensor:
        if not hasattr(self._env, "nav2_occupancy_map"):
            return cmd

        robot = self._asset
        dt = float(getattr(self._env, "step_dt", 1.0 / 30.0))
        radius = float(getattr(self.cfg, "shield_robot_radius", 0.22))
        num_points = int(getattr(self.cfg, "shield_num_points", 16))

        root_xy = robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
        q = robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
        )

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx_b = cmd[:, 0]
        vy_b = cmd[:, 1]

        vx_w = cos_yaw * vx_b - sin_yaw * vy_b
        vy_w = sin_yaw * vx_b + cos_yaw * vy_b

        scales = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], device=cmd.device)
        safe_cmd = cmd.clone()

        angles = torch.linspace(
            0.0, 2.0 * math.pi, num_points + 1, device=cmd.device
        )[:-1]

        offsets = torch.stack(
            [torch.cos(angles) * radius, torch.sin(angles) * radius],
            dim=-1,
        )

        chosen = torch.zeros(cmd.shape[0], dtype=torch.bool, device=cmd.device)

        for s in scales:
            test_xy = root_xy + torch.stack([vx_w, vy_w], dim=-1) * dt * s
            footprint = test_xy[:, None, :] + offsets[None, :, :]

            occupied = self._env.nav2_occupancy_map.is_occupied_world(
                footprint.reshape(-1, 2),
                unknown_is_occupied=True,
            ).reshape(cmd.shape[0], num_points)

            center_occ = self._env.nav2_occupancy_map.is_occupied_world(
                test_xy,
                unknown_is_occupied=True,
            )

            collision = occupied.any(dim=-1) | center_occ
            can_use = (~collision) & (~chosen)

            safe_cmd[can_use] = cmd[can_use] * s
            chosen = chosen | can_use

        safe_cmd[~chosen] = 0.0
        return safe_cmd


@configclass
class KinematicMecanumActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = KinematicMecanumAction

    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING

    wheel_radius: float = 0.035
    wheel_base_x: float = 0.0795
    wheel_base_y: float = 0.09775

    max_vx: float = 0.75
    max_vy: float = 0.75
    max_wz: float = 2.0

    max_delta_vx: float = 0.04
    max_delta_vy: float = 0.04
    max_delta_wz: float = 0.12
    enable_static_action_shield: bool = True
    shield_robot_radius: float = 0.22
    shield_num_points: int = 16
