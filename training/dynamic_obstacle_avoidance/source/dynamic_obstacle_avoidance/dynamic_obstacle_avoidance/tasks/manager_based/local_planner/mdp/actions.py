from __future__ import annotations

from dataclasses import MISSING
import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp.actions import KinematicMecanumAction, KinematicMecanumActionCfg
from .observations import build_rl_local_path, get_rl_local_path



class LocalPlannerOffsetAction(ActionTerm):

    def __init__(self, cfg: LocalPlannerOffsetActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, cfg.num_path_points, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._env.rl_local_path_offsets = torch.zeros(env.num_envs, cfg.num_path_points, device=env.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.num_path_points

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_offset
        self._env.rl_local_path_offsets[:] = self._processed_actions

    def apply_actions(self):
        pass

@configclass
class LocalPlannerOffsetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LocalPlannerOffsetAction
    asset_name: str = MISSING
    num_path_points: int = 8
    max_offset: float = 0.8


class LocalPathTrackerAction(KinematicMecanumAction):
    cfg: LocalPathTrackerActionCfg

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions):
        # no policy action for tracker
        pass

    def apply_actions(self):
    
        _, path_robot = get_rl_local_path(self._env, num_points=self.cfg.num_path_points, step=self.cfg.step)

        idx = min(self.cfg.target_point_index, self.cfg.num_path_points - 1)
        # target = self._env.rl_local_path_robot[:, idx, :]
        target = path_robot[:, idx, :]

        x = target[:, 0]
        y = target[:, 1]

        heading_error = torch.atan2(y, torch.clamp(x, min=1e-4))

        vx = torch.clamp(self.cfg.kx * x, 0.35, self.cfg.max_vx)
        vy = torch.clamp(self.cfg.ky * y, -self.cfg.max_vy, self.cfg.max_vy)
        wz = torch.clamp(self.cfg.kyaw * heading_error, -self.cfg.max_wz, self.cfg.max_wz)

        tracker_action = torch.stack([vx, vy, wz], dim=-1)

        super().process_actions(tracker_action)
        super().apply_actions()

@configclass
class LocalPathTrackerActionCfg(KinematicMecanumActionCfg):
    class_type: type[ActionTerm] = LocalPathTrackerAction

    num_path_points: int = 8
    step: int = 4
    target_point_index: int = 1

    kx: float = 1.4
    ky: float = 1.8
    kyaw: float = 2.2

    wheel_joint_names=[
        "lwheel1_Joint",
        "lwheel2_Joint",
        "rwheel1_Joint",
        "rwheel2_Joint",
    ],
    wheel_radius=0.035,
    wheel_base_x=0.0795,
    wheel_base_y=0.09775,
    max_vx=0.5,
    max_vy=0.5,
    max_wz=1.2,
    max_delta_vx=0.025,
    max_delta_vy=0.025,
    max_delta_wz=0.08,



class FrozenPolicyTrackerAction(KinematicMecanumAction):
    cfg: FrozenPolicyTrackerActionCfg

    def __init__(self, cfg: FrozenPolicyTrackerActionCfg, env):
        super().__init__(cfg, env)

        self._tracker_policy = torch.jit.load(cfg.policy_path, map_location=env.device)
        self._tracker_policy.eval()

        self._prev_tracker_action = torch.zeros(env.num_envs, 3, device=env.device)
        self._stuck_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._recovery_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    @property
    def action_dim(self) -> int:
        # no policy action from PPO
        return 0

    def process_actions(self, actions: torch.Tensor):
        pass

    def _build_tracker_obs(self) -> torch.Tensor:
        from .observations import get_rl_local_path, combined_static_dynamic_scan

        env = self._env
        robot = env.scene[self.cfg.asset_name]

        _, path_robot = get_rl_local_path(
            env,
            num_points=self.cfg.num_path_points,
            step=self.cfg.step,
        )

        # 1. local path window, same role as old controller local_path_window
        local_path_window = torch.clamp(
            path_robot.reshape(env.num_envs, self.cfg.num_path_points * 2) / self.cfg.path_norm_m,
            -2.0,
            2.0,
        )

        # 2. heading error to generated local path
        target = path_robot[:, min(self.cfg.target_point_index, self.cfg.num_path_points - 1), :]
        heading_error = torch.atan2(
            target[:, 1],
            torch.clamp(target[:, 0], min=1e-4),
        ).unsqueeze(-1)

        # 3. cross-track error to local path
        # y of first local path point in robot frame
        cross_track_error = torch.clamp(
            path_robot[:, 0, 1:2] / 1.0,
            -1.0,
            1.0,
        )

        # 4. same scan observation as old velocity policy
        scan = combined_static_dynamic_scan(
            env,
            num_rays=144,
            max_range=4.0,
            step_size=0.10,
        )

        # 5. robot velocity in body frame
        base_lin_vel = torch.clamp(robot.data.root_lin_vel_b[:, :2] / 1.0, -2.0, 2.0)
        base_ang_vel = torch.clamp(robot.data.root_ang_vel_b[:, 2:3] / 2.0, -2.0, 2.0)

        # 6. previous velocity action, normalized
        prev_action = torch.clamp(self._prev_tracker_action, -1.0, 1.0)

        obs = torch.cat(
            [
                local_path_window,     # 16
                heading_error,         # 1
                cross_track_error,     # 1
                scan,                  # 144
                base_lin_vel,          # 2
                base_ang_vel,          # 1
                prev_action,           # 3
            ],
            dim=-1,
        )

        if obs.shape[1] != self.cfg.obs_dim:
            raise RuntimeError(
                f"Frozen tracker obs_dim mismatch. Got {obs.shape[1]}, expected {self.cfg.obs_dim}. "
                "velocity policy observation order/size is different."
            )

        return obs

    def apply_actions(self):
        with torch.no_grad():
            tracker_obs = self._build_tracker_obs()
            vel_action = self._tracker_policy(tracker_obs)

        vel_action = torch.clamp(vel_action, -1.0, 1.0)

        robot = self._env.scene[self.cfg.asset_name]
        speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)

        commanded = torch.norm(vel_action[:, :2], dim=-1) > 0.5
        stuck_now = commanded & (speed < 0.05) & (self._env.episode_length_buf > 60)

        self._stuck_counter = torch.where(
            stuck_now,
            self._stuck_counter + 1,
            torch.zeros_like(self._stuck_counter),
        )

        enter_recovery = self._stuck_counter > 15
        self._recovery_counter = torch.where(
            enter_recovery,
            torch.full_like(self._recovery_counter, 25),
            self._recovery_counter,
        )

        in_recovery = self._recovery_counter > 0

        if in_recovery.any():
            recovery_action = torch.zeros_like(vel_action)

            # reverse + rotate. This is normalized [vx, vy, wz]
            recovery_action[:, 0] = -0.4
            recovery_action[:, 1] = 0.0
            recovery_action[:, 2] = 0.8

            vel_action = torch.where(
                in_recovery.unsqueeze(-1),
                recovery_action,
                vel_action,
            )

            self._recovery_counter = torch.clamp(self._recovery_counter - 1, min=0)

        self._prev_tracker_action[:] = vel_action

        super().process_actions(vel_action)
        super().apply_actions()


@configclass
class FrozenPolicyTrackerActionCfg(KinematicMecanumActionCfg):
    class_type: type[ActionTerm] = FrozenPolicyTrackerAction

    policy_path: str = MISSING

    num_path_points: int = 8
    step: int = 4
    path_norm_m: float = 4.0
    target_point_index: int = 3

    # Must match old velocity-policy actor observation size.
    # local_path_window 16
    # heading_error 1
    # cross_track_error 1
    # combined_scan 144
    # base_lin_vel 2
    # base_ang_vel 1
    # previous_action 3
    # total = 168
    obs_dim: int = 168