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
        self._processed_actions[:, 1] = vy
        self._processed_actions[:, 2] = wz


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

        self._wheel_velocity_targets[:, 0] = w_fl
        self._wheel_velocity_targets[:, 1] = w_rl
        self._wheel_velocity_targets[:, 2] = w_fr
        self._wheel_velocity_targets[:, 3] = w_rr

    def apply_actions(self):
        self._asset.set_joint_velocity_target(
            self._wheel_velocity_targets,
            joint_ids=self._joint_ids,
        )   

    