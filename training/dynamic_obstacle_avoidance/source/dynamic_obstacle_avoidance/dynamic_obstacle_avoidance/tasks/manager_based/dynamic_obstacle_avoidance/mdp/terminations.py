from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy, map_collision_flag


def final_goal_reached(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30,) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    distance = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    return distance < threshold

def map_collision_termination(env, asset_cfg: SceneEntityCfg, radius: float = 0.22,) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,)