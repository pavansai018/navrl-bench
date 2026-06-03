from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy, _robot_yaw, _wrap_to_pi, map_collision_flag



def constant_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def _nearest_path_index(env, robot_xy: torch.Tensor) -> torch.Tensor:
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    dist = torch.norm(path - robot_xy[:, None, :], dim=-1)

    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]

    dist = torch.where(valid_mask, dist, torch.ones_like(dist) * 1e6)

    return torch.argmin(dist, dim=-1)

def progress_along_nav2_path(env, asset_cfg: SceneEntityCfg, max_step_progress: float = 0.05,) -> torch.Tensor:
    """Reward forward progress along the Nav2 global path."""
    robot_xy = _robot_xy(env, asset_cfg.name)
    nearest_idx = _nearest_path_index(env, robot_xy)

    env_ids = torch.arange(env.num_envs, device=env.device)
    current_s = env.navrl_path_cum_s[env_ids, nearest_idx]

    delta_s = current_s - env.navrl_prev_progress_s
    env.navrl_prev_progress_s[:] = current_s

    return torch.clamp(delta_s, -max_step_progress, max_step_progress)

def nav2_cross_track_penalty(env, asset_cfg: SceneEntityCfg, max_error: float = 1.0,) -> torch.Tensor:
    """Penalty for distance away from nearest Nav2 path point."""
    robot_xy = _robot_xy(env, asset_cfg.name)
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    dist = torch.norm(path - robot_xy[:, None, :], dim=-1)

    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]

    dist = torch.where(valid_mask, dist, torch.ones_like(dist) * 1e6)
    min_dist = torch.min(dist, dim=-1).values

    return torch.clamp(min_dist, 0.0, max_error)

def nav2_heading_alignment_reward(env, asset_cfg: SceneEntityCfg, lookahead_index_offset: int = 4,) -> torch.Tensor:
    """Reward robot yaw alignment with local Nav2 path tangent."""
    robot_xy = _robot_xy(env, asset_cfg.name)
    robot_yaw = _robot_yaw(env, asset_cfg.name)

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    nearest_idx = _nearest_path_index(env, robot_xy)

    next_idx = torch.minimum(nearest_idx + lookahead_index_offset, valid_count - 1)
    next_idx = torch.clamp(next_idx, min=0)

    env_ids = torch.arange(env.num_envs, device=env.device)

    p0 = path[env_ids, nearest_idx]
    p1 = path[env_ids, next_idx]

    path_yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    err = _wrap_to_pi(path_yaw - robot_yaw)

    return torch.cos(err)

def final_goal_reward(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30,) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    return (dist < threshold).float()

def map_collision_penalty(env, asset_cfg: SceneEntityCfg, radius: float = 0.22, ) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,).float()

def action_smoothness_penalty(env) -> torch.Tensor:
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    current_action = env.action_manager.action

    if not hasattr(env, "navrl_prev_action_for_reward"):
        env.navrl_prev_action_for_reward = current_action.clone()

    diff = torch.norm(current_action - env.navrl_prev_action_for_reward, dim=-1)
    env.navrl_prev_action_for_reward[:] = current_action

    return diff

def time_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)