from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy


def _safe_min_distance(robot_xy: torch.Tensor, points: torch.Tensor):
    if points.numel() == 0 or points.shape[1] == 0:
        return torch.ones(robot_xy.shape[0], device=robot_xy.device) * 999.0

    distances = torch.norm(points - robot_xy[:, None, :], dim=-1)
    return torch.min(distances, dim=-1).values


def final_goal_reached(
    env,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.30,
) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    distance = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    return distance < threshold


def collision_termination(
    env,
    robot_cfg: SceneEntityCfg,
    static_asset_cfg: SceneEntityCfg,
    dynamic_asset_cfg: SceneEntityCfg,
    collision_distance: float = 0.22,
) -> torch.Tensor:
    robot_xy = _robot_xy(env, robot_cfg.name)

    points = []

    if hasattr(env, "navrl_static_xy"):
        n = getattr(env, "navrl_active_static_count", env.navrl_static_xy.shape[1])
        points.append(env.navrl_static_xy[:, :n, :])

    if hasattr(env, "navrl_dynamic_xy"):
        n = getattr(env, "navrl_active_dynamic_count", env.navrl_dynamic_xy.shape[1])
        points.append(env.navrl_dynamic_xy[:, :n, :])

    if len(points) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obstacle_xy = torch.cat(points, dim=1)
    min_distance = _safe_min_distance(robot_xy, obstacle_xy)

    return min_distance < collision_distance


def robot_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg,
    x_bounds=(-6.0, 6.0),
    y_bounds=(-3.5, 3.5),
) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    out_x = (robot_xy[:, 0] < x_bounds[0]) | (robot_xy[:, 0] > x_bounds[1])
    out_y = (robot_xy[:, 1] < y_bounds[0]) | (robot_xy[:, 1] > y_bounds[1])

    return out_x | out_y