from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import (
    _robot_xy,
    lookahead_distance,
    lookahead_angle,
    path_heading_error,
    cross_track_error,
)


def _safe_min_distance(robot_xy: torch.Tensor, points: torch.Tensor):
    if points.numel() == 0 or points.shape[1] == 0:
        return torch.ones(robot_xy.shape[0], device=robot_xy.device) * 999.0

    distances = torch.norm(points - robot_xy[:, None, :], dim=-1)
    return torch.min(distances, dim=-1).values


def progress_along_path(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    current_distance = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    if not hasattr(env, "navrl_previous_goal_distance"):
        env.navrl_previous_goal_distance = current_distance.clone()

    progress = env.navrl_previous_goal_distance - current_distance
    env.navrl_previous_goal_distance = current_distance.clone()

    return progress


def lookahead_tracking_reward(
    env,
    asset_cfg: SceneEntityCfg,
    distance_scale: float = 1.0,
    angle_scale: float = 1.0,
) -> torch.Tensor:
    dist = lookahead_distance(env).squeeze(-1)
    angle = torch.abs(lookahead_angle(env).squeeze(-1))

    reward = torch.exp(-distance_scale * dist) * torch.exp(-angle_scale * angle)
    return reward


def path_alignment_reward(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    error = torch.abs(path_heading_error(env).squeeze(-1))
    return torch.exp(-error)


def cross_track_error_penalty(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    error = cross_track_error(env).squeeze(-1)
    return error


def obstacle_clearance_reward(
    env,
    static_asset_cfg: SceneEntityCfg,
    dynamic_asset_cfg: SceneEntityCfg,
    safe_distance: float = 0.35,
) -> torch.Tensor:
    robot_xy = _robot_xy(env)

    points = []

    if hasattr(env, "navrl_static_xy"):
        n = getattr(env, "navrl_active_static_count", env.navrl_static_xy.shape[1])
        points.append(env.navrl_static_xy[:, :n, :])

    if hasattr(env, "navrl_dynamic_xy"):
        n = getattr(env, "navrl_active_dynamic_count", env.navrl_dynamic_xy.shape[1])
        points.append(env.navrl_dynamic_xy[:, :n, :])

    if len(points) == 0:
        return torch.ones(env.num_envs, device=env.device)

    obstacle_xy = torch.cat(points, dim=1)
    min_dist = _safe_min_distance(robot_xy, obstacle_xy)

    return torch.clamp(min_dist / safe_distance, 0.0, 1.0)


def collision_penalty(
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
        return torch.zeros(env.num_envs, device=env.device)

    obstacle_xy = torch.cat(points, dim=1)
    min_dist = _safe_min_distance(robot_xy, obstacle_xy)

    return (min_dist < collision_distance).float()


def dynamic_obstacle_clearance_reward(
    env,
    dynamic_asset_cfg: SceneEntityCfg,
    safe_distance: float = 0.45,
) -> torch.Tensor:
    robot_xy = _robot_xy(env)

    if not hasattr(env, "navrl_dynamic_xy"):
        return torch.ones(env.num_envs, device=env.device)

    n = getattr(env, "navrl_active_dynamic_count", env.navrl_dynamic_xy.shape[1])
    dyn_xy = env.navrl_dynamic_xy[:, :n, :]

    min_dist = _safe_min_distance(robot_xy, dyn_xy)

    return torch.clamp(min_dist / safe_distance, 0.0, 1.0)


def unnecessary_stop_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float = 0.04,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]

    if hasattr(robot.data, "root_lin_vel_b"):
        speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    else:
        speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)

    distance_to_lookahead = lookahead_distance(env).squeeze(-1)

    stopped = speed < speed_threshold
    still_has_target = distance_to_lookahead > 0.30

    return (stopped & still_has_target).float()


def action_smoothness_penalty(env) -> torch.Tensor:
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    current_action = env.action_manager.action

    if not hasattr(env, "navrl_previous_action"):
        env.navrl_previous_action = current_action.clone()

    penalty = torch.norm(current_action - env.navrl_previous_action, dim=-1)
    env.navrl_previous_action = current_action.clone()

    return penalty


def final_goal_reached_reward(
    env,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.30,
) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    distance = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    return (distance < threshold).float()


def constant_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)