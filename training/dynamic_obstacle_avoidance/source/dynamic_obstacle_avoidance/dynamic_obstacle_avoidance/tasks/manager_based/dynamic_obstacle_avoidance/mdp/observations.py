from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


def _env_origins(env, env_ids: torch.Tensor | None = None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if hasattr(env.scene, "env_origins"):
        return env.scene.env_origins[env_ids, :2]

    return torch.zeros(len(env_ids), 2, device=env.device)


def _robot_xy(env, asset_name: str = "robot"):
    robot = env.scene[asset_name]
    env_ids = torch.arange(env.num_envs, device=env.device)
    return robot.data.root_pos_w[:, :2] - _env_origins(env, env_ids)


def _robot_yaw(env, asset_name: str = "robot"):
    robot = env.scene[asset_name]
    q = robot.data.root_quat_w

    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return yaw


def _wrap_to_pi(angle: torch.Tensor):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _ensure_attr(env, name: str, shape):
    if not hasattr(env, name):
        setattr(env, name, torch.zeros(shape, device=env.device))


def lookahead_distance(env) -> torch.Tensor:
    _ensure_attr(env, "navrl_lookahead_xy", (env.num_envs, 2))
    robot_xy = _robot_xy(env)
    return torch.norm(env.navrl_lookahead_xy - robot_xy, dim=-1, keepdim=True)


def lookahead_angle(env) -> torch.Tensor:
    _ensure_attr(env, "navrl_lookahead_xy", (env.num_envs, 2))
    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    vec = env.navrl_lookahead_xy - robot_xy
    target_angle = torch.atan2(vec[:, 1], vec[:, 0])
    angle = _wrap_to_pi(target_angle - yaw)

    return angle.unsqueeze(-1)


def path_heading_error(env) -> torch.Tensor:
    _ensure_attr(env, "navrl_path_points", (env.num_envs, 32, 2))

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)
    path = env.navrl_path_points

    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)
    closest = torch.argmin(distances, dim=-1)

    next_ids = torch.clamp(closest + 1, max=path.shape[1] - 1)

    p0 = path[torch.arange(env.num_envs, device=env.device), closest]
    p1 = path[torch.arange(env.num_envs, device=env.device), next_ids]

    heading = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    error = _wrap_to_pi(heading - yaw)

    return error.unsqueeze(-1)


def cross_track_error(env) -> torch.Tensor:
    _ensure_attr(env, "navrl_path_points", (env.num_envs, 32, 2))

    robot_xy = _robot_xy(env)
    path = env.navrl_path_points

    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)
    error = torch.min(distances, dim=-1).values

    return error.unsqueeze(-1)


def base_lin_vel(env) -> torch.Tensor:
    robot = env.scene["robot"]

    if hasattr(robot.data, "root_lin_vel_b"):
        return robot.data.root_lin_vel_b[:, :2]

    return robot.data.root_lin_vel_w[:, :2]


def base_ang_vel(env) -> torch.Tensor:
    robot = env.scene["robot"]

    if hasattr(robot.data, "root_ang_vel_b"):
        return robot.data.root_ang_vel_b[:, 2:3]

    return robot.data.root_ang_vel_w[:, 2:3]


def obstacle_scan(
    env,
    static_asset_cfg: SceneEntityCfg,
    dynamic_asset_cfg: SceneEntityCfg,
    num_rays: int = 72,
    max_range: float = 4.0,
) -> torch.Tensor:
    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    angles = torch.linspace(-math.pi, math.pi, num_rays, device=env.device)
    world_angles = yaw[:, None] + angles[None, :]

    scan = torch.ones(env.num_envs, num_rays, device=env.device) * max_range

    points = []

    if hasattr(env, "navrl_static_xy"):
        n = getattr(env, "navrl_active_static_count", env.navrl_static_xy.shape[1])
        points.append(env.navrl_static_xy[:, :n, :])

    if hasattr(env, "navrl_dynamic_xy"):
        n = getattr(env, "navrl_active_dynamic_count", env.navrl_dynamic_xy.shape[1])
        points.append(env.navrl_dynamic_xy[:, :n, :])

    if len(points) == 0:
        return scan

    obstacle_xy = torch.cat(points, dim=1)

    rel = obstacle_xy[:, :, :] - robot_xy[:, None, :]
    dist = torch.norm(rel, dim=-1).clamp_min(1e-6)
    angle = torch.atan2(rel[:, :, 1], rel[:, :, 0])

    rel_angle = _wrap_to_pi(angle[:, :, None] - world_angles[:, None, :])

    angular_width = math.pi / num_rays
    mask = torch.abs(rel_angle) < angular_width

    dist_expanded = dist[:, :, None].expand_as(mask)
    dist_masked = torch.where(mask, dist_expanded, torch.ones_like(dist_expanded) * max_range)

    scan = torch.min(dist_masked, dim=1).values
    scan = torch.clamp(scan, 0.0, max_range)

    return scan / max_range


def nearest_dynamic_obstacle(
    env,
    asset_cfg: SceneEntityCfg,
    max_range: float = 4.0,
) -> torch.Tensor:
    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    if not hasattr(env, "navrl_dynamic_xy"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    n = getattr(env, "navrl_active_dynamic_count", env.navrl_dynamic_xy.shape[1])
    dyn_xy = env.navrl_dynamic_xy[:, :n, :]

    rel = dyn_xy - robot_xy[:, None, :]
    dist = torch.norm(rel, dim=-1)

    nearest_id = torch.argmin(dist, dim=-1)
    nearest_xy = dyn_xy[torch.arange(env.num_envs, device=env.device), nearest_id]
    nearest_rel = nearest_xy - robot_xy

    nearest_dist = torch.norm(nearest_rel, dim=-1)
    nearest_angle = torch.atan2(nearest_rel[:, 1], nearest_rel[:, 0])
    nearest_angle = _wrap_to_pi(nearest_angle - yaw)

    if hasattr(env, "navrl_dynamic_vel_xy"):
        dyn_vel = env.navrl_dynamic_vel_xy[:, :n, :]
        nearest_vel = dyn_vel[torch.arange(env.num_envs, device=env.device), nearest_id]
        nearest_speed = torch.norm(nearest_vel, dim=-1)
    else:
        nearest_speed = torch.zeros(env.num_envs, device=env.device)

    return torch.stack(
        [
            torch.clamp(nearest_dist / max_range, 0.0, 1.0),
            nearest_angle / math.pi,
            nearest_speed / 1.5,
        ],
        dim=-1,
    )


def previous_action(env) -> torch.Tensor:
    if hasattr(env, "action_manager"):
        return env.action_manager.action

    return torch.zeros(env.num_envs, 3, device=env.device)