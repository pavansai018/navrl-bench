from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from .nav2_map import Nav2OccupancyMap

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


def previous_action(env) -> torch.Tensor:
    if hasattr(env, "action_manager"):
        return env.action_manager.action

    return torch.zeros(env.num_envs, 3, device=env.device)


def local_path_window(
    env,
    num_points: int = 8,
    step: int = 8,
) -> torch.Tensor:
    """Return future Nav2 path points in robot frame.

    Output shape:
        [num_envs, num_points * 2]

    This is the key observation for RL local-controller behavior.
    """
    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, num_points * 2, device=env.device)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    # Ignore padded zeros after valid_count by setting distance large.
    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)

    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]
    distances = torch.where(valid_mask, distances, torch.ones_like(distances) * 1e6)

    nearest_idx = torch.argmin(distances, dim=-1)

    out = torch.zeros(env.num_envs, num_points, 2, device=env.device)

    env_ids = torch.arange(env.num_envs, device=env.device)

    for k in range(num_points):
        idx = nearest_idx + (k + 1) * step
        idx = torch.minimum(idx, valid_count - 1)
        idx = torch.clamp(idx, min=0)

        p_world = path[env_ids, idx]
        rel = p_world - robot_xy

        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)

        x_b = cos_yaw * rel[:, 0] - sin_yaw * rel[:, 1]
        y_b = sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]

        out[:, k, 0] = x_b
        out[:, k, 1] = y_b

    return out.reshape(env.num_envs, num_points * 2)


def nav2_path_heading_error(env) -> torch.Tensor:
    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)

    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]
    distances = torch.where(valid_mask, distances, torch.ones_like(distances) * 1e6)

    nearest_idx = torch.argmin(distances, dim=-1)
    next_idx = torch.minimum(nearest_idx + 3, valid_count - 1)
    next_idx = torch.clamp(next_idx, min=0)

    env_ids = torch.arange(env.num_envs, device=env.device)

    p0 = path[env_ids, nearest_idx]
    p1 = path[env_ids, next_idx]

    heading = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    error = _wrap_to_pi(heading - yaw)

    return error.unsqueeze(-1)


def nav2_cross_track_error(env) -> torch.Tensor:
    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot_xy = _robot_xy(env)

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)

    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]
    distances = torch.where(valid_mask, distances, torch.ones_like(distances) * 1e6)

    error = torch.min(distances, dim=-1).values

    return error.unsqueeze(-1)



def map_collision_flag(
    env,
    radius: float = 0.22,
    num_points: int = 16,
) -> torch.Tensor:
    _ensure_nav2_map(env)

    robot_xy = _robot_xy(env)

    angles = torch.linspace(0.0, 2.0 * math.pi, num_points + 1, device=env.device)[:-1]
    offsets = torch.stack(
        [
            torch.cos(angles) * radius,
            torch.sin(angles) * radius,
        ],
        dim=-1,
    )

    footprint_points = robot_xy[:, None, :] + offsets[None, :, :]

    occupied = env.nav2_occupancy_map.is_occupied_world(
        footprint_points.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(env.num_envs, num_points)

    center_occupied = env.nav2_occupancy_map.is_occupied_world(
        robot_xy,
        unknown_is_occupied=True,
    )

    return occupied.any(dim=-1) | center_occupied

def _ensure_nav2_map(env):
    if not hasattr(env, "nav2_occupancy_map"):
        env.nav2_occupancy_map = Nav2OccupancyMap(
            map_yaml_path=env.cfg.nav2_map_yaml_path,
            device=env.device,
            inflation_radius_m=0.12,
        )


def map_based_scan(
    env,
    num_rays: int = 72,
    max_range: float = 4.0,
    step_size: float = 0.05,
) -> torch.Tensor:
    """Map-only lidar scan.

    This scan sees:
      - Nav2 occupancy map

    This scan does NOT see:
      - other robots
      - path markers
      - goal markers
      - visual map blocks
    """
    _ensure_nav2_map(env)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    return env.nav2_occupancy_map.raycast_scan(
        robot_xy=robot_xy,
        robot_yaw=yaw,
        num_rays=num_rays,
        max_range=max_range,
        step_size=step_size,
        unknown_is_occupied=True,
    )