from __future__ import annotations

import math

import torch

from .nav2_map import Nav2OccupancyMap
from .observations import _robot_xy, _robot_yaw, _wrap_to_pi


def _teacher_cfg(env) -> dict:
    return dict(getattr(env.cfg, "mppi_teacher", {}) or {})


def _cfg(env, name: str, default):
    return _teacher_cfg(env).get(name, default)


def _ensure_nav2_map(env):
    if not hasattr(env, "nav2_occupancy_map"):
        env.nav2_occupancy_map = Nav2OccupancyMap(
            map_yaml_path=env.cfg.nav2_map_yaml_path,
            device=env.device,
            inflation_radius_m=float(_cfg(env, "static_map_inflation_radius_m", 0.12)),
        )


def _current_base_action(env) -> torch.Tensor:
    if hasattr(env, "action_manager") and "base_velocity" in env.action_manager._terms:
        return env.action_manager._terms["base_velocity"].processed_actions.detach()
    return torch.zeros(env.num_envs, 3, device=env.device)


def _nearest_path_index(env, robot_xy: torch.Tensor) -> torch.Tensor:
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    dist = torch.norm(path - robot_xy[:, None, :], dim=-1)
    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]
    dist = torch.where(valid_mask, dist, torch.ones_like(dist) * 1.0e6)
    return torch.argmin(dist, dim=-1)


def _path_reference(env, robot_xy: torch.Tensor, horizon: int, path_step_per_t: int):
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    nearest = _nearest_path_index(env, robot_xy)

    t = torch.arange(horizon, device=env.device)
    idx = nearest[:, None] + (t[None, :] + 1) * path_step_per_t
    idx = torch.minimum(idx, valid_count[:, None] - 1)
    idx = torch.clamp(idx, min=0)

    env_ids = torch.arange(robot_xy.shape[0], device=env.device)[:, None]
    ref = path[env_ids, idx]

    tangent_idx = torch.minimum(nearest + 4, valid_count - 1)
    p0 = path[torch.arange(robot_xy.shape[0], device=env.device), nearest]
    p1 = path[torch.arange(robot_xy.shape[0], device=env.device), tangent_idx]
    tangent = p1 - p0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)
    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    return ref, path_yaw


def _rate_limit_action_sequence(env, u: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
    """Apply the same style of action slew limit as KinematicMecanumAction.

    Args:
        u: [B, K, H, 3] physical action candidates.
        current_action: [B, 3] current physical action.
    """
    max_delta = torch.tensor(
        [
            float(_cfg(env, "max_delta_vx", getattr(env.cfg.actions.base_velocity, "max_delta_vx", 0.025))),
            float(_cfg(env, "max_delta_vy", getattr(env.cfg.actions.base_velocity, "max_delta_vy", 0.025))),
            float(_cfg(env, "max_delta_wz", getattr(env.cfg.actions.base_velocity, "max_delta_wz", 0.08))),
        ],
        device=u.device,
        dtype=u.dtype,
    )

    out = torch.empty_like(u)
    prev = current_action[:, None, :]
    for t in range(u.shape[2]):
        delta = torch.clamp(u[:, :, t, :] - prev, min=-max_delta, max=max_delta)
        prev = prev + delta
        out[:, :, t, :] = prev
    return out


def _rollout_mecanum(
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    u: torch.Tensor,
    dt: float,
):
    """Roll out holonomic/mecanum base kinematics.

    Args:
        robot_xy: [B, 2]
        robot_yaw: [B]
        u: [B, K, H, 3] physical [vx_b, vy_b, wz]

    Returns:
        xy: [B, K, H, 2]
        yaw: [B, K, H]
    """
    b, k, h, _ = u.shape
    xy = torch.empty(b, k, h, 2, device=u.device, dtype=u.dtype)
    yaw_out = torch.empty(b, k, h, device=u.device, dtype=u.dtype)

    x = robot_xy[:, None, 0].expand(b, k)
    y = robot_xy[:, None, 1].expand(b, k)
    yaw = robot_yaw[:, None].expand(b, k)

    for t in range(h):
        vx_b = u[:, :, t, 0]
        vy_b = u[:, :, t, 1]
        wz = u[:, :, t, 2]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx_w = cos_yaw * vx_b - sin_yaw * vy_b
        vy_w = sin_yaw * vx_b + cos_yaw * vy_b

        x = x + vx_w * dt
        y = y + vy_w * dt
        yaw = _wrap_to_pi(yaw + wz * dt)

        xy[:, :, t, 0] = x
        xy[:, :, t, 1] = y
        yaw_out[:, :, t] = yaw

    return xy, yaw_out


def _static_map_cost(env, xy: torch.Tensor, robot_radius: float, num_footprint_points: int) -> torch.Tensor:
    """Static map collision cost for rollout xy.

    Args:
        xy: [B, K, H, 2]

    Returns:
        cost: [B, K]
    """
    _ensure_nav2_map(env)

    b, k, h, _ = xy.shape
    center_occ = env.nav2_occupancy_map.is_occupied_world(
        xy.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(b, k, h)

    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        num_footprint_points + 1,
        device=xy.device,
    )[:-1]
    offsets = torch.stack(
        [
            torch.cos(angles) * robot_radius,
            torch.sin(angles) * robot_radius,
        ],
        dim=-1,
    )

    footprint = xy[:, :, :, None, :] + offsets[None, None, None, :, :]
    footprint_occ = env.nav2_occupancy_map.is_occupied_world(
        footprint.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(b, k, h, num_footprint_points)

    collision = center_occ | footprint_occ.any(dim=-1)
    return collision.float().sum(dim=-1)


def _dynamic_obstacle_cost(
    env,
    xy: torch.Tensor,
    robot_radius: float,
    clearance: float,
    dt: float,
):
    """Dynamic obstacle collision and clearance cost.

    Args:
        xy: [B, K, H, 2]

    Returns:
        collision_cost: [B, K]
        clearance_cost: [B, K]
    """
    if not hasattr(env, "dyn_obs_xy"):
        zeros = torch.zeros(xy.shape[0], xy.shape[1], device=xy.device)
        return zeros, zeros

    b, k, h, _ = xy.shape
    n = env.dyn_obs_xy.shape[1]

    t = torch.arange(1, h + 1, device=xy.device, dtype=xy.dtype) * dt
    obs_xy = env.dyn_obs_xy[:b, None, None, :, :] + env.dyn_obs_vel_xy[:b, None, None, :, :] * t[None, None, :, None, None]
    obs_radius = env.dyn_obs_radius[:b, None, None, :]
    active = env.dyn_obs_active[:b, None, None, :]

    d = torch.norm(xy[:, :, :, None, :] - obs_xy, dim=-1)
    collision_dist = obs_radius + robot_radius
    clear_dist = collision_dist + clearance

    collision = active & (d < collision_dist)
    clearance_violation = torch.clamp(clear_dist - d, min=0.0) / max(clearance, 1.0e-6)
    clearance_violation = torch.where(active, clearance_violation, torch.zeros_like(clearance_violation))

    return collision.float().sum(dim=(-1, -2)), clearance_violation.sum(dim=(-1, -2))


@torch.no_grad()
def compute_mppi_teacher_action(env, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Compute vectorized torch-MPPI teacher action.

    Returns physical base action [vx, vy, wz] with shape [len(env_ids), 3].
    This is intended as a teacher label/reward target, not as a runtime ROS2 controller.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "navrl_global_path_xy") or not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(len(env_ids), 3, device=env.device)

    num_samples = int(_cfg(env, "num_samples", 96))
    horizon = int(_cfg(env, "horizon", 20))
    dt = float(_cfg(env, "dt", getattr(env, "step_dt", 1.0 / 30.0)))
    temperature = float(_cfg(env, "temperature", 0.35))
    chunk_size = int(_cfg(env, "env_chunk_size", 128))

    max_vx = float(getattr(env.cfg.actions.base_velocity, "max_vx", 0.5))
    max_vy = float(getattr(env.cfg.actions.base_velocity, "max_vy", 0.5))
    max_wz = float(getattr(env.cfg.actions.base_velocity, "max_wz", 1.5))

    vx_std = float(_cfg(env, "vx_std", 0.22))
    vy_std = float(_cfg(env, "vy_std", 0.32))
    wz_std = float(_cfg(env, "wz_std", 0.50))

    path_step_per_t = int(_cfg(env, "path_step_per_t", 2))
    robot_radius = float(_cfg(env, "robot_radius", 0.22))
    clearance = float(_cfg(env, "dynamic_clearance", 0.20))
    num_footprint_points = int(_cfg(env, "num_footprint_points", 8))

    w_static_collision = float(_cfg(env, "w_static_collision", 5000.0))
    w_dynamic_collision = float(_cfg(env, "w_dynamic_collision", 5000.0))
    w_dynamic_clearance = float(_cfg(env, "w_dynamic_clearance", 40.0))
    w_path = float(_cfg(env, "w_path", 8.0))
    w_goal = float(_cfg(env, "w_goal", 6.0))
    w_heading = float(_cfg(env, "w_heading", 1.0))
    w_wait = float(_cfg(env, "w_wait", 2.0))
    w_action = float(_cfg(env, "w_action", 0.05))
    w_smooth = float(_cfg(env, "w_smooth", 0.08))

    result = torch.zeros(len(env_ids), 3, device=env.device)

    full_robot_xy = _robot_xy(env)
    full_robot_yaw = _robot_yaw(env)
    full_current_action = _current_base_action(env)

    for start in range(0, len(env_ids), chunk_size):
        local_ids = env_ids[start : start + chunk_size]
        b = len(local_ids)

        robot_xy = full_robot_xy[local_ids]
        robot_yaw = full_robot_yaw[local_ids]
        current_action = full_current_action[local_ids]

        # Temporarily index path-related tensors by local env ids without copying the whole env.
        path_backup = env.navrl_global_path_xy
        count_backup = env.navrl_path_valid_count
        goal_backup = env.navrl_final_goal_xy
        dyn_backup = None
        try:
            env.navrl_global_path_xy = path_backup[local_ids]
            env.navrl_path_valid_count = count_backup[local_ids]
            env.navrl_final_goal_xy = goal_backup[local_ids]

            if hasattr(env, "dyn_obs_xy"):
                dyn_backup = (
                    env.dyn_obs_xy,
                    env.dyn_obs_vel_xy,
                    env.dyn_obs_radius,
                    env.dyn_obs_active,
                )
                env.dyn_obs_xy = dyn_backup[0][local_ids]
                env.dyn_obs_vel_xy = dyn_backup[1][local_ids]
                env.dyn_obs_radius = dyn_backup[2][local_ids]
                env.dyn_obs_active = dyn_backup[3][local_ids]

            ref_path, ref_path_yaw = _path_reference(env, robot_xy, horizon, path_step_per_t)
            goal_xy = env.navrl_final_goal_xy

            mean = current_action[:, None, None, :].expand(b, num_samples, horizon, 3)
            noise = torch.randn(b, num_samples, horizon, 3, device=env.device)
            noise[..., 0] *= vx_std
            noise[..., 1] *= vy_std
            noise[..., 2] *= wz_std
            u = mean + noise

            # Keep a few deterministic useful candidates.
            if num_samples >= 1:
                u[:, 0, :, :] = current_action[:, None, :]
            if num_samples >= 2:
                u[:, 1, :, :] = torch.tensor([0.25 * max_vx, 0.0, 0.0], device=env.device)
            if num_samples >= 3:
                u[:, 2, :, :] = torch.tensor([0.20 * max_vx, 0.45 * max_vy, 0.0], device=env.device)
            if num_samples >= 4:
                u[:, 3, :, :] = torch.tensor([0.20 * max_vx, -0.45 * max_vy, 0.0], device=env.device)
            if num_samples >= 5:
                u[:, 4, :, :] = torch.tensor([0.0, 0.55 * max_vy, 0.0], device=env.device)
            if num_samples >= 6:
                u[:, 5, :, :] = torch.tensor([0.0, -0.55 * max_vy, 0.0], device=env.device)

            u[..., 0] = torch.clamp(u[..., 0], -max_vx, max_vx)
            u[..., 1] = torch.clamp(u[..., 1], -max_vy, max_vy)
            u[..., 2] = torch.clamp(u[..., 2], -max_wz, max_wz)
            u = _rate_limit_action_sequence(env, u, current_action)

            xy, yaw = _rollout_mecanum(robot_xy, robot_yaw, u, dt)

            static_col = _static_map_cost(env, xy, robot_radius, num_footprint_points)
            dyn_col, dyn_clear = _dynamic_obstacle_cost(env, xy, robot_radius, clearance, dt)

            path_error = torch.norm(xy - ref_path[:, None, :, :], dim=-1).mean(dim=-1)
            final_goal_error = torch.norm(xy[:, :, -1, :] - goal_xy[:, None, :], dim=-1)

            heading_error = 1.0 - torch.cos(_wrap_to_pi(yaw[:, :, -1] - ref_path_yaw[:, None]))
            speed = torch.norm(u[..., :2], dim=-1)
            wait_cost = torch.clamp(0.12 - speed, min=0.0).mean(dim=-1)
            action_cost = torch.sum(u * u, dim=-1).mean(dim=-1)
            smooth_cost = torch.sum((u[:, :, 1:, :] - u[:, :, :-1, :]) ** 2, dim=-1).mean(dim=-1)

            cost = (
                w_static_collision * static_col
                + w_dynamic_collision * dyn_col
                + w_dynamic_clearance * dyn_clear
                + w_path * path_error
                + w_goal * final_goal_error
                + w_heading * heading_error
                + w_wait * wait_cost
                + w_action * action_cost
                + w_smooth * smooth_cost
            )

            cost = cost - cost.min(dim=1, keepdim=True).values
            weights = torch.softmax(-cost / max(temperature, 1.0e-6), dim=1)
            action = torch.sum(weights[:, :, None] * u[:, :, 0, :], dim=1)
            result[start : start + b] = action

        finally:
            env.navrl_global_path_xy = path_backup
            env.navrl_path_valid_count = count_backup
            env.navrl_final_goal_xy = goal_backup
            if dyn_backup is not None:
                env.dyn_obs_xy, env.dyn_obs_vel_xy, env.dyn_obs_radius, env.dyn_obs_active = dyn_backup

    return result


@torch.no_grad()
def get_mppi_teacher_action(env) -> torch.Tensor:
    """Cached teacher action for reward use.

    The cache avoids solving MPPI multiple times if multiple reward/metric terms ask
    for the teacher during the same control step.
    """
    recompute_interval = int(_cfg(env, "recompute_interval", 1))

    if hasattr(env, "common_step_counter"):
        counter = int(env.common_step_counter)
    else:
        if not hasattr(env, "navrl_mppi_teacher_fallback_counter"):
            env.navrl_mppi_teacher_fallback_counter = 0
        env.navrl_mppi_teacher_fallback_counter += 1
        counter = int(env.navrl_mppi_teacher_fallback_counter)
    if not hasattr(env, "navrl_mppi_teacher_action"):
        env.navrl_mppi_teacher_action = torch.zeros(env.num_envs, 3, device=env.device)
        env.navrl_mppi_teacher_counter = -10**9

    should_recompute = (counter - int(env.navrl_mppi_teacher_counter)) >= recompute_interval
    if should_recompute:
        env.navrl_mppi_teacher_action[:] = compute_mppi_teacher_action(env)
        env.navrl_mppi_teacher_counter = counter

    return env.navrl_mppi_teacher_action
