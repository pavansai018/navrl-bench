from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
        term = env.action_manager._terms["base_velocity"]
        a = term.processed_actions

        out = torch.zeros_like(a)
        out[:, 0] = a[:, 0] / term.cfg.max_vx
        out[:, 1] = a[:, 1] / term.cfg.max_vy
        out[:, 2] = a[:, 2] / term.cfg.max_wz

        return torch.clamp(out, -1.0, 1.0)

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

    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)

    for k in range(num_points):
        idx = nearest_idx + (k + 1) * step
        idx = torch.minimum(idx, valid_count - 1)
        idx = torch.clamp(idx, min=0)

        p_world = path[env_ids, idx]
        rel = p_world - robot_xy

        x_b = cos_yaw * rel[:, 0] - sin_yaw * rel[:, 1]
        y_b = sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]

        out[:, k, 0] = x_b
        out[:, k, 1] = y_b

    # Keep scale stable. 4m lookahead should not dominate scan/velocity obs.
    norm_scale = float(getattr(env.cfg, "path_window_normalization_m", 4.0))
    return torch.clamp(out.reshape(env.num_envs, num_points * 2) / norm_scale, -2.0, 2.0)

def _nearest_path_index(env, robot_xy: torch.Tensor) -> torch.Tensor:
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    dist = torch.norm(path - robot_xy[:, None, :], dim=-1)
    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    valid_mask = ids < valid_count[:, None]
    dist = torch.where(valid_mask, dist, torch.ones_like(dist) * 1e6)
    return torch.argmin(dist, dim=-1)

def nav2_path_heading_error(env) -> torch.Tensor:
    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    nearest_idx = _nearest_path_index(env, robot_xy)
    next_idx = torch.minimum(nearest_idx + 4, valid_count - 1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    p0 = path[env_ids, nearest_idx]
    p1 = path[env_ids, next_idx]
    heading = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    return _wrap_to_pi(heading - yaw).unsqueeze(-1)


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

def _ensure_dynamic_buffers(env):
    if not hasattr(env, "dyn_obs_xy"):
        num_obs = int(getattr(env.cfg, "max_dynamic_obstacles", 6))
        env.dyn_obs_xy = torch.zeros(env.num_envs, num_obs, 2, device=env.device)
        env.dyn_obs_vel_xy = torch.zeros(env.num_envs, num_obs, 2, device=env.device)
        env.dyn_obs_radius = torch.zeros(env.num_envs, num_obs, device=env.device)
        env.dyn_obs_active = torch.zeros(env.num_envs, num_obs, dtype=torch.bool, device=env.device)


def map_based_scan(env, num_rays: int = 360, max_range: float = 4.0, step_size: float = 0.05,) -> torch.Tensor:
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
        robot_xy=robot_xy, robot_yaw=yaw, num_rays=num_rays, max_range=max_range, step_size=step_size, unknown_is_occupied=True,
    )

def distance_to_final_goal(env) -> torch.Tensor:
    if not hasattr(env, 'navrl_final_goal_xy'):
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    robot_xy = _robot_xy(env)
    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    return dist.unsqueeze(-1)

def nav2_path_progress_fraction(env) -> torch.Tensor:
    if not hasattr(env, 'navrl_global_path_xy') or not hasattr(env, 'navrl_path_cum_s'):
        return torch.zeros(env.num_envs, 1, device=env.device)
    robot_xy = _robot_xy(env)
    nearest_idx = _nearest_path_index(env, robot_xy)
    env_ids = torch.arange(env.num_envs, device=env.device)
    current_s = env.navrl_path_cum_s[env_ids, nearest_idx]
    total_s = env.navrl_path_cum_s[env_ids, env.navrl_path_valid_count - 1].clamp_min(1e-6)
    return (current_s / total_s).unsqueeze(-1)

def map_collision_observation(env) -> torch.Tensor:
    return map_collision_flag(
        env,
        radius=0.22,
        num_points=16,
    ).float().unsqueeze(-1)


def dynamic_obstacle_scan(env, num_rays: int = 360, max_range: float = 4.0) -> torch.Tensor:
    """Analytic local lidar scan against tensor dynamic circular obstacles."""
    _ensure_dynamic_buffers(env)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)
    device = env.device

    # ray_angles = torch.linspace(-math.pi, math.pi, num_rays, device=device)
    ray_angles = torch.linspace(-math.pi, math.pi,num_rays + 1,device=device,)[:-1]
    world_angles = yaw[:, None] + ray_angles[None, :]
    ray_dir = torch.stack([torch.cos(world_angles), torch.sin(world_angles)], dim=-1)  # [E, R, 2]

    obs_rel = env.dyn_obs_xy[:, None, :, :] - robot_xy[:, None, None, :]              # [E, 1, O, 2]
    d = ray_dir[:, :, None, :]                                                       # [E, R, 1, 2]

    proj = torch.sum(obs_rel * d, dim=-1)                                            # [E, R, O]
    perp_vec = obs_rel - proj[..., None] * d
    perp_dist = torch.norm(perp_vec, dim=-1)

    radius = env.dyn_obs_radius[:, None, :]
    active = env.dyn_obs_active[:, None, :]
    valid = active & (proj > 0.0) & (proj < max_range) & (perp_dist <= radius)

    chord = torch.sqrt(torch.clamp(radius * radius - perp_dist * perp_dist, min=0.0))
    hit_dist = torch.clamp(proj - chord, min=0.0, max=max_range)
    hit_dist = torch.where(valid, hit_dist, torch.ones_like(hit_dist) * max_range)

    scan = torch.min(hit_dist, dim=-1).values
    return torch.clamp(scan / max_range, 0.0, 1.0)


def combined_static_dynamic_scan(env, num_rays: int = 360, max_range: float = 4.0, step_size: float = 0.05) -> torch.Tensor:
    """Deployment-style local scan: nearest hit from static map or dynamic obstacles."""
    max_rays = int(getattr(env.cfg, "lidar_max_rays", num_rays))
    if hasattr(env, "dr_lidar_rays"):
        current_rays = int(env.dr_lidar_rays.max().item())
    else:
        current_rays = max_rays
    static_scan = map_based_scan(env, num_rays=num_rays, max_range=max_range, step_size=step_size)
    dyn_scan = dynamic_obstacle_scan(env, num_rays=num_rays, max_range=max_range)
    scan = torch.minimum(static_scan, dyn_scan)
    if current_rays != max_rays:
        scan = F.interpolate(
            scan.unsqueeze(1),
            size=max_rays,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    scan = _apply_scan_domain_randomization(env, scan)
    return scan

def scan_history(
    env,
    history_len: int = 8,
    num_rays: int = 144,
    max_range: float = 10.0,
    step_size: float = 0.10,
) -> torch.Tensor:
    """Cached scan history.

    Policy and critic both use scan_history.
    Without caching, the scan is recomputed and rolled twice per env step.

    This function updates only once per env step and returns the same flattened
    history to both policy and critic.
    """

    device = env.device
    expected_shape = (env.num_envs, history_len, num_rays)

    step_id = int(env.common_step_counter)

    need_init = (
        not hasattr(env, "navrl_scan_history")
        or env.navrl_scan_history.shape != expected_shape
        or env.navrl_scan_history.device != device
    )

    if need_init:
        scan = combined_static_dynamic_scan(
            env,
            num_rays=num_rays,
            max_range=max_range,
            step_size=step_size,
        )

        env.navrl_scan_history = scan[:, None, :].repeat(1, history_len, 1)
        env.navrl_scan_history_last_step = step_id

        return env.navrl_scan_history.reshape(env.num_envs, history_len * num_rays)

    last_step = getattr(env, "navrl_scan_history_last_step", -1)

    if last_step != step_id:
        scan = combined_static_dynamic_scan(
            env,
            num_rays=num_rays,
            max_range=max_range,
            step_size=step_size,
        )

        env.navrl_scan_history = torch.roll(
            env.navrl_scan_history,
            shifts=-1,
            dims=1,
        )
        env.navrl_scan_history[:, -1, :] = scan
        env.navrl_scan_history_last_step = step_id

    return env.navrl_scan_history.reshape(env.num_envs, history_len * num_rays)

def dynamic_obstacle_states(env, num_obstacles: int = 4, max_range: float = 4.0) -> torch.Tensor:
    """Nearest dynamic obstacles in robot frame: [x, y, vx, vy, radius, active] per obstacle.

    This is local/predictive context, not global map context. It gives velocity information
    that a single scan frame cannot provide.
    """
    _ensure_dynamic_buffers(env)

    robot_xy = _robot_xy(env)
    yaw = _robot_yaw(env)
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)

    rel_w = env.dyn_obs_xy - robot_xy[:, None, :]
    dist = torch.norm(rel_w, dim=-1)
    dist_masked = torch.where(env.dyn_obs_active, dist, torch.ones_like(dist) * 1e6)
    k = min(num_obstacles, env.dyn_obs_xy.shape[1])
    ids = torch.topk(dist_masked, k=k, dim=-1, largest=False).indices

    env_ids = torch.arange(env.num_envs, device=env.device)[:, None]
    rel = rel_w[env_ids, ids]
    vel_w = env.dyn_obs_vel_xy[env_ids, ids]
    rad = env.dyn_obs_radius[env_ids, ids]
    active = env.dyn_obs_active[env_ids, ids].float()

    x_b = cos_yaw[:, None] * rel[..., 0] - sin_yaw[:, None] * rel[..., 1]
    y_b = sin_yaw[:, None] * rel[..., 0] + cos_yaw[:, None] * rel[..., 1]
    vx_b = cos_yaw[:, None] * vel_w[..., 0] - sin_yaw[:, None] * vel_w[..., 1]
    vy_b = sin_yaw[:, None] * vel_w[..., 0] + cos_yaw[:, None] * vel_w[..., 1]

    out = torch.stack([
        torch.clamp(x_b / max_range, -1.5, 1.5),
        torch.clamp(y_b / max_range, -1.5, 1.5),
        torch.clamp(vx_b / 1.0, -1.5, 1.5),
        torch.clamp(vy_b / 1.0, -1.5, 1.5),
        torch.clamp(rad / 0.5, 0.0, 2.0),
        active,
    ], dim=-1)

    if k < num_obstacles:
        pad = torch.zeros(env.num_envs, num_obstacles - k, 6, device=env.device)
        out = torch.cat([out, pad], dim=1)

    return out.reshape(env.num_envs, num_obstacles * 6)


def dynamic_path_blockage(env, lookahead_points: int = 32, path_radius: float = 0.35) -> torch.Tensor:
    """Scalar: 1 if active obstacle overlaps the near-future path corridor."""
    _ensure_dynamic_buffers(env)
    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot_xy = _robot_xy(env)
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    nearest_idx = _nearest_path_index(env, robot_xy)
    env_ids = torch.arange(env.num_envs, device=env.device)

    blocked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # for k in range(lookahead_points):
    #     idx = torch.minimum(nearest_idx + k, valid_count - 1)
    for k in range(-8, lookahead_points):
        idx = torch.clamp(nearest_idx + k, min=0)
        idx = torch.minimum(idx, valid_count - 1)
        p = path[env_ids, idx]
        d = torch.norm(env.dyn_obs_xy - p[:, None, :], dim=-1)
        hit = env.dyn_obs_active & (d < (env.dyn_obs_radius + path_radius))
        blocked = blocked | hit.any(dim=-1)
    env.navrl_path_blocked = blocked
    return blocked.float().unsqueeze(-1)

def dynamic_obstacle_collision_flag(env, robot_radius: float = 0.22, safety_margin: float = 0.02) -> torch.Tensor:
    _ensure_dynamic_buffers(env)
    robot_xy = _robot_xy(env)
    d = torch.norm(env.dyn_obs_xy - robot_xy[:, None, :], dim=-1)
    collision = env.dyn_obs_active & (d < (env.dyn_obs_radius + robot_radius + safety_margin))
    return collision.any(dim=-1)

def dynamic_collision_observation(env) -> torch.Tensor:
    return dynamic_obstacle_collision_flag(env).float().unsqueeze(-1)

def _apply_scan_domain_randomization(env, scan: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(env.cfg, "dr_enable", True)):
        return torch.clamp(scan, 0.0, 1.0)

    if hasattr(env, "dr_scan_noise_std"):
        noise = torch.randn_like(scan) * env.dr_scan_noise_std[:, None]
        scan = scan + noise

    if hasattr(env, "dr_scan_dropout_prob"):
        dropout = torch.rand_like(scan) < env.dr_scan_dropout_prob[:, None]
        scan = torch.where(dropout, torch.ones_like(scan), scan)

    return torch.clamp(scan, 0.0, 1.0)

def time_to_closest_approach(
    env,
    num_obstacles: int = 2,
    max_range: float = 4.0,
    horizon_s: float = 3.0,
) -> torch.Tensor:
    """
    For the N nearest dynamic obstacles, compute:
        - time to closest approach (TCA): seconds until the gap between
          robot and obstacle is at its minimum, assuming constant velocities
        - distance at closest approach (DCA): how close they will actually get

    Output shape: [num_envs, num_obstacles * 2]
        [tca_0, dca_0, tca_1, dca_1, ...]

    Both are normalized to [0, 1]:
        - tca normalized by horizon_s
        - dca normalized by max_range

    Why this helps:
        A single-frame observation cannot tell the robot "I have 1.5 seconds to
        sidestep this obstacle." TCA gives the neural network a direct signal
        for HOW URGENTLY it needs to start a lateral maneuver.

        With TCA, the robot learns:
            TCA=2.0s -> gradual lateral slide at high speed
            TCA=0.3s -> sharp emergency turn

        Without TCA, the robot only sees "obstacle is close" and reacts late.
    """
    _ensure_dynamic_buffers(env)

    robot = env.scene["robot"]
    robot_xy = _robot_xy(env)
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    # Relative position and velocity
    rel_p = env.dyn_obs_xy - robot_xy[:, None, :]          # [E, O, 2]
    rel_v = env.dyn_obs_vel_xy - robot_vel[:, None, :]     # [E, O, 2]

    rel_v2 = torch.sum(rel_v * rel_v, dim=-1).clamp_min(1e-6)  # [E, O]

    # TCA: time of closest approach
    # t* = -dot(rel_p, rel_v) / |rel_v|^2
    # Clamped to [0, horizon_s]: only future approaches within horizon
    tca = -torch.sum(rel_p * rel_v, dim=-1) / rel_v2           # [E, O]
    tca = torch.clamp(tca, 0.0, horizon_s)

    # DCA: position at closest approach
    closest_point = rel_p + rel_v * tca[..., None]             # [E, O, 2]
    dca = torch.norm(closest_point, dim=-1)                     # [E, O]

    # Mask inactive obstacles
    # Inactive obstacles get tca=horizon_s (far future) and dca=max_range (far away)
    tca = torch.where(env.dyn_obs_active, tca, torch.ones_like(tca) * horizon_s)
    dca = torch.where(env.dyn_obs_active, dca, torch.ones_like(dca) * max_range)

    # Sort by urgency: smallest TCA first among active obstacles
    dist_now = torch.norm(rel_p, dim=-1)
    dist_masked = torch.where(env.dyn_obs_active, dist_now, torch.ones_like(dist_now) * 1e6)
    k = min(num_obstacles, env.dyn_obs_xy.shape[1])
    ids = torch.topk(dist_masked, k=k, dim=-1, largest=False).indices

    env_ids = torch.arange(env.num_envs, device=env.device)[:, None]
    tca_k = tca[env_ids, ids]                                  # [E, k]
    dca_k = dca[env_ids, ids]                                  # [E, k]

    # Normalize
    tca_norm = torch.clamp(tca_k / horizon_s, 0.0, 1.0)
    dca_norm = torch.clamp(dca_k / max_range, 0.0, 1.0)

    # Interleave: [tca_0, dca_0, tca_1, dca_1, ...]
    out = torch.stack([tca_norm, dca_norm], dim=-1)             # [E, k, 2]

    if k < num_obstacles:
        pad = torch.ones(env.num_envs, num_obstacles - k, 2, device=env.device)
        # Pad with "safe" values: tca=1.0 (far future), dca=1.0 (far away)
        out = torch.cat([out, pad], dim=1)

    return out.reshape(env.num_envs, num_obstacles * 2)

def map_collision_direction_flags(
    env,
    radius: float = 0.22,
    num_points: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Returns collision direction flags:
        front, left, right, rear, center

    Directions are relative to robot heading.
    """

    _ensure_nav2_map(env)

    robot_xy = _robot_xy(env)
    robot_yaw = _robot_yaw(env)

    angles = torch.linspace(
        -math.pi,
        math.pi,
        num_points + 1,
        device=env.device,
    )[:-1]

    # Convert robot-relative footprint points to world/map frame.
    local_offsets = torch.stack(
        [
            torch.cos(angles) * radius,
            torch.sin(angles) * radius,
        ],
        dim=-1,
    )

    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    x = local_offsets[:, 0][None, :]
    y = local_offsets[:, 1][None, :]

    wx = cos_yaw[:, None] * x - sin_yaw[:, None] * y
    wy = sin_yaw[:, None] * x + cos_yaw[:, None] * y

    footprint_points = robot_xy[:, None, :] + torch.stack([wx, wy], dim=-1)

    occupied = env.nav2_occupancy_map.is_occupied_world(
        footprint_points.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(env.num_envs, num_points)

    center_hit = env.nav2_occupancy_map.is_occupied_world(
        robot_xy,
        unknown_is_occupied=True,
    )

    # Angles are robot-relative:
    # front: -45° to +45°
    # left : +45° to +135°
    # right: -135° to -45°
    # rear : outside the above, around ±180°
    front_mask = torch.abs(angles) <= (math.pi / 4.0)
    left_mask = (angles > math.pi / 4.0) & (angles <= 3.0 * math.pi / 4.0)
    right_mask = (angles < -math.pi / 4.0) & (angles >= -3.0 * math.pi / 4.0)
    rear_mask = torch.abs(angles) > 3.0 * math.pi / 4.0

    front_hit = occupied[:, front_mask].any(dim=-1)
    left_hit = occupied[:, left_mask].any(dim=-1)
    right_hit = occupied[:, right_mask].any(dim=-1)
    rear_hit = occupied[:, rear_mask].any(dim=-1)

    return {
        "front": front_hit,
        "left": left_hit,
        "right": right_hit,
        "rear": rear_hit,
        "center": center_hit,
    }