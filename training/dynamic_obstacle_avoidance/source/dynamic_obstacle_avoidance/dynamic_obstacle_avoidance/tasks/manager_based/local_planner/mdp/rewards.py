from isaaclab.managers import SceneEntityCfg
from .observations import dynamic_obstacle_collision_flag, map_collision_flag, build_rl_local_path, get_rl_local_path, dynamic_path_blockage
import torch

def dynamic_obstacle_collision_penalty(env, asset_cfg: SceneEntityCfg, robot_radius: float = 0.22) -> torch.Tensor:
    return dynamic_obstacle_collision_flag(env, robot_radius=robot_radius).float()

def map_collision_penalty(env, asset_cfg: SceneEntityCfg, radius: float = 0.22, ) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,).float()

def local_path_smoothness_penalty(env):
    if not hasattr(env, "rl_local_path_offsets"):
        return torch.zeros(env.num_envs, device=env.device)

    offsets = env.rl_local_path_offsets
    diff = offsets[:, 1:] - offsets[:, :-1]
    return torch.mean(diff * diff, dim=1)

def local_path_rejoin_penalty(env, num_points: int = 8):
    if not hasattr(env, "rl_local_path_offsets"):
        return torch.zeros(env.num_envs, device=env.device)

    return torch.abs(env.rl_local_path_offsets[:, num_points - 1])


def local_path_dynamic_clearance_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    num_points: int = 8,
    safe_distance: float = 0.35,
):
    if not hasattr(env, "dyn_obs_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    build_rl_local_path(env, num_points=num_points)

    pts = env.rl_local_path_world  # [E, P, 2]

    diff = env.dyn_obs_xy[:, :, None, :] - pts[:, None, :, :]
    dist = torch.norm(diff, dim=-1)

    clearance = dist - env.dyn_obs_radius[:, :, None]

    active_clearance = torch.where(
        env.dyn_obs_active[:, :, None],
        clearance,
        torch.ones_like(clearance) * 10.0,
    )

    min_clear = active_clearance.min(dim=1).values.min(dim=1).values
    return torch.clamp(safe_distance - min_clear, min=0.0) / safe_distance

def local_path_static_clearance_penalty(
    env,
    num_points: int = 8,
    safe_distance: float = 0.30,
):
    if not hasattr(env, "nav2_occupancy_map"):
        return torch.zeros(env.num_envs, device=env.device)

    # build_rl_local_path(env, num_points=num_points)
    # pts = env.rl_local_path_world.reshape(-1, 2)
    pts, _ = get_rl_local_path(env, num_points=num_points, step=4)

    occupied = env.nav2_occupancy_map.is_occupied_world(
        pts,
        unknown_is_occupied=True,
    ).reshape(env.num_envs, num_points)

    return occupied.float().max(dim=1).values

def progress_along_global_path(env, asset_cfg: SceneEntityCfg, max_step_progress: float = 0.05):
    if not hasattr(env, "navrl_path_cum_s"):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count

    dist = torch.norm(path - robot_xy[:, None, :], dim=-1)
    ids = torch.arange(path.shape[1], device=env.device)[None, :]
    dist = torch.where(ids < valid_count[:, None], dist, torch.ones_like(dist) * 1e6)

    nearest_idx = torch.argmin(dist, dim=-1)
    env_ids = torch.arange(env.num_envs, device=env.device)

    progress_s = env.navrl_path_cum_s[env_ids, nearest_idx]

    if not hasattr(env, "navrl_prev_progress_s"):
        env.navrl_prev_progress_s = progress_s.clone()

    delta = progress_s - env.navrl_prev_progress_s
    env.navrl_prev_progress_s[:] = progress_s

    return torch.clamp(delta, 0.0, max_step_progress) / max_step_progress

def final_goal_reward(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30):
    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    return (dist < threshold).float()

def stuck_penalty(env, asset_cfg: SceneEntityCfg, speed_threshold: float = 0.05):
    robot = env.scene[asset_cfg.name]
    speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    return (speed < speed_threshold).float()

def conditional_offset_penalty(
    env,
    lookahead_points: int = 32,
    path_radius: float = 0.35,
    free_scale: float = 1.0,
    blocked_scale: float = 0.10,
):

    if not hasattr(env, "rl_local_path_offsets"):
        return torch.zeros(env.num_envs, device=env.device)

    blocked = dynamic_path_blockage(
        env,
        lookahead_points=lookahead_points,
        path_radius=path_radius,
    ).squeeze(-1)

    offsets = env.rl_local_path_offsets

    # Penalize all offset magnitude.
    mag = torch.mean(torch.abs(offsets), dim=1)

    scale = torch.where(
        blocked > 0.5,
        torch.full_like(mag, blocked_scale),
        torch.full_like(mag, free_scale),
    )

    return mag * scale

def conditional_structured_offset_penalty(
    env,
    lookahead_points: int = 32,
    path_radius: float = 0.35,
):

    if not hasattr(env, "rl_local_path_offsets"):
        return torch.zeros(env.num_envs, device=env.device)

    blocked = dynamic_path_blockage(
        env,
        lookahead_points=lookahead_points,
        path_radius=path_radius,
    ).squeeze(-1)

    offsets = torch.abs(env.rl_local_path_offsets)  # [E, 8]

    # free path: punish all offsets strongly
    free_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        device=env.device,
    )

    # blocked path:
    # keep first points stable,
    # allow middle deviation,
    # force last points to rejoin
    blocked_weights = torch.tensor(
        [1.0, 0.8, 0.35, 0.15, 0.15, 0.35, 0.8, 1.0],
        device=env.device,
    )

    weights = torch.where(
        blocked[:, None] > 0.5,
        blocked_weights[None, :],
        free_weights[None, :],
    )

    return torch.mean(offsets * weights, dim=1)

def local_path_static_body_clearance_penalty(
    env,
    num_points: int = 8,
    robot_radius: float = 0.22,
):
    if not hasattr(env, "nav2_occupancy_map"):
        return torch.zeros(env.num_envs, device=env.device)

    pts, _ = get_rl_local_path(env, num_points=num_points, step=4)

    angles = torch.linspace(0.0, 2.0 * torch.pi, 12, device=env.device)[:-1]
    circle = torch.stack(
        [torch.cos(angles) * robot_radius, torch.sin(angles) * robot_radius],
        dim=-1,
    )

    check_pts = pts[:, :, None, :] + circle[None, None, :, :]
    flat = check_pts.reshape(-1, 2)

    occupied = env.nav2_occupancy_map.is_occupied_world(
        flat,
        unknown_is_occupied=True,
    ).reshape(env.num_envs, num_points, -1)

    return occupied.float().amax(dim=(1, 2))


def local_path_segment_dynamic_clearance_penalty(
    env,
    num_points: int = 8,
    samples_per_segment: int = 5,
    safe_distance: float = 0.35,
):
    if not hasattr(env, "dyn_obs_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    pts, _ = get_rl_local_path(env, num_points=num_points, step=4)

    p0 = pts[:, :-1, :]
    p1 = pts[:, 1:, :]

    t = torch.linspace(0.0, 1.0, samples_per_segment, device=env.device)
    samples = p0[:, :, None, :] * (1.0 - t[None, None, :, None]) + p1[:, :, None, :] * t[None, None, :, None]

    samples = samples.reshape(env.num_envs, -1, 2)

    diff = env.dyn_obs_xy[:, :, None, :] - samples[:, None, :, :]
    dist = torch.norm(diff, dim=-1)

    clearance = dist - env.dyn_obs_radius[:, :, None]

    clearance = torch.where(
        env.dyn_obs_active[:, :, None],
        clearance,
        torch.ones_like(clearance) * 10.0,
    )

    min_clear = clearance.amin(dim=(1, 2))

    return torch.clamp(safe_distance - min_clear, min=0.0) / safe_distance

def local_path_segment_static_clearance_penalty(
    env,
    num_points: int = 8,
    samples_per_segment: int = 5,
):
    if not hasattr(env, "nav2_occupancy_map"):
        return torch.zeros(env.num_envs, device=env.device)

    pts, _ = get_rl_local_path(env, num_points=num_points, step=4)

    p0 = pts[:, :-1, :]
    p1 = pts[:, 1:, :]

    t = torch.linspace(0.0, 1.0, samples_per_segment, device=env.device)
    samples = p0[:, :, None, :] * (1.0 - t[None, None, :, None]) + p1[:, :, None, :] * t[None, None, :, None]

    samples = samples.reshape(env.num_envs, -1, 2)

    occupied = env.nav2_occupancy_map.is_occupied_world(
        samples.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(env.num_envs, -1)

    return occupied.float().amax(dim=1)



def dense_path_static_clearance_penalty(
    env,
    clearance: float = 0.30,
    samples_per_segment: int = 8,
    circle_samples: int = 16,
):
    if not hasattr(env, "nav2_occupancy_map"):
        return torch.zeros(env.num_envs, device=env.device)

    dense = build_dense_rl_local_path(env, samples_per_segment=samples_per_segment)

    angles = torch.linspace(0.0, 2.0 * torch.pi, circle_samples + 1, device=env.device)[:-1]
    circle = torch.stack(
        [torch.cos(angles) * clearance, torch.sin(angles) * clearance],
        dim=-1,
    )

    check_pts = dense[:, :, None, :] + circle[None, None, :, :]
    occupied = env.nav2_occupancy_map.is_occupied_world(
        check_pts.reshape(-1, 2),
        unknown_is_occupied=True,
    ).reshape(env.num_envs, -1, circle_samples)

    return occupied.float().amax(dim=(1, 2))


def dense_path_dynamic_clearance_penalty(
    env,
    clearance: float = 0.30,
    samples_per_segment: int = 8,
):
    if not hasattr(env, "dyn_obs_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    dense = build_dense_rl_local_path(env, samples_per_segment=samples_per_segment)

    diff = env.dyn_obs_xy[:, :, None, :] - dense[:, None, :, :]
    dist = torch.norm(diff, dim=-1)

    required = env.dyn_obs_radius[:, :, None] + clearance

    violation = required - dist
    violation = torch.where(
        env.dyn_obs_active[:, :, None],
        violation,
        torch.zeros_like(violation),
    )

    return torch.clamp(violation, min=0.0).amax(dim=(1, 2)) / clearance


def build_dense_rl_local_path(env, samples_per_segment: int = 8):
    if not hasattr(env, "rl_local_path_world"):
        build_rl_local_path(env, num_points=8, step=4)

    pts = env.rl_local_path_world  # [E, 8, 2]
    E, N, _ = pts.shape
    device = env.device

    # Catmull-Rom needs padded endpoints
    p = torch.cat(
        [
            pts[:, 0:1, :],
            pts,
            pts[:, -1:, :],
        ],
        dim=1,
    )

    dense = []

    t = torch.linspace(0.0, 1.0, samples_per_segment, device=device)
    t2 = t * t
    t3 = t2 * t

    for i in range(N - 1):
        p0 = p[:, i, :]
        p1 = p[:, i + 1, :]
        p2 = p[:, i + 2, :]
        p3 = p[:, i + 3, :]

        seg = 0.5 * (
            (2.0 * p1[:, None, :])
            + (-p0[:, None, :] + p2[:, None, :]) * t[None, :, None]
            + (2.0 * p0[:, None, :] - 5.0 * p1[:, None, :] + 4.0 * p2[:, None, :] - p3[:, None, :]) * t2[None, :, None]
            + (-p0[:, None, :] + 3.0 * p1[:, None, :] - 3.0 * p2[:, None, :] + p3[:, None, :]) * t3[None, :, None]
        )

        dense.append(seg)

    dense = torch.cat(dense, dim=1)  # [E, dense_points, 2]
    env.rl_local_path_dense_world = dense
    return dense