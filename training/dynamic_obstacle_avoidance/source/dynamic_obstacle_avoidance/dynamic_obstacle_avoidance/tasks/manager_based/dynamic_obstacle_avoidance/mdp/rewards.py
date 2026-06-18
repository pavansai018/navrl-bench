from __future__ import annotations

import torch
import math
from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy, _robot_yaw, _wrap_to_pi, map_collision_flag, dynamic_obstacle_collision_flag



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

def _path_tangent_normal(env, robot_xy: torch.Tensor, lookahead: int = 4):
    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    nearest_idx = _nearest_path_index(env, robot_xy)
    next_idx = torch.minimum(nearest_idx + lookahead, valid_count - 1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    p0 = path[env_ids, nearest_idx]
    p1 = path[env_ids, next_idx]
    tangent = p1 - p0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
    return nearest_idx, p0, tangent, normal

def progress_along_nav2_path(env, asset_cfg: SceneEntityCfg, max_step_progress: float = 0.05,) -> torch.Tensor:
    """Reward forward progress along the Nav2 global path."""
    robot_xy = _robot_xy(env, asset_cfg.name)
    nearest_idx = _nearest_path_index(env, robot_xy)

    env_ids = torch.arange(env.num_envs, device=env.device)
    current_s = env.navrl_path_cum_s[env_ids, nearest_idx]

    delta_s = current_s - env.navrl_prev_progress_s
    env.navrl_prev_progress_s[:] = current_s
    _, _, path_alpha, _ = _goal_blend(env, asset_cfg, start=1.5, end=0.30)
    return path_alpha * torch.clamp(delta_s, -max_step_progress, max_step_progress)

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
    """
    Reward robot yaw alignment with local Nav2 path tangent.
    When path is blocked, allow mecanum/lateral bypass without forcing diff drive style heading behavior
    """
    robot_xy = _robot_xy(env, asset_cfg.name)
    robot_yaw = _robot_yaw(env, asset_cfg.name)

    # path = env.navrl_global_path_xy
    # valid_count = env.navrl_path_valid_count
    # nearest_idx = _nearest_path_index(env, robot_xy)

    # next_idx = torch.minimum(nearest_idx + lookahead_index_offset, valid_count - 1)
    # next_idx = torch.clamp(next_idx, min=0)

    # env_ids = torch.arange(env.num_envs, device=env.device)

    # p0 = path[env_ids, nearest_idx]
    # p1 = path[env_ids, next_idx]
    _, _, tangent, _ = _path_tangent_normal(env, robot_xy, lookahead=lookahead_index_offset)

    # path_yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    err = _wrap_to_pi(path_yaw - robot_yaw)
    heading_reward = torch.cos(err)
    _, _, path_alpha, _ = _goal_blend(env, asset_cfg, start=1.5, end=0.30)
    if hasattr(env, "navrl_path_blocked"):
        path_clear = 1.0 - env.navrl_path_blocked.float()
        return heading_reward * path_clear * path_alpha

    return heading_reward * path_alpha

def goal_approach_reward(env, asset_cfg: SceneEntityCfg, max_step_progress: float = 0.08) -> torch.Tensor:
    """Dense goal-distance improvement. Helps when robot temporarily leaves path to bypass obstacle."""
    robot_xy = _robot_xy(env, asset_cfg.name)
    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    if not hasattr(env, "navrl_prev_goal_dist"):
        env.navrl_prev_goal_dist = dist.clone()
    improvement = env.navrl_prev_goal_dist - dist
    env.navrl_prev_goal_dist[:] = dist
    return torch.clamp(improvement, -max_step_progress, max_step_progress)

def final_goal_reward(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30,) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    return (dist < threshold).float()

def map_collision_penalty(env, asset_cfg: SceneEntityCfg, radius: float = 0.22, ) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,).float()


def dynamic_obstacle_collision_penalty(env, asset_cfg: SceneEntityCfg, robot_radius: float = 0.22) -> torch.Tensor:
    return dynamic_obstacle_collision_flag(env, robot_radius=robot_radius).float()


def dynamic_obstacle_clearance_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    robot_radius: float = 0.22,
    clearance: float = 0.25,
) -> torch.Tensor:
    """Penalty when too close to dynamic obstacles, before actual collision."""
    if not hasattr(env, "dyn_obs_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot_xy = _robot_xy(env, asset_cfg.name)
    d = torch.norm(env.dyn_obs_xy - robot_xy[:, None, :], dim=-1)
    safe_distance = env.dyn_obs_radius + robot_radius + clearance
    violation = torch.clamp(safe_distance - d, min=0.0)
    violation = torch.where(env.dyn_obs_active, violation, torch.zeros_like(violation))
    return torch.max(violation, dim=-1).values


def dynamic_time_to_collision_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    robot_radius: float = 0.22,
    horizon_s: float = 2.0,
) -> torch.Tensor:
    """Predictive dynamic obstacle penalty using relative velocity.

    Penalizes low time-to-collision only when closest approach enters the safety radius.
    This is what pushes early sidestep/slowdown instead of last-second collision avoidance.
    """
    if not hasattr(env, "dyn_obs_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_xy = _robot_xy(env, asset_cfg.name)
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    rel_p = env.dyn_obs_xy - robot_xy[:, None, :]
    rel_v = env.dyn_obs_vel_xy - robot_vel[:, None, :]
    rel_v2 = torch.sum(rel_v * rel_v, dim=-1).clamp_min(1e-6)

    # Time of closest approach. Positive means future approach.
    t_ca = -torch.sum(rel_p * rel_v, dim=-1) / rel_v2
    t_ca = torch.clamp(t_ca, 0.0, horizon_s)
    closest = rel_p + rel_v * t_ca[..., None]
    d_ca = torch.norm(closest, dim=-1)

    safety = env.dyn_obs_radius + robot_radius + 0.12
    risky = env.dyn_obs_active & (d_ca < safety) & (t_ca > 0.0) & (t_ca < horizon_s)
    penalty = torch.where(risky, (horizon_s - t_ca) / horizon_s, torch.zeros_like(t_ca))
    return torch.max(penalty, dim=-1).values

def _processed_base_action(env) -> torch.Tensor:
    action_term = env.action_manager._terms["base_velocity"]
    return action_term.processed_actions

def action_smoothness_penalty(env) -> torch.Tensor:
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    current_action = _processed_base_action(env)

    if not hasattr(env, "navrl_prev_action_for_reward"):
        env.navrl_prev_action_for_reward = current_action.clone()

    diff = torch.norm(current_action - env.navrl_prev_action_for_reward, dim=-1)
    env.navrl_prev_action_for_reward[:] = current_action

    return torch.clamp(diff, 0.0, 2.0)

def yaw_rate_penalty(env) -> torch.Tensor:
    """Small penalty only to avoid spinning as an avoidance exploit."""
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = _processed_base_action(env)
    return torch.clamp(torch.abs(action[:, 2]), 0.0, 2.0)

def path_rejoin_reward(env, asset_cfg: SceneEntityCfg, active_threshold: float = 0.20) -> torch.Tensor:
    """Reward reducing cross-track error only after significant deviation.

    This is not a raw mecanum reward. It pays for returning to the path after a bypass.
    """
    cte = nav2_cross_track_penalty(env, asset_cfg, max_error=1.5)
    if not hasattr(env, "navrl_prev_cte_for_reward"):
        env.navrl_prev_cte_for_reward = cte.clone()
    improvement = env.navrl_prev_cte_for_reward - cte
    env.navrl_prev_cte_for_reward[:] = cte
    active = (cte > active_threshold).float()
    return torch.clamp(improvement, -0.06, 0.06) * active

def mecanum_usage_metric(env) -> torch.Tensor:
    """Metric only. Do not add this as a reward."""
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    a = _processed_base_action(env)
    return torch.abs(a[:, 1]) / (torch.abs(a[:, 0]) + torch.abs(a[:, 1]) + 1e-6)

def time_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def no_wait_penalty(env, asset_cfg: SceneEntityCfg, speed_threshold: float = 0.10) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    goal_dist = torch.norm(env.navrl_final_goal_xy - _robot_xy(env, asset_cfg.name), dim=-1)

    not_goal = (goal_dist > 0.35).float()
    stopped = (speed < speed_threshold).float()

    return stopped * not_goal


def path_velocity_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    robot_xy, goal_dir, path_mode, goal_mode = _goal_blend(env, asset_cfg, start=1.5, end=0.3)


    vel_w = robot.data.root_lin_vel_w[:, :2]

    _, _, tangent, _ = _path_tangent_normal(env, robot_xy, lookahead=4)

    v_path = torch.sum(vel_w * tangent, dim=-1)
    v_goal = torch.sum(vel_w * goal_dir, dim=-1)

    v = path_mode * v_path + goal_mode * v_goal

    # max_v = float(getattr(env.cfg.actions.base_velocity, "max_vx", 0.75))
    # return torch.clamp(v / max_v, 0.0, 1.0)
    max_v = float(getattr(env.cfg.actions.base_velocity, "max_vx", 0.75))
    result = torch.clamp(v / max_v, 0.0, 1.0)
    if hasattr(env, "navrl_path_blocked"):
        return result * (1.0 - env.navrl_path_blocked.float())
    return result


def lateral_oscillation_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    action = env.action_manager._terms["base_velocity"].processed_actions
    vy = action[:, 1]

    if not hasattr(env, "navrl_prev_vy_for_reward"):
        env.navrl_prev_vy_for_reward = torch.zeros(env.num_envs, device=env.device)

    sign_flip = (vy * env.navrl_prev_vy_for_reward < 0.0).float()
    magnitude = torch.abs(vy - env.navrl_prev_vy_for_reward)

    env.navrl_prev_vy_for_reward[:] = vy

    return torch.clamp(sign_flip * magnitude, 0.0, 1.0)

def _goal_blend(env, asset_cfg: SceneEntityCfg, start: float = 1.5, end: float = 0.30):
    robot_xy = _robot_xy(env, asset_cfg.name)
    goal_vec = env.navrl_final_goal_xy - robot_xy
    goal_dist = torch.norm(goal_vec, dim=-1)

    goal_dir = goal_vec / torch.clamp(goal_dist[:, None], min=1e-6)

    alpha = (start - goal_dist) / max(start - end, 1e-6)
    goal_alpha = torch.clamp(alpha, 0.0, 1.0)
    path_alpha = 1.0 - goal_alpha

    return robot_xy, goal_dir, path_alpha, goal_alpha

def static_velocity_clearance_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    safe_distance: float = 0.30,
    max_range: float = 4.0,
    num_rays: int = 144,
    sector_half_angle_rad: float = 0.785398,  # 45 deg
    min_speed: float = 0.05,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]

    robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    vel_w = robot.data.root_lin_vel_w[:, :2]
    speed = torch.norm(vel_w, dim=-1)

    robot_yaw = _robot_yaw(env, asset_cfg.name)

    move_yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])

    # If almost stopped, fall back to robot heading.
    scan_yaw = torch.where(speed > min_speed, move_yaw, robot_yaw)

    scan = env.nav2_occupancy_map.raycast_scan(
        robot_xy=robot_xy,
        robot_yaw=scan_yaw,
        num_rays=num_rays,
        max_range=max_range,
        step_size=0.05,
    )

    scan_m = scan * max_range

    ray_angles = torch.linspace(
        -math.pi,
        math.pi,
        num_rays + 1,
        device=env.device,
    )[:-1]

    sector_mask = torch.abs(ray_angles) <= sector_half_angle_rad

    move_sector_scan = scan_m[:, sector_mask]
    move_dir_clearance = move_sector_scan.min(dim=-1).values

    penalty = (safe_distance - move_dir_clearance) / safe_distance
    result = torch.clamp(penalty, 0.0, 1.0)
    if hasattr(env, "navrl_path_blocked"):
        return result * (1.0 - env.navrl_path_blocked.float())
    return result

def start_speed_penalty(env, asset_cfg: SceneEntityCfg, warmup_s: float = 2.0):
    robot = env.scene[asset_cfg.name]
    step_dt = float(getattr(env, "step_dt", 1.0 / 30.0))

    if not hasattr(env, "episode_step_count"):
        return torch.zeros(env.num_envs, device=env.device)

    t = env.episode_step_count.float() * step_dt
    active = t < warmup_s

    vx = robot.data.root_lin_vel_b[:, 0]
    vy = robot.data.root_lin_vel_b[:, 1]
    speed = torch.sqrt(vx * vx + vy * vy)

    return torch.where(active, speed, torch.zeros_like(speed))

def lateral_bypass_reward(env, asset_cfg, max_cte: float = 0.8) -> torch.Tensor:
    if not hasattr(env, "navrl_path_blocked"):
        return torch.zeros(env.num_envs, device=env.device)
    # Only reward lateral motion when still close to the path.
    # Prevents the robot from staying permanently parallel while path_blocked stays
    # True from a different obstacle further ahead.
    cte = nav2_cross_track_penalty(env, asset_cfg, max_error=max_cte + 0.1)
    near_path = (cte < max_cte).float()
    action = env.action_manager._terms["base_velocity"].processed_actions
    vy_abs = torch.abs(action[:, 1])
    max_vy = float(getattr(env.cfg.actions.base_velocity, "max_vy", 0.5))
    return env.navrl_path_blocked.float() * near_path * torch.clamp(vy_abs / max_vy, 0.0, 1.0)