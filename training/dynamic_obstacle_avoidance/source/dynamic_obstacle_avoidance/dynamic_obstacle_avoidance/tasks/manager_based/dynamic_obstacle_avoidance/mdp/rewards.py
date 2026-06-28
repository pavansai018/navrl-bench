from __future__ import annotations

import torch
import math
from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy, _robot_yaw, _wrap_to_pi, map_collision_flag, dynamic_obstacle_collision_flag
from .teacher_mppi import get_mppi_teacher_action


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
    # if hasattr(env, "navrl_path_blocked"):
    #     path_clear = 1.0 - env.navrl_path_blocked.float()
    #     return heading_reward * path_clear * path_alpha

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
    return (dist < threshold).float() * _event_scale(env)

def map_collision_penalty(env, asset_cfg: SceneEntityCfg, radius: float = 0.22, ) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,).float() * _event_scale(env)


def dynamic_obstacle_collision_penalty(env, asset_cfg: SceneEntityCfg, robot_radius: float = 0.22) -> torch.Tensor:
    return dynamic_obstacle_collision_flag(env, robot_radius=robot_radius).float() * _event_scale(env)


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
    # rel_v = env.dyn_obs_vel_xy - robot_vel[:, None, :]
    obs_vel = getattr(env, "dyn_obs_eff_vel_xy", env.dyn_obs_vel_xy)
    rel_v = obs_vel - robot_vel[:, None, :]
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

def timeout_penalty(env) -> torch.Tensor:
    return (env.episode_length_buf >= env.max_episode_length - 1).float()

# def no_wait_penalty(env, asset_cfg: SceneEntityCfg, speed_threshold: float = 0.10) -> torch.Tensor:
#     robot = env.scene[asset_cfg.name]
#     speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
#     goal_dist = torch.norm(env.navrl_final_goal_xy - _robot_xy(env, asset_cfg.name), dim=-1)

#     not_goal = (goal_dist > 0.35).float()
#     stopped = (speed < speed_threshold).float()

#     return stopped * not_goal

def no_wait_penalty(env, asset_cfg: SceneEntityCfg, speed_threshold: float = 0.10) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    goal_dist = torch.norm(env.navrl_final_goal_xy - _robot_xy(env, asset_cfg.name), dim=-1)

    not_goal = (goal_dist > 0.35).float()
    stopped = (speed < speed_threshold).float()

    result = stopped * not_goal

    # Do not punish yielding when a dynamic obstacle blocks the near-future path.
    # Otherwise PPO is pushed to keep moving into the obstacle.
    valid_dyn_block, _, _, _, _ = _dynamic_corridor_obstacle_info(env, asset_cfg)

    if hasattr(env, "navrl_path_blocked"):
        blocked = torch.maximum(env.navrl_path_blocked.float(), valid_dyn_block.float())
    else:
        blocked = valid_dyn_block.float()

    return result * (1.0 - blocked)


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
    # if hasattr(env, "navrl_path_blocked"):
    #     return result * (1.0 - env.navrl_path_blocked.float())
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
    # if hasattr(env, "navrl_path_blocked"):
    #     return result * (1.0 - env.navrl_path_blocked.float())
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

def mppi_teacher_imitation_reward(env) -> torch.Tensor:
    """Imitation reward for the torch-MPPI teacher.

    Returns 0 for perfect match and negative values for mismatch.
    This should be used only in early teacher-guided PPO training.
    """
    if not hasattr(env, "action_manager") or "base_velocity" not in env.action_manager._terms:
        return torch.zeros(env.num_envs, device=env.device)

    action_term = env.action_manager._terms["base_velocity"]
    policy_action = action_term.processed_actions
    teacher_action = get_mppi_teacher_action(env)

    scale = torch.tensor(
        [
            float(getattr(action_term.cfg, "max_vx", 0.5)),
            float(getattr(action_term.cfg, "max_vy", 0.5)),
            float(getattr(action_term.cfg, "max_wz", 1.5)),
        ],
        device=env.device,
        dtype=policy_action.dtype,
    )

    err = (policy_action - teacher_action) / scale.clamp_min(1.0e-6)
    err2 = torch.sum(err * err, dim=-1)

    return -torch.clamp(err2, 0.0, 4.0)

def _event_scale(env) -> float:
    return 1.0 / max(float(getattr(env, "step_dt", 1.0 / 30.0)), 1.0e-6)


# -----------------------------------------------------------------------------
# Dynamic obstacle mode rewards
# Actor remains scan/path/velocity only.
# These use simulator dynamic obstacle buffers only for reward credit assignment.
# -----------------------------------------------------------------------------

def _dynamic_corridor_obstacle_info(
    env,
    asset_cfg: SceneEntityCfg,
    lookahead_m: float = 2.0,
    corridor_half_width: float = 0.55,
):
    """
    Nearest active dynamic obstacle in the near-future path corridor.

    This is reward-only. It is not added to actor observation.
    """
    robot_xy = _robot_xy(env, asset_cfg.name)
    _, _, tangent, normal = _path_tangent_normal(env, robot_xy, lookahead=4)

    if not hasattr(env, "dyn_obs_xy") or not hasattr(env, "dyn_obs_active"):
        valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        large = torch.ones(env.num_envs, device=env.device) * 1.0e6
        lat = torch.zeros(env.num_envs, device=env.device)
        return valid, large, lat, tangent, normal

    rel = env.dyn_obs_xy - robot_xy[:, None, :]  # [env, obstacle, 2]

    along = torch.sum(rel * tangent[:, None, :], dim=-1)
    lateral = torch.sum(rel * normal[:, None, :], dim=-1)
    dist = torch.norm(rel, dim=-1)

    in_corridor = (
        env.dyn_obs_active
        & (along > -0.25)
        & (along < lookahead_m)
        & (torch.abs(lateral) < corridor_half_width)
    )

    masked_dist = torch.where(in_corridor, dist, torch.ones_like(dist) * 1.0e6)

    nearest_idx = torch.argmin(masked_dist, dim=-1)
    env_ids = torch.arange(env.num_envs, device=env.device)

    nearest_dist = masked_dist[env_ids, nearest_idx]
    nearest_lat = lateral[env_ids, nearest_idx]

    valid = nearest_dist < 1.0e5

    return valid, nearest_dist, nearest_lat, tangent, normal


def _static_clearance_in_world_direction(
    env,
    robot_xy: torch.Tensor,
    world_yaw: torch.Tensor,
    max_range: float = 1.2,
    num_rays: int = 72,
    sector_half_angle_rad: float = 0.35,
) -> torch.Tensor:
    """
    Static-map clearance centered on a world-frame direction.
    Used only for reward gating.
    """
    scan = env.nav2_occupancy_map.raycast_scan(
        robot_xy=robot_xy,
        robot_yaw=world_yaw,
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
        dtype=robot_xy.dtype,
    )[:-1]

    sector_mask = torch.abs(ray_angles) <= sector_half_angle_rad

    return scan_m[:, sector_mask].min(dim=-1).values


def _path_side_static_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    max_range: float = 1.2,
    num_rays: int = 72,
    sector_half_angle_rad: float = 0.35,
):
    """
    Static clearance on path-left and path-right.
    """
    robot_xy = _robot_xy(env, asset_cfg.name)
    _, _, _, normal = _path_tangent_normal(env, robot_xy, lookahead=4)

    left_yaw = torch.atan2(normal[:, 1], normal[:, 0])
    right_yaw = torch.atan2(-normal[:, 1], -normal[:, 0])

    left_clear = _static_clearance_in_world_direction(
        env,
        robot_xy,
        left_yaw,
        max_range=max_range,
        num_rays=num_rays,
        sector_half_angle_rad=sector_half_angle_rad,
    )

    right_clear = _static_clearance_in_world_direction(
        env,
        robot_xy,
        right_yaw,
        max_range=max_range,
        num_rays=num_rays,
        sector_half_angle_rad=sector_half_angle_rad,
    )

    return left_clear, right_clear

def _world_vel_to_body_action(
    env,
    asset_cfg: SceneEntityCfg,
    vel_w: torch.Tensor,
) -> torch.Tensor:
    """Convert world-frame xy velocity into robot body-frame [vx, vy]."""
    yaw = _robot_yaw(env, asset_cfg.name)

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    vx_b = c * vel_w[:, 0] + s * vel_w[:, 1]
    vy_b = -s * vel_w[:, 0] + c * vel_w[:, 1]

    return torch.stack([vx_b, vy_b], dim=-1)


def _future_dynamic_corridor_blocked(
    env,
    asset_cfg: SceneEntityCfg,
    horizon_s: float = 1.0,
    lookahead_m: float = 2.5,
    corridor_half_width: float = 0.45,
    robot_radius: float = 0.22,
) -> torch.Tensor:
    """
    Reward-only future blockage classifier.

    True:
        a dynamic obstacle will still block the near path corridor after horizon_s.

    This is NOT actor observation.
    """

    if not hasattr(env, "dyn_obs_xy") or not hasattr(env, "dyn_obs_active"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot_xy = _robot_xy(env, asset_cfg.name)
    _, _, tangent, normal = _path_tangent_normal(env, robot_xy, lookahead=4)

    t = float(horizon_s)

    obs_xy_future = env.dyn_obs_xy + env.dyn_obs_vel_xy * t

    # Include nonlinear obstacle wobble.
    if hasattr(env, "dyn_obs_phase"):
        nonlinear_mask = env.dyn_obs_active & (env.dyn_obs_scenario == 4)

        if torch.any(nonlinear_mask):
            phase_now = env.dyn_obs_phase
            phase_future = env.dyn_obs_phase + env.dyn_obs_omega * t

            delta_wobble = (
                torch.sin(phase_future) - torch.sin(phase_now)
            ) * env.dyn_obs_amp

            obs_xy_future = obs_xy_future + (
                env.dyn_obs_normal
                * delta_wobble.unsqueeze(-1)
                * nonlinear_mask.unsqueeze(-1).float()
            )

    rel = obs_xy_future - robot_xy[:, None, :]

    along = torch.sum(rel * tangent[:, None, :], dim=-1)
    lateral = torch.sum(rel * normal[:, None, :], dim=-1)

    effective_half_width = (
        corridor_half_width
        + env.dyn_obs_radius
        + robot_radius
    )

    blocked = (
        env.dyn_obs_active
        & (along > -0.25)
        & (along < lookahead_m)
        & (torch.abs(lateral) < effective_half_width)
    )

    return blocked.any(dim=-1)


def future_aware_action_teacher_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    horizon_s: float = 1.0,
    lookahead_m: float = 2.5,
    corridor_half_width: float = 0.45,
    robot_radius: float = 0.22,

    normal_track_speed: float = 0.45,
    cautious_track_speed: float = 0.20,
    detour_forward_speed: float = 0.12,
    detour_lateral_speed: float = 0.30,

    k_track_cte: float = 0.80,
    k_detour_cte: float = 0.25,
    max_track_lateral_speed: float = 0.20,
    max_detour_correction_speed: float = 0.12,

    yaw_gain: float = 1.5,
) -> torch.Tensor:
    """
    Direct action-shaping penalty for the behavior you want.

    Modes:

    1. Path clear:
        track global path, correct CTE, no wide lateral bypass.

    2. Blocked now but clear after horizon_s:
        keep tracking path cautiously.
        Do NOT lateral bypass.

    3. Blocked now and still blocked after horizon_s:
        controlled detour is allowed.
        Move away from obstacle side, but still keep mild rejoin pressure.

    Return:
        positive action error.
        Use negative reward weight.
    """

    if not hasattr(env, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    action = _processed_base_action(env)

    robot_xy = _robot_xy(env, asset_cfg.name)
    robot_yaw = _robot_yaw(env, asset_cfg.name)

    nearest_idx, path_xy, tangent, normal = _path_tangent_normal(
        env,
        robot_xy,
        lookahead=4,
    )

    # Signed cross-track error.
    # +ve means robot is left of path normal direction.
    signed_cte = torch.sum((robot_xy - path_xy) * normal, dim=-1)

    # Current dynamic obstacle blocking path corridor.
    now_blocked, _, obstacle_lat, _, _ = _dynamic_corridor_obstacle_info(
        env,
        asset_cfg,
        lookahead_m=lookahead_m,
        corridor_half_width=corridor_half_width,
    )

    # Future dynamic obstacle blocking path corridor.
    future_blocked = _future_dynamic_corridor_blocked(
        env,
        asset_cfg,
        horizon_s=horizon_s,
        lookahead_m=lookahead_m,
        corridor_half_width=corridor_half_width,
        robot_radius=robot_radius,
    )

    persistent_block = now_blocked & future_blocked

    # If currently blocked or soon blocked, track cautiously.
    cautious_track = now_blocked | future_blocked

    # --------------------------------------------------
    # TRACK / CLEAR-SOON target
    # --------------------------------------------------
    track_speed = torch.where(
        cautious_track,
        torch.ones(env.num_envs, device=env.device) * cautious_track_speed,
        torch.ones(env.num_envs, device=env.device) * normal_track_speed,
    )

    track_lateral = torch.clamp(
        -k_track_cte * signed_cte,
        -max_track_lateral_speed,
        max_track_lateral_speed,
    )

    track_vel_w = (
        track_speed[:, None] * tangent
        + track_lateral[:, None] * normal
    )

    # --------------------------------------------------
    # PERSISTENT-BLOCK target
    # --------------------------------------------------
    # If obstacle is left of path, detour right.
    # If obstacle is right of path, detour left.
    # If obstacle is centered, choose side that moves robot back toward path center.
    obs_side_known = torch.abs(obstacle_lat) > 0.05

    side_from_obstacle = -torch.sign(obstacle_lat)

    side_from_cte = -torch.sign(signed_cte)
    side_from_cte = torch.where(
        torch.abs(side_from_cte) < 0.5,
        torch.ones_like(side_from_cte),
        side_from_cte,
    )

    detour_side = torch.where(
        obs_side_known,
        side_from_obstacle,
        side_from_cte,
    )

    detour_lateral = detour_side * detour_lateral_speed

    # Mild rejoin pressure even during detour.
    detour_lateral = detour_lateral + torch.clamp(
        -k_detour_cte * signed_cte,
        -max_detour_correction_speed,
        max_detour_correction_speed,
    )

    detour_vel_w = (
        detour_forward_speed * tangent
        + detour_lateral[:, None] * normal
    )

    # --------------------------------------------------
    # Select mode target
    # --------------------------------------------------
    target_vel_w = torch.where(
        persistent_block[:, None],
        detour_vel_w,
        track_vel_w,
    )

    target_body_xy = _world_vel_to_body_action(
        env,
        asset_cfg,
        target_vel_w,
    )

    # Yaw target: keep robot heading aligned with path tangent.
    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    yaw_err = _wrap_to_pi(path_yaw - robot_yaw)

    max_wz = float(getattr(env.cfg.actions.base_velocity, "max_wz", 2.0))
    target_wz = torch.clamp(
        yaw_gain * yaw_err,
        -max_wz,
        max_wz,
    )

    target_action = torch.cat(
        [
            target_body_xy,
            target_wz.unsqueeze(-1),
        ],
        dim=-1,
    )

    max_vx = float(getattr(env.cfg.actions.base_velocity, "max_vx", 0.75))
    max_vy = float(getattr(env.cfg.actions.base_velocity, "max_vy", 0.75))

    scale = torch.tensor(
        [max_vx, max_vy, max_wz],
        device=env.device,
        dtype=action.dtype,
    ).clamp_min(1.0e-6)

    err = (action - target_action) / scale

    # Slightly higher penalty on vy because your failure is unnecessary lateral escape.
    weights = torch.tensor(
        [1.0, 1.5, 0.5],
        device=env.device,
        dtype=action.dtype,
    )

    return torch.sum(weights * err * err, dim=-1).clamp(0.0, 4.0)