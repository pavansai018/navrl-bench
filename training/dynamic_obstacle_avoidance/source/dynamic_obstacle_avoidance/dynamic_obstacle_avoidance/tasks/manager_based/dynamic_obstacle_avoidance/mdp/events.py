from __future__ import annotations

import math
import os
import torch
from .path_dataset import Nav2PathDataset
from .nav2_map import Nav2OccupancyMap
from isaaclab.managers import SceneEntityCfg
from .observations import map_collision_direction_flags, combined_static_dynamic_scan, dynamic_path_blockage
from . import config

def _ensure_nav2_path_buffers(env, max_path_points: int = 600):
    device = env.device

    if not hasattr(env, "nav2_path_dataset"):
        env.nav2_path_dataset = Nav2PathDataset(
            dataset_dir=env.cfg.nav2_path_dataset_dir,
            device=device,
            max_path_points=max_path_points,
        )

    if not hasattr(env, "navrl_global_path_xy"):
        env.navrl_global_path_xy = torch.zeros(
            env.num_envs, max_path_points, 2, device=device
        )

    if not hasattr(env, "navrl_path_valid_count"):
        env.navrl_path_valid_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )

    if not hasattr(env, "navrl_final_goal_xy"):
        env.navrl_final_goal_xy = torch.zeros(env.num_envs, 2, device=device)

    if not hasattr(env, "navrl_start_pose"):
        env.navrl_start_pose = torch.zeros(env.num_envs, 3, device=device)

    if not hasattr(env, "navrl_goal_pose"):
        env.navrl_goal_pose = torch.zeros(env.num_envs, 3, device=device)


def reset_nav2_path_dataset(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
    final_goal_marker_cfg=None,
    max_path_points: int = 600,
):
    """
    Load one real Nav2 global path per reset and place robot at path start.
    Noise is intentionally small. Dynamic obstacles, not reset offset, are the
    mechanism that should create meaningful holonomic bypass behavior.
    """

    _ensure_nav2_path_buffers(env, max_path_points=max_path_points)

    robot = env.scene[asset_cfg.name]

    starts, goals, paths, valid_counts, path_lengths = env.nav2_path_dataset.sample_batch(env_ids)

    # Force final goal/marker to match actual path end.
    env_ids_local = torch.arange(len(env_ids), device=env.device)
    last_idx = torch.clamp(valid_counts - 1, min=0)
    last_xy = paths[env_ids_local, last_idx]

    goals = goals.clone()
    goals[:, 0:2] = last_xy
    env.navrl_start_pose[env_ids] = starts
    env.navrl_goal_pose[env_ids] = goals
    env.navrl_global_path_xy[env_ids] = paths
    env.navrl_path_valid_count[env_ids] = valid_counts
    env.navrl_final_goal_xy[env_ids] = goals[:, :2]

    root_state = robot.data.default_root_state[env_ids].clone()

    # --------------------------------------------------
    # Original Nav2 path start pose
    # --------------------------------------------------
    path_x = starts[:, 0]
    path_y = starts[:, 1]
    path_yaw = starts[:, 2]

    lateral_noise = float(getattr(env.cfg, "reset_lateral_noise_m", 0.10))
    yaw_noise_mag = float(getattr(env.cfg, "reset_yaw_noise_rad", 0.10))
    # --------------------------------------------------
    # Mecanum-relevant reset randomization
    # Creates lateral path error from episode start.
    # This gives the robot a reason to use vy immediately.
    # --------------------------------------------------
    lateral_offset = torch.empty(len(env_ids), device=env.device).uniform_(-lateral_noise, lateral_noise)
    yaw_offset = torch.empty(len(env_ids), device=env.device).uniform_(-yaw_noise_mag, yaw_noise_mag)

    # Path-normal direction in map frame.
    # If path_yaw points forward, this is left/right of the path.
    normal_x = -torch.sin(path_yaw)
    normal_y = torch.cos(path_yaw)

    root_state[:, 0] = path_x + lateral_offset * normal_x + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = path_y + lateral_offset * normal_y + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = 0.05

    yaw = path_yaw + yaw_offset

    root_state[:, 3] = torch.cos(yaw * 0.5)   # qw
    root_state[:, 4] = 0.0                    # qx
    root_state[:, 5] = 0.0                    # qy
    root_state[:, 6] = torch.sin(yaw * 0.5)   # qz

    root_state[:, 7:13] = 0.0

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Optional: move visual final-goal marker.
    if final_goal_marker_cfg is not None:
        marker = env.scene[final_goal_marker_cfg.name]

        pose = torch.zeros(len(env_ids), 7, device=env.device)
        pose[:, 0] = goals[:, 0] + env.scene.env_origins[env_ids, 0]
        pose[:, 1] = goals[:, 1] + env.scene.env_origins[env_ids, 1]
        pose[:, 2] = 0.15
        pose[:, 3] = 1.0

        marker.write_root_pose_to_sim(pose, env_ids=env_ids)

def _ensure_nav2_map(env):
    if not hasattr(env, "nav2_occupancy_map"):
        env.nav2_occupancy_map = Nav2OccupancyMap(
            map_yaml_path=env.cfg.nav2_map_yaml_path,
            device=env.device,
        )





def reset_nav2_path_and_debug_validate(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    final_goal_marker_cfg: SceneEntityCfg | None = None,
    max_path_points: int = 600,
):
    """Load Nav2 path dataset, place robot, place goal marker, validate map/path alignment."""
    reset_nav2_path_dataset(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        final_goal_marker_cfg=final_goal_marker_cfg,
        max_path_points=max_path_points,
    )
    _ensure_training_reward_buffers(env, max_path_points=max_path_points)

    path = env.navrl_global_path_xy[env_ids]
    diff = path[:, 1:, :] - path[:, :-1, :]
    seg_len = torch.norm(diff, dim=-1)

    env.navrl_path_cum_s[env_ids, :] = 0.0
    env.navrl_path_cum_s[env_ids, 1:] = torch.cumsum(seg_len, dim=-1)

    env.navrl_prev_progress_s[env_ids] = 0.0
    env.navrl_prev_action_for_reward[env_ids] = 0.0

    # Initialize stateful reward buffers to current values to avoid false first-step rewards.
    robot = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    nearest_idx = torch.argmin(torch.norm(env.navrl_global_path_xy - robot_xy[:, None, :], dim=-1), dim=-1)
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    cte_now = torch.norm(env.navrl_global_path_xy[all_env_ids, nearest_idx] - robot_xy, dim=-1)
    goal_dist_now = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    if not hasattr(env, "navrl_prev_cte_for_reward"):
        env.navrl_prev_cte_for_reward = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "navrl_prev_goal_dist"):
        env.navrl_prev_goal_dist = torch.zeros(env.num_envs, device=env.device)

    # if not hasattr(env, "navrl_prev_abs_lat_error"):
    #     env.navrl_prev_abs_lat_error = torch.zeros(env.num_envs, device=env.device)

    # env.navrl_prev_abs_lat_error[env_ids] = 0.0
    if not hasattr(env, "navrl_prev_vy_for_reward"):
        env.navrl_prev_vy_for_reward = torch.zeros(env.num_envs, device=env.device)

    env.navrl_prev_vy_for_reward[env_ids] = 0.0

    env.navrl_prev_cte_for_reward[env_ids] = cte_now[env_ids]
    env.navrl_prev_goal_dist[env_ids] = goal_dist_now[env_ids]

    # validate_nav2_path_against_map(
    #     env=env,
    #     env_ids=env_ids,
    #     max_bad_fraction=0.02,
    # )



def _get_debug_draw():
    """Acquire Isaac Sim DebugDraw interface.

    This is visual-only. It does not create PhysX bodies.
    """
    try:
        from isaacsim.util.debug_draw import _debug_draw # type: ignore
    except Exception:
        from omni.isaac.debug_draw import _debug_draw  # type: ignore

    return _debug_draw.acquire_debug_draw_interface()


def _clear_debug_draw(draw):
    for fn in ("clear", "clear_points", "clear_lines"):
        if hasattr(draw, fn):
            try:
                getattr(draw, fn)()
            except Exception:
                pass


def validate_nav2_path_against_map(
    env,
    env_ids: torch.Tensor,
    max_bad_fraction: float = 0.02,
):
    """Print whether loaded Nav2 paths are free in the loaded Nav2 occupancy map."""
    _ensure_nav2_map(env)

    for env_id in env_ids:
        valid_count = int(env.navrl_path_valid_count[env_id].item())

        if valid_count <= 1:
            print(
                f"[NAV2 MAP CHECK] env={int(env_id)} invalid path count={valid_count}",
                flush=True,
            )
            continue

        path_xy = env.navrl_global_path_xy[env_id, :valid_count]

        report = env.nav2_occupancy_map.path_occupancy_report(
            path_xy,
            unknown_is_occupied=True,
        )

        msg = (
            f"[NAV2 MAP CHECK] env={int(env_id)} "
            f"points={report['num_points']} "
            f"occupied={report['num_occupied']} "
            f"fraction={report['occupied_fraction']:.4f}"
        )

        if report["occupied_fraction"] > max_bad_fraction:
            print("[NAV2 MAP CHECK][ERROR] " + msg, flush=True)
            print(
                f"[NAV2 MAP CHECK][ERROR] first bad points: {report['first_bad_points']}",
                flush=True,
            )
        else:
            print("[NAV2 MAP CHECK][OK] " + msg, flush=True)


def _ensure_dynamic_obstacle_buffers(env):
    """Tensor-only dynamic obstacles."""
    num_obs = int(getattr(env.cfg, "max_dynamic_obstacles", 6))
    device = env.device

    def needs_resize(name, shape):
        return (not hasattr(env, name)) or (getattr(env, name).shape != shape)

    if needs_resize("dyn_obs_xy", (env.num_envs, num_obs, 2)):
        env.dyn_obs_xy = torch.zeros(env.num_envs, num_obs, 2, device=device)

    if needs_resize("dyn_obs_vel_xy", (env.num_envs, num_obs, 2)):
        env.dyn_obs_vel_xy = torch.zeros(env.num_envs, num_obs, 2, device=device)

    if needs_resize("dyn_obs_radius", (env.num_envs, num_obs)):
        env.dyn_obs_radius = torch.zeros(env.num_envs, num_obs, device=device)

    if needs_resize("dyn_obs_active", (env.num_envs, num_obs)):
        env.dyn_obs_active = torch.zeros(
            env.num_envs, num_obs, dtype=torch.bool, device=device
        )

    if needs_resize("dyn_obs_scenario", (env.num_envs, num_obs)):
        env.dyn_obs_scenario = torch.zeros(
            env.num_envs, num_obs, dtype=torch.long, device=device
        )
    
    if needs_resize("dyn_obs_phase", (env.num_envs, num_obs)):
        env.dyn_obs_phase = torch.zeros(env.num_envs, num_obs, device=device)

    if needs_resize("dyn_obs_omega", (env.num_envs, num_obs)):
        env.dyn_obs_omega = torch.zeros(env.num_envs, num_obs, device=device)

    if needs_resize("dyn_obs_amp", (env.num_envs, num_obs)):
        env.dyn_obs_amp = torch.zeros(env.num_envs, num_obs, device=device)

    if needs_resize("dyn_obs_normal", (env.num_envs, num_obs, 2)):
        env.dyn_obs_normal = torch.zeros(env.num_envs, num_obs, 2, device=device)

    if not hasattr(env, "navrl_reset_count"):
        env.navrl_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    if not hasattr(env, "navrl_curriculum_level"):
        env.navrl_curriculum_level = torch.zeros(env.num_envs, dtype=torch.long, device=device)


def reset_dynamic_obstacles_tensor(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, max_path_points: int = 600):
    """Create tensor-based dynamic obstacle scenarios near the future path.

    Curriculum is reset-count based:
      0: no obstacles
      1: stationary path blocker
      2: one slow crossing obstacle
      3: one faster crossing obstacle
      4: crossing + slow same-lane obstacle
      5+: mixed 2-3 obstacles
    """
    _ensure_dynamic_obstacle_buffers(env)
    if hasattr(env, "termination_manager"):
        update_performance_curriculum_on_reset(env, env_ids)
    robot = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    fixed_level = int(getattr(env.cfg, "fixed_curriculum_level", -1))

    if fixed_level >= 0:
        env.navrl_curriculum_level[env_ids] = fixed_level
    else:
        _ensure_performance_curriculum_buffers(env)
        # env.navrl_curriculum_level[env_ids] = int(env.navrl_curriculum_level_global)
        global_level = int(env.navrl_curriculum_level_global)

        if bool(getattr(env.cfg, "sample_curriculum_levels", False)):
            min_level = int(getattr(env.cfg, "curriculum_sample_min_level", 0))
            max_level_cfg = int(getattr(env.cfg, "curriculum_sample_max_level", global_level))

            # Use the smaller of configured max and adaptive global level,
            # unless you are manually forcing a staged max.
            max_level = max(min_level, max_level_cfg)

            env.navrl_curriculum_level[env_ids] = torch.randint(
                low=min_level,
                high=max_level + 1,
                size=(len(env_ids),),
                device=env.device,
            )
        else:
            env.navrl_curriculum_level[env_ids] = global_level
    env.navrl_reset_count[env_ids] += 1
    reset_domain_randomization(env, env_ids, asset_cfg=asset_cfg)

    env.dyn_obs_active[env_ids] = False
    env.dyn_obs_xy[env_ids] = 0.0
    env.dyn_obs_vel_xy[env_ids] = 0.0
    env.dyn_obs_radius[env_ids] = 0.0
    env.dyn_obs_scenario[env_ids] = 0
    env.dyn_obs_phase[env_ids] = 0.0
    env.dyn_obs_omega[env_ids] = 0.0
    env.dyn_obs_amp[env_ids] = 0.0
    env.dyn_obs_normal[env_ids] = 0.0

    path = env.navrl_global_path_xy
    valid_count = env.navrl_path_valid_count
    num_obs = env.dyn_obs_xy.shape[1]

    for local_i, env_id_t in enumerate(env_ids):
        env_id = int(env_id_t.item())
        # obstacle_level = int(env.navrl_curriculum_level[env_id].item())
        raw_level = int(env.navrl_curriculum_level[env_id].item())
        obstacle_level, _ = get_curriculum_sublevels(env, raw_level)
        vc = int(valid_count[env_id].item())
        if vc < 12 or obstacle_level <= 0:
            continue

        modes = config.obstacle_stages[:obstacle_level]
        modes = modes[:num_obs]

        start_idx = max(5, int(0.10 * vc))
        # end_idx = max(start_idx + 1, min(vc - 2, int(0.65 * vc)))
        end_idx = max(start_idx + 1, min(vc - 2, int(0.35 * vc)))

        for j, mode in enumerate(modes):
            idx = int(torch.randint(start_idx, end_idx, (1,), device=env.device).item())

            p0 = path[env_id, idx]
            p1 = path[env_id, min(idx + 4, vc - 1)]

            tangent = p1 - p0
            tangent = tangent / torch.norm(tangent).clamp_min(1e-6)

            normal = torch.stack([-tangent[1], tangent[0]])
            side = 1.0 if torch.rand((), device=env.device).item() > 0.5 else -1.0

            if mode == "side_stationary_tiny":
                radius = torch.empty((), device=env.device).uniform_(0.08, 0.11)
                offset = side * torch.empty((), device=env.device).uniform_(0.45, 0.60)
                pos = p0 + offset * normal
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "side_stationary_small":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.14)
                offset = side * torch.empty((), device=env.device).uniform_(0.35, 0.50)
                pos = p0 + offset * normal
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "center_stationary_tiny":
                radius = torch.empty((), device=env.device).uniform_(0.08, 0.11)
                # pos = p0
                offset = side * torch.empty((), device=env.device).uniform_(0.05, 0.20)
                pos = p0 + offset * normal
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "center_stationary_small":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.14)
                # pos = p0
                offset = side * torch.empty((), device=env.device).uniform_(0.05, 0.20)
                pos = p0 + offset * normal
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "center_stationary_medium":
                radius = torch.empty((), device=env.device).uniform_(0.14, 0.18)
                # pos = p0
                offset = side * torch.empty((), device=env.device).uniform_(0.05, 0.20)
                pos = p0 + offset * normal
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "slow_crossing_far":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.15)
                offset = side * torch.empty((), device=env.device).uniform_(1.10, 1.50)
                speed = torch.empty((), device=env.device).uniform_(0.04, 0.10)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            elif mode == "slow_crossing_near":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.16)
                offset = side * torch.empty((), device=env.device).uniform_(0.75, 1.10)
                speed = torch.empty((), device=env.device).uniform_(0.06, 0.14)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            elif mode == "medium_crossing_far":
                radius = torch.empty((), device=env.device).uniform_(0.12, 0.18)
                offset = side * torch.empty((), device=env.device).uniform_(1.10, 1.50)
                speed = torch.empty((), device=env.device).uniform_(0.12, 0.22)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            elif mode == "medium_crossing_near":
                radius = torch.empty((), device=env.device).uniform_(0.12, 0.20)
                offset = side * torch.empty((), device=env.device).uniform_(0.75, 1.10)
                speed = torch.empty((), device=env.device).uniform_(0.16, 0.28)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            elif mode == "fast_crossing_far":
                radius = torch.empty((), device=env.device).uniform_(0.12, 0.22)
                offset = side * torch.empty((), device=env.device).uniform_(1.10, 1.60)
                speed = torch.empty((), device=env.device).uniform_(0.25, 0.40)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            elif mode == "same_lane_slow":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.16)
                ahead = torch.empty((), device=env.device).uniform_(0.30, 0.70)
                speed = torch.empty((), device=env.device).uniform_(0.04, 0.10)
                pos = p0 + ahead * tangent
                vel = speed * tangent
                scenario = 3

            elif mode == "same_lane_medium":
                radius = torch.empty((), device=env.device).uniform_(0.12, 0.20)
                ahead = torch.empty((), device=env.device).uniform_(0.30, 0.80)
                speed = torch.empty((), device=env.device).uniform_(0.10, 0.20)
                pos = p0 + ahead * tangent
                vel = speed * tangent
                scenario = 3

            elif mode == "reverse_same_lane_slow":
                radius = torch.empty((), device=env.device).uniform_(0.10, 0.18)
                ahead = torch.empty((), device=env.device).uniform_(0.50, 1.00)
                speed = torch.empty((), device=env.device).uniform_(0.04, 0.12)
                pos = p0 + ahead * tangent
                vel = -speed * tangent
                scenario = 3

            elif mode == "center_stationary_large":
                radius = torch.empty((), device=env.device).uniform_(0.18, 0.24)
                pos = p0
                vel = torch.zeros(2, device=env.device)
                scenario = 1

            elif mode == "real_room_dynamic_clutter":
                continue

            else:  # two_crossing_combo
                radius = torch.empty((), device=env.device).uniform_(0.12, 0.22)
                offset = side * torch.empty((), device=env.device).uniform_(0.80, 1.30)
                speed = torch.empty((), device=env.device).uniform_(0.12, 0.30)
                pos = p0 + offset * normal
                vel = -side * speed * normal
                scenario = 2

            env.dyn_obs_xy[env_id, j] = pos
            env.dyn_obs_vel_xy[env_id, j] = vel
            env.dyn_obs_radius[env_id, j] = radius
            env.dyn_obs_active[env_id, j] = True
            env.dyn_obs_scenario[env_id, j] = scenario
            
        if "real_room_dynamic_clutter" in modes:

            _spawn_real_room_dynamic_clutter(
                env=env,
                env_id=env_id,
                path_xy=path[env_id],
                valid_count=vc,
                num_obs=num_obs,
            )

def update_dynamic_obstacles_tensor(env, env_ids: torch.Tensor | None = None):
    """Move tensor obstacles and respawn inactive/out-of-range ones ahead of the robot."""
    if not hasattr(env, "dyn_obs_xy"):
        return

    if env_ids is None or len(env_ids) == 0:
        env_ids = torch.arange(env.num_envs, device=env.device)

    dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 1.0 / 30.0)))
    env.dyn_obs_xy[env_ids] += env.dyn_obs_vel_xy[env_ids] * dt * env.dyn_obs_active[env_ids].unsqueeze(-1).float()

    # Nonlinear motion for scenario 4:
    # linear forward motion + sinusoidal sideways wobble.
    nonlinear_mask = env.dyn_obs_active[env_ids] & (env.dyn_obs_scenario[env_ids] == 4)

    if torch.any(nonlinear_mask):
        old_phase = env.dyn_obs_phase[env_ids].clone()

        env.dyn_obs_phase[env_ids] += (
            env.dyn_obs_omega[env_ids]
            * dt
            * nonlinear_mask.float()
        )

        delta_wobble = (
            torch.sin(env.dyn_obs_phase[env_ids])
            - torch.sin(old_phase)
        ) * env.dyn_obs_amp[env_ids]

        env.dyn_obs_xy[env_ids] += (
            env.dyn_obs_normal[env_ids]
            * delta_wobble.unsqueeze(-1)
            * nonlinear_mask.unsqueeze(-1).float()
        )

    # Deactivate obstacles far from robot; next reset creates new scenarios.
    robot = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    rel = env.dyn_obs_xy[env_ids] - robot_xy[:, None, :]
    dist = torch.norm(rel, dim=-1)
    far = dist > float(getattr(env.cfg, "dynamic_obstacle_deactivate_range", 6.0))
    env.dyn_obs_active[env_ids] = env.dyn_obs_active[env_ids] & (~far)
    if hasattr(env, "navrl_global_path_xy"):
        dynamic_path_blockage(
            env,
            lookahead_points=config.OBSERVATIONS["actor"]["path_blocked"]["params"]["lookahead_points"],
            path_radius=config.OBSERVATIONS["actor"]["path_blocked"]["params"]["path_radius"],
        )


def draw_nav2_map_path_scan_debug(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    map_stride: int = 1,
    max_map_points: int = 30000,
    path_stride: int = 4,
    num_rays: int = 72,
    max_range: float = 4.0,
    step_size: float = 0.05,
):
    """Draw map, path, and lidar scan using Isaac DebugDraw.

    Debug only:
    - no PhysX objects
    - no collision
    - no lidar interference
    - other robots are still ignored by map_based_scan
    """

    if not bool(getattr(env.cfg, "debug_draw_nav2", True)):
        return

    draw_map = bool(getattr(env.cfg, "debug_draw_map", True))
    draw_path = bool(getattr(env.cfg, "debug_draw_path", True))
    draw_lidar = bool(getattr(env.cfg, "debug_draw_lidar", True))
    draw_dyn = bool(getattr(env.cfg, "debug_draw_dynamic_obstacles", True))

    _ensure_nav2_map(env)

    draw = _get_debug_draw()
    _clear_debug_draw(draw)

    robot = env.scene[asset_cfg.name]
    device = robot.data.root_pos_w.device

    # If env_ids is empty, draw all envs.
    if env_ids is None or len(env_ids) == 0:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # DebugDraw can become heavy with many envs.
    # For now, draw all envs passed by EventManager.
    # draw_env_ids = env_ids
    # Keep debug drawing cheap.
    draw_env_ids = env_ids[: min(len(env_ids), int(getattr(env.cfg, "debug_draw_max_envs", 4)))]


    # --------------------------------------------------
    # Draw occupied map for envs
    # --------------------------------------------------
    if draw_map:
        occ_xy = env.nav2_occupancy_map.debug_occupied_world_points(
            stride=map_stride,
            max_points=max_map_points,
        )

        all_occ_points = []

        for eid_t in draw_env_ids:
            eid = int(eid_t.item())
            origin = env.scene.env_origins[eid, :2]

            occ_world = occ_xy + origin[None, :]

            for p in occ_world:
                all_occ_points.append(
                    (
                        float(p[0].item()),
                        float(p[1].item()),
                        0.25,
                    )
                )

        if len(all_occ_points) > 0:
            draw.draw_points(
                all_occ_points,
                [(0.02, 0.02, 0.02, 1.0)] * len(all_occ_points),
                [6.0] * len(all_occ_points),
            )
    # ----------------------------
    # Draw Nav2 global path for envs
    # ----------------------------
    if draw_path:
        all_p0 = []
        all_p1 = []

        for eid_t in draw_env_ids:
            eid = int(eid_t.item())
            origin = env.scene.env_origins[eid, :2]

            valid_count = int(env.navrl_path_valid_count[eid].item())
            if valid_count <= 1:
                continue

            path_xy = env.navrl_global_path_xy[eid, :valid_count:path_stride]
            path_world = path_xy + origin[None, :]

            for i in range(path_world.shape[0] - 1):
                a = path_world[i]
                b = path_world[i + 1]

                all_p0.append((float(a[0].item()), float(a[1].item()), 0.12))
                all_p1.append((float(b[0].item()), float(b[1].item()), 0.12))

        if len(all_p0) > 0:
            draw.draw_lines(
                all_p0,
                all_p1,
                [(0.0, 0.3, 1.0, 1.0)] * len(all_p0),
                [3.0] * len(all_p0),
            )
    # if draw_dyn and hasattr(env, "dyn_obs_xy"):
    #     points, sizes = [], []
    #     for eid_t in draw_env_ids:
    #         eid = int(eid_t.item())
    #         origin = env.scene.env_origins[eid, :2]
    #         active_ids = torch.nonzero(env.dyn_obs_active[eid], as_tuple=False).flatten()
    #         for oid in active_ids:
    #             p = env.dyn_obs_xy[eid, oid] + origin
    #             points.append((float(p[0].item()), float(p[1].item()), 0.35))
    #             sizes.append(float(max(8.0, env.dyn_obs_radius[eid, oid].item() * 60.0)))
    #     if len(points) > 0:
    #         draw.draw_points(points, [(1.0, 0.45, 0.0, 1.0)] * len(points), sizes)
    if draw_dyn and hasattr(env, "dyn_obs_xy"):
        center_points = []
        circle_p0 = []
        circle_p1 = []

        num_circle_segments = 32
        circle_angles = torch.linspace(0.0,2.0 * math.pi,num_circle_segments + 1,device=device,)

        for eid_t in draw_env_ids:
            eid = int(eid_t.item())
            origin = env.scene.env_origins[eid, :2]
            active_ids = torch.nonzero(env.dyn_obs_active[eid], as_tuple=False).flatten()

            for oid in active_ids:
                center = env.dyn_obs_xy[eid, oid] + origin
                radius = env.dyn_obs_radius[eid, oid]
                center_points.append((float(center[0].item()),float(center[1].item()),0.35,))
                xs = center[0] + torch.cos(circle_angles) * radius
                ys = center[1] + torch.sin(circle_angles) * radius
                for k in range(num_circle_segments):
                    circle_p0.append((float(xs[k].item()),float(ys[k].item()),0.32,))
                    circle_p1.append((float(xs[k + 1].item()),float(ys[k + 1].item()),0.32,))

        if len(circle_p0) > 0:
            draw.draw_lines(circle_p0,circle_p1,[(1.0, 0.45, 0.0, 1.0)] * len(circle_p0),[2.0] * len(circle_p0),)

        if len(center_points) > 0:
            draw.draw_points(center_points,[(1.0, 0.8, 0.0, 1.0)] * len(center_points),[5.0] * len(center_points),)

    # Goal points for all envs
    if hasattr(env, "navrl_final_goal_xy"):
        goal_points = []

        for eid_t in draw_env_ids:
            eid = int(eid_t.item())
            origin = env.scene.env_origins[eid, :2]
            g = env.navrl_final_goal_xy[eid] + origin

            goal_points.append((float(g[0].item()), float(g[1].item()), 0.35))

        if len(goal_points) > 0:
            draw.draw_points(
                goal_points,
                [(0.0, 1.0, 0.0, 1.0)] * len(goal_points),
                [18.0] * len(goal_points),
            )

    if not draw_lidar:
        return

    # --------------------------------------------------
    # Draw lidar scan for ALL envs
    # --------------------------------------------------
    all_start_lines = []
    all_end_lines = []
    all_hit_points = []

    ray_angles = torch.linspace(-math.pi, math.pi, num_rays + 1, device=device)[:-1]

    for eid_t in draw_env_ids:
        eid = int(eid_t.item())

        origin = env.scene.env_origins[eid, :2]

        robot_xy_world = robot.data.root_pos_w[eid, :2]
        robot_xy = robot_xy_world - origin
        yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[eid].unsqueeze(0))[0]
        # Use combined scan if available, otherwise static map scan.
        # try:
        #     from .observations import combined_static_dynamic_scan
        #     scan_norm = combined_static_dynamic_scan(env, num_rays=num_rays, max_range=max_range, step_size=step_size)[eid]
        # except Exception:
        #     scan_norm = env.nav2_occupancy_map.raycast_scan(robot_xy=robot_xy.unsqueeze(0), robot_yaw=yaw.unsqueeze(0), num_rays=num_rays, max_range=max_range, step_size=step_size)[0]

        # scan_norm = env.nav2_occupancy_map.raycast_scan(
        #     robot_xy=robot_xy.unsqueeze(0),
        #     robot_yaw=yaw.unsqueeze(0),
        #     num_rays=num_rays,
        #     max_range=max_range,
        #     step_size=step_size,
        # )[0]
        scan_norm = combined_static_dynamic_scan(
            env,
            num_rays=num_rays,
            max_range=max_range,
            step_size=step_size,
        )[eid]
        scan_m = scan_norm * max_range
        world_angles = yaw + ray_angles

        robot_draw_xy = robot_xy + origin

        for i in range(num_rays):
            hx = robot_xy[0] + torch.cos(world_angles[i]) * scan_m[i]
            hy = robot_xy[1] + torch.sin(world_angles[i]) * scan_m[i]

            hit_world = torch.stack([hx, hy]) + origin

            all_start_lines.append(
                (
                    float(robot_draw_xy[0].item()),
                    float(robot_draw_xy[1].item()),
                    0.18,
                )
            )

            all_end_lines.append(
                (
                    float(hit_world[0].item()),
                    float(hit_world[1].item()),
                    0.18,
                )
            )

            all_hit_points.append(
                (
                    float(hit_world[0].item()),
                    float(hit_world[1].item()),
                    0.20,
                )
            )

    if len(all_start_lines) > 0:
        draw.draw_lines(
            all_start_lines,
            all_end_lines,
            [(1.0, 0.0, 0.0, 0.65)] * len(all_start_lines),
            [1.0] * len(all_start_lines),
        )

        draw.draw_points(
            all_hit_points,
            [(1.0, 0.0, 0.0, 1.0)] * len(all_hit_points),
            [5.0] * len(all_hit_points),
        )

def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

def _ensure_training_reward_buffers(env, max_path_points: int = 600):
    if not hasattr(env, "navrl_path_cum_s"):
        env.navrl_path_cum_s = torch.zeros(
            env.num_envs,
            max_path_points,
            device=env.device,
        )

    if not hasattr(env, "navrl_prev_progress_s"):
        env.navrl_prev_progress_s = torch.zeros(
            env.num_envs,
            device=env.device,
        )

    if not hasattr(env, "navrl_prev_action_for_reward"):
        env.navrl_prev_action_for_reward = torch.zeros(
            env.num_envs,
            3,
            device=env.device,
        )

def _ensure_performance_curriculum_buffers(env):
    device = env.device
    window = int(getattr(env.cfg, "curriculum_perf_window", 1000))

    if not hasattr(env, "navrl_perf_success"):
        env.navrl_perf_success = torch.zeros(window, device=device)
        env.navrl_perf_map_collision = torch.zeros(window, device=device)
        env.navrl_perf_dynamic_collision = torch.zeros(window, device=device)
        env.navrl_perf_timeout = torch.zeros(window, device=device)
        env.navrl_perf_idx = 0
        env.navrl_perf_filled = 0
        env.navrl_curriculum_level_global = 0

def _ensure_domain_randomization_buffers(env):
    device = env.device

    if not hasattr(env, "dr_lidar_rays"):
        env.dr_lidar_rays = torch.ones(env.num_envs, dtype=torch.long, device=device) * int(
            getattr(env.cfg, "lidar_level_0_rays", 72)
        )

    if not hasattr(env, "dr_scan_noise_std"):
        env.dr_scan_noise_std = torch.zeros(env.num_envs, device=device)

    if not hasattr(env, "dr_scan_dropout_prob"):
        env.dr_scan_dropout_prob = torch.zeros(env.num_envs, device=device)

    if not hasattr(env, "motor_strength_scale"):
        env.motor_strength_scale = torch.ones(env.num_envs, device=device)

    if not hasattr(env, "action_delay_steps"):
        env.action_delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    if not hasattr(env, "dr_mass_scale"):
        env.dr_mass_scale = torch.ones(env.num_envs, device=device)

    if not hasattr(env, "dr_com_shift_norm"):
        env.dr_com_shift_norm = torch.zeros(env.num_envs, device=device)
    
    if not hasattr(env, "wheel_slip_scale"):
        env.wheel_slip_scale = torch.ones(env.num_envs, 4, device=device)

    if not hasattr(env, "wheel_radius_scale"):
        env.wheel_radius_scale = torch.ones(env.num_envs, 4, device=device)

def update_performance_curriculum_on_reset(env, env_ids: torch.Tensor):
    _ensure_performance_curriculum_buffers(env)

    window = env.navrl_perf_success.shape[0]

    # These buffers are usually available in ManagerBasedRLEnv.
    terminated = env.termination_manager.terminated[env_ids]
    time_out = env.termination_manager.time_outs[env_ids]

    # Individual termination terms
    final_goal = env.termination_manager.get_term("final_goal_reached")[env_ids]
    map_col = env.termination_manager.get_term("map_collision")[env_ids]
    dyn_col = env.termination_manager.get_term("dynamic_collision")[env_ids]

    for i in range(len(env_ids)):
        idx = env.navrl_perf_idx % window

        env.navrl_perf_success[idx] = final_goal[i].float()
        env.navrl_perf_map_collision[idx] = map_col[i].float()
        env.navrl_perf_dynamic_collision[idx] = dyn_col[i].float()
        env.navrl_perf_timeout[idx] = time_out[i].float()

        env.navrl_perf_idx += 1
        env.navrl_perf_filled = min(env.navrl_perf_filled + 1, window)

    # Do not promote before enough data exists.
    min_samples = int(getattr(env.cfg, "curriculum_min_samples", 300))
    if env.navrl_perf_filled < min_samples:
        return

    n = env.navrl_perf_filled
    success_rate = env.navrl_perf_success[:n].mean()
    map_rate = env.navrl_perf_map_collision[:n].mean()
    dyn_rate = env.navrl_perf_dynamic_collision[:n].mean()
    timeout_rate = env.navrl_perf_timeout[:n].mean()

    promote = (
        success_rate >= float(getattr(env.cfg, "curriculum_success_promote", 0.70))
        and map_rate <= float(getattr(env.cfg, "curriculum_map_collision_promote", 0.10))
        and dyn_rate <= float(getattr(env.cfg, "curriculum_dynamic_collision_promote", 0.05))
        and timeout_rate <= float(getattr(env.cfg, "curriculum_timeout_promote", 0.20))
    )

    demote = (
        success_rate < float(getattr(env.cfg, "curriculum_success_demote", 0.30))
        or map_rate > float(getattr(env.cfg, "curriculum_map_collision_demote", 0.30))
        or dyn_rate > float(getattr(env.cfg, "curriculum_dynamic_collision_demote", 0.20))
        or timeout_rate > float(getattr(env.cfg, "curriculum_timeout_demote", 0.50))
    )

    # max_level = int(getattr(env.cfg, "curriculum_max_level", 5))
    max_level = int(getattr(env.cfg, "curriculum_max_level", -1))
    if max_level < 0:
        raw = 0
        while True:
            obs_level, dr_level = get_curriculum_sublevels(env, raw)
            if obs_level >= get_num_obstacle_levels() and dr_level >= get_num_dr_levels():
                max_level = raw
                break
            raw += 1

    if promote:
        env.navrl_curriculum_level_global = min(env.navrl_curriculum_level_global + 1, max_level)
        env.navrl_perf_success.zero_()
        env.navrl_perf_map_collision.zero_()
        env.navrl_perf_dynamic_collision.zero_()
        env.navrl_perf_timeout.zero_()
        env.navrl_perf_idx = 0
        env.navrl_perf_filled = 0

    elif demote:
        if bool(getattr(env.cfg, "sample_curriculum_levels", False)):
            floor_level = int(getattr(env.cfg, "curriculum_sample_min_level", 0))
        else:
            floor_level = 0

        env.navrl_curriculum_level_global = max(
            env.navrl_curriculum_level_global - 1,
            floor_level,
        )
        env.navrl_perf_success.zero_()
        env.navrl_perf_map_collision.zero_()
        env.navrl_perf_dynamic_collision.zero_()
        env.navrl_perf_timeout.zero_()
        env.navrl_perf_idx = 0
        env.navrl_perf_filled = 0

    # Only overwrite all env levels when NOT using sampled curriculum.
    # If sample_curriculum_levels=True, reset_dynamic_obstacles_tensor()
    # will assign levels only for the resetting env_ids.
    if not bool(getattr(env.cfg, "sample_curriculum_levels", False)):
        env.navrl_curriculum_level[:] = int(env.navrl_curriculum_level_global)

def reset_domain_randomization(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg | None = None):
    _ensure_domain_randomization_buffers(env)

    if hasattr(env, "navrl_curriculum_level"):
        level = env.navrl_curriculum_level[env_ids]
    else:
        level = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
    
    dr_levels = torch.zeros_like(level)

    for i in range(len(env_ids)):
        raw_level = int(level[i].item())
        _, dr_level_i = get_curriculum_sublevels(env, raw_level)
        dr_levels[i] = dr_level_i

    dr_mask = dr_levels > 0
    # Defaults: no DR
    env.dr_lidar_rays[env_ids] = int(getattr(env.cfg, "lidar_level_0_rays", 72))
    env.dr_scan_noise_std[env_ids] = 0.0
    env.dr_scan_dropout_prob[env_ids] = 0.0
    env.motor_strength_scale[env_ids] = 1.0
    env.action_delay_steps[env_ids] = 0
    env.dr_mass_scale[env_ids] = 1.0
    env.dr_com_shift_norm[env_ids] = 0.0
    env.wheel_slip_scale[env_ids] = 1.0
    env.wheel_radius_scale[env_ids] = 1.0

    if not torch.any(dr_mask):
        return

    ids = env_ids[dr_mask]
    active_levels = dr_levels[dr_mask]
    stages = config.domain_randomization_stages

    # --------------------------------------------------
    # YAML-name-based cumulative DR.
    # If dr_level = 3, first 3 YAML stages are active.
    # No fixed 8/10-length table indexing.
    # --------------------------------------------------
    for local_i, env_id_t in enumerate(ids):
        env_id = int(env_id_t.item())
        dr_level = int(active_levels[local_i].item())
        active_dr = set(stages[:dr_level])

        if "lidar_rays" in active_dr:
            choices = torch.tensor([72, 96, 120, 144], dtype=torch.long, device=env.device)
            pick = torch.randint(0, choices.numel(), (1,), device=env.device)
            env.dr_lidar_rays[env_id] = choices[pick]

        if "scan_noise" in active_dr:
            env.dr_scan_noise_std[env_id] = torch.rand((), device=env.device) * 0.005

        if "scan_dropout" in active_dr:
            env.dr_scan_dropout_prob[env_id] = torch.rand((), device=env.device) * 0.01

        if "action_delay" in active_dr:
            env.action_delay_steps[env_id] = torch.randint(0, 2, (1,), device=env.device)

        if "motor_strength" in active_dr:
            env.motor_strength_scale[env_id] = 0.90 + torch.rand((), device=env.device) * 0.10

        if "mass" in active_dr:
            env.dr_mass_scale[env_id] = 0.90 + torch.rand((), device=env.device) * 0.20

        if "com_shift" in active_dr:
            com_max = 0.010
            shift_x = (torch.rand((), device=env.device) * 2.0 - 1.0) * com_max
            shift_y = (torch.rand((), device=env.device) * 2.0 - 1.0) * com_max
            shift_z = (
                (torch.rand((), device=env.device) * 2.0 - 1.0)
                * float(getattr(env.cfg, "com_z_m", 0.005))
            )

            env.dr_com_shift_norm[env_id] = torch.sqrt(
                shift_x * shift_x + shift_y * shift_y + shift_z * shift_z
            )

        if "wheel_radius" in active_dr:
            env.wheel_radius_scale[env_id] = (
                0.90 + torch.rand(4, device=env.device) * 0.20
            )

        if "wheel_slip" in active_dr:
            env.wheel_slip_scale[env_id] = (
                0.50 + torch.rand(4, device=env.device) * 0.50
            )

        if "combined_strong" in active_dr:
            env.dr_scan_noise_std[env_id] = torch.rand((), device=env.device) * 0.010
            env.dr_scan_dropout_prob[env_id] = torch.rand((), device=env.device) * 0.02
            env.motor_strength_scale[env_id] = 0.80 + torch.rand((), device=env.device) * 0.20
            env.action_delay_steps[env_id] = torch.randint(0, 3, (1,), device=env.device)
            env.dr_mass_scale[env_id] = 0.90 + torch.rand((), device=env.device) * 0.20
            env.wheel_radius_scale[env_id] = 0.90 + torch.rand(4, device=env.device) * 0.20
            env.wheel_slip_scale[env_id] = 0.50 + torch.rand(4, device=env.device) * 0.50

    # Apply mass randomization to PhysX if available.
    if asset_cfg is not None:
        robot = env.scene[asset_cfg.name]
        if hasattr(robot, "root_physx_view"):
            try:
                if not hasattr(env, "default_body_masses"):
                    env.default_body_masses = robot.root_physx_view.get_masses().clone()

                masses = env.default_body_masses.clone()
                masses[ids] = env.default_body_masses[ids] * env.dr_mass_scale[ids, None]
                robot.root_physx_view.set_masses(masses, ids)
            except Exception:
                pass

def randomize_robot_mass_com(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]

    if not bool(getattr(env.cfg, "mass_randomization_enable", True)) and not bool(
        getattr(env.cfg, "com_randomization_enable", True)
    ):
        return

    # Save default values once
    if not hasattr(env, "default_body_masses"):
        env.default_body_masses = robot.root_physx_view.get_masses().clone()

    if not hasattr(env, "default_body_coms"):
        env.default_body_coms = robot.root_physx_view.get_coms().clone()

    masses = env.default_body_masses.clone()
    coms = env.default_body_coms.clone()

    if bool(getattr(env.cfg, "mass_randomization_enable", True)):
        mass_min = float(getattr(env.cfg, "mass_scale_min", 0.85))
        mass_max = float(getattr(env.cfg, "mass_scale_max", 1.15))

        scale = torch.empty(
            len(env_ids),
            1,
            device=env.device,
        ).uniform_(mass_min, mass_max)

        masses[env_ids] = env.default_body_masses[env_ids] * scale

        env.dr_mass_scale = getattr(
            env,
            "dr_mass_scale",
            torch.ones(env.num_envs, device=env.device),
        )
        env.dr_mass_scale[env_ids] = scale.squeeze(-1)

    if bool(getattr(env.cfg, "com_randomization_enable", True)):
        sx = float(getattr(env.cfg, "com_shift_x_m", 0.02))
        sy = float(getattr(env.cfg, "com_shift_y_m", 0.02))
        sz = float(getattr(env.cfg, "com_shift_z_m", 0.01))

        shift = torch.zeros(len(env_ids), 1, 3, device=env.device)
        shift[:, :, 0] = torch.empty(len(env_ids), 1, device=env.device).uniform_(-sx, sx)
        shift[:, :, 1] = torch.empty(len(env_ids), 1, device=env.device).uniform_(-sy, sy)
        shift[:, :, 2] = torch.empty(len(env_ids), 1, device=env.device).uniform_(-sz, sz)

        coms[env_ids] = env.default_body_coms[env_ids] + shift

        if not hasattr(env, "dr_com_shift_norm"):
            env.dr_com_shift_norm = torch.zeros(env.num_envs, device=env.device)

        env.dr_com_shift_norm[env_ids] = torch.norm(shift.squeeze(1), dim=-1)

    robot.root_physx_view.set_masses(masses, env_ids)
    robot.root_physx_view.set_coms(coms, env_ids)

def get_num_obstacle_levels() -> int:
    return len(config.obstacle_stages)


def get_num_dr_levels() -> int:
    return len(config.domain_randomization_stages)


def get_curriculum_sublevels(env, raw_level: int) -> tuple[int, int]:
    order = str(getattr(env.cfg, "curriculum_order", "obstacles_first"))

    num_obstacle_levels = get_num_obstacle_levels()
    num_dr_levels = get_num_dr_levels()

    if order == "dr_first":
        dr_level = min(raw_level, num_dr_levels)
        obstacle_level = max(0, min(raw_level - num_dr_levels, num_obstacle_levels))
    elif order == 'mixed':
        obstacle_level = min(raw_level, num_obstacle_levels)
        dr_level = min(raw_level // 2, num_dr_levels)
    else:
        obstacle_level = min(raw_level, num_obstacle_levels)
        dr_level = max(0, min(raw_level - num_obstacle_levels, num_dr_levels))

    return obstacle_level, dr_level

def _spawn_real_room_dynamic_clutter(
    env,
    env_id: int,
    path_xy: torch.Tensor,
    valid_count: int,
    num_obs: int,
):
    """Extra deployment-style obstacle level.

    Adds:
    - static clutter like chairs/tables
    - nonlinear moving obstacles
    - obstacles sampled across the whole path, not only early path
    """

    if valid_count < 20:
        return

    free_slots = torch.nonzero(~env.dyn_obs_active[env_id], as_tuple=False).flatten()
    if free_slots.numel() == 0:
        return

    # Use up to 5 extra obstacles for this real-room stage.
    # 3 static clutter + 2 nonlinear movers if slots are available.
    max_extra = min(5, int(free_slots.numel()))
    slots = free_slots[:max_extra]

    n_static = min(3, max_extra)
    n_moving = max_extra - n_static

    start_idx = max(5, int(0.10 * valid_count))
    end_idx = max(start_idx + 1, min(valid_count - 5, int(0.90 * valid_count)))

    # -------------------------
    # Static table/chair clutter
    # -------------------------
    for k in range(n_static):
        slot = int(slots[k].item())

        idx = int(torch.randint(start_idx, end_idx, (1,), device=env.device).item())

        p0 = path_xy[idx]
        p1 = path_xy[min(idx + 4, valid_count - 1)]

        tangent = p1 - p0
        tangent = tangent / torch.norm(tangent).clamp_min(1e-6)
        normal = torch.stack([-tangent[1], tangent[0]])

        side = 1.0 if torch.rand((), device=env.device).item() > 0.5 else -1.0

        # Table/chair-like static clutter near, but not always exactly on, the path.
        lateral_offset = side * torch.empty((), device=env.device).uniform_(0.35, 1.30)
        along_jitter = torch.empty((), device=env.device).uniform_(-0.30, 0.30)

        pos = p0 + lateral_offset * normal + along_jitter * tangent

        radius = torch.empty((), device=env.device).uniform_(0.18, 0.38)

        env.dyn_obs_xy[env_id, slot] = pos
        env.dyn_obs_vel_xy[env_id, slot] = 0.0
        env.dyn_obs_radius[env_id, slot] = radius
        env.dyn_obs_active[env_id, slot] = True
        env.dyn_obs_scenario[env_id, slot] = 5  # static clutter

    # -------------------------
    # Nonlinear moving obstacles
    # -------------------------
    for k in range(n_moving):
        slot = int(slots[n_static + k].item())

        idx = int(torch.randint(start_idx, end_idx, (1,), device=env.device).item())

        p0 = path_xy[idx]
        p1 = path_xy[min(idx + 6, valid_count - 1)]

        tangent = p1 - p0
        tangent = tangent / torch.norm(tangent).clamp_min(1e-6)
        normal = torch.stack([-tangent[1], tangent[0]])

        side = 1.0 if torch.rand((), device=env.device).item() > 0.5 else -1.0

        lateral_offset = side * torch.empty((), device=env.device).uniform_(0.60, 1.50)
        pos = p0 + lateral_offset * normal

        speed = torch.empty((), device=env.device).uniform_(0.08, 0.25)
        radius = torch.empty((), device=env.device).uniform_(0.12, 0.24)

        env.dyn_obs_xy[env_id, slot] = pos
        env.dyn_obs_vel_xy[env_id, slot] = speed * tangent
        env.dyn_obs_radius[env_id, slot] = radius
        env.dyn_obs_active[env_id, slot] = True
        env.dyn_obs_scenario[env_id, slot] = 4  # nonlinear mover

        env.dyn_obs_phase[env_id, slot] = torch.empty((), device=env.device).uniform_(0.0, 2.0 * math.pi)
        env.dyn_obs_omega[env_id, slot] = torch.empty((), device=env.device).uniform_(0.8, 1.8)
        env.dyn_obs_amp[env_id, slot] = torch.empty((), device=env.device).uniform_(0.15, 0.45)
        env.dyn_obs_normal[env_id, slot] = normal

def print_curriculum_training_metrics(env, env_ids: torch.Tensor | None = None):
    if not hasattr(env, "dyn_obs_active"):
        return

    log_every = int(getattr(env.cfg, "curriculum_log_every_steps", 500))
    step_count = int(getattr(env, "common_step_counter", 0))

    if step_count % log_every != 0:
        return

    levels = env.navrl_curriculum_level
    active_count = env.dyn_obs_active.float().sum(dim=1)

    action_term = env.action_manager._terms["base_velocity"]
    action = action_term.processed_actions
    vx = action[:, 0]
    vy = action[:, 1]
    wz = action[:, 2]

    scenario_1 = (env.dyn_obs_scenario == 1).float().sum().item()
    scenario_2 = (env.dyn_obs_scenario == 2).float().sum().item()
    scenario_3 = (env.dyn_obs_scenario == 3).float().sum().item()

    print(
        "\n[TRAINING METRICS]\n"
        f"  step_count             : {step_count}\n"
        f"  reset_max              : {int(env.navrl_reset_count.max().item())}\n"
        f"  curriculum_level_mean  : {levels.float().mean().item():.2f}\n"
        f"  curriculum_level_min   : {int(levels.min().item())}\n"
        f"  curriculum_level_max   : {int(levels.max().item())}\n"
        f"  active_obstacles_mean  : {active_count.mean().item():.2f}\n"
        f"  active_obstacles_max   : {int(active_count.max().item())}\n"
        f"  stationary_blockers    : {int(scenario_1)}\n"
        f"  crossing_obstacles     : {int(scenario_2)}\n"
        f"  same_lane_obstacles    : {int(scenario_3)}\n"
        f"  mean_abs_vx            : {vx.abs().mean().item():.3f}\n"
        f"  mean_abs_vy            : {vy.abs().mean().item():.3f}\n"
        f"  mean_abs_wz            : {wz.abs().mean().item():.3f}\n"
        f"  vy_ratio               : {(vy.abs() / (vx.abs() + vy.abs() + 1e-6)).mean().item():.3f}\n",
        flush=True,
    )

def print_domain_randomization_metrics(env, env_ids: torch.Tensor | None = None):
    if not hasattr(env, "dr_lidar_rays"):
        return

    step_count = int(getattr(env, "common_step_counter", 0))
    log_every = int(getattr(env.cfg, "dr_log_every_steps", 500))
    if step_count % log_every != 0:
        return

    action_term = env.action_manager._terms["base_velocity"]
    action = action_term.processed_actions
    vx = action[:, 0]
    vy = action[:, 1]
    wz = action[:, 2]

    print(
        "\n[DOMAIN RANDOMIZATION METRICS]\n"
        f"  step_count             : {step_count}\n"
        f"  curriculum_level_mean  : {env.navrl_curriculum_level.float().mean().item():.2f}\n"
        f"  lidar_rays_mean        : {env.dr_lidar_rays.float().mean().item():.1f}\n"
        f"  scan_noise_std_mean    : {env.dr_scan_noise_std.mean().item():.4f}\n"
        f"  scan_dropout_mean      : {env.dr_scan_dropout_prob.mean().item():.4f}\n"
        f"  battery_scale_mean     : {env.motor_strength_scale.mean().item():.3f}\n"
        f"  battery_scale_min      : {env.motor_strength_scale.min().item():.3f}\n"
        f"  battery_scale_max      : {env.motor_strength_scale.max().item():.3f}\n"
        f"  action_delay_mean      : {env.action_delay_steps.float().mean().item():.2f}\n"
        f"  mass_scale_mean        : {env.dr_mass_scale.mean().item():.3f}\n"
        f"  com_shift_mean_m       : {env.dr_com_shift_norm.mean().item():.4f}\n"
        f"  mean_abs_vx            : {vx.abs().mean().item():.3f}\n"
        f"  mean_abs_vy            : {vy.abs().mean().item():.3f}\n"
        f"  mean_abs_wz            : {wz.abs().mean().item():.3f}\n"
        f"  vy_ratio               : {(vy.abs() / (vx.abs() + vy.abs() + 1e-6)).mean().item():.3f}\n",
        flush=True,
    )

def log_curriculum_progress(env, env_ids: torch.Tensor | None = None):
    if not hasattr(env, "navrl_curriculum_level"):
        return

    if not hasattr(env, "extras"):
        return

    if "log" not in env.extras:
        env.extras["log"] = {}

    levels = env.navrl_curriculum_level.float()

    env.extras["log"]["Curriculum/level_mean"] = levels.mean()
    env.extras["log"]["Curriculum/level_min"] = levels.min()
    env.extras["log"]["Curriculum/level_max"] = levels.max()

    if hasattr(env, "navrl_perf_filled") and env.navrl_perf_filled > 0:
        n = env.navrl_perf_filled

        env.extras["log"]["Curriculum/recent_success_rate"] = env.navrl_perf_success[:n].mean()
        env.extras["log"]["Curriculum/recent_map_collision_rate"] = env.navrl_perf_map_collision[:n].mean()
        env.extras["log"]["Curriculum/recent_dynamic_collision_rate"] = env.navrl_perf_dynamic_collision[:n].mean()
        env.extras["log"]["Curriculum/recent_timeout_rate"] = env.navrl_perf_timeout[:n].mean()
        env.extras["log"]["Curriculum/perf_samples"] = torch.tensor(float(n), device=env.device)


def log_map_collision_directions(env, env_ids: torch.Tensor | None = None):
    if not hasattr(env, "extras"):
        return

    if "log" not in env.extras:
        env.extras["log"] = {}

    flags = map_collision_direction_flags(
        env,
        radius=0.22,
        num_points=32,
    )

    env.extras["log"]["MapCollision/front_rate"] = flags["front"].float().mean()
    env.extras["log"]["MapCollision/left_rate"] = flags["left"].float().mean()
    env.extras["log"]["MapCollision/right_rate"] = flags["right"].float().mean()
    env.extras["log"]["MapCollision/rear_rate"] = flags["rear"].float().mean()
    env.extras["log"]["MapCollision/center_rate"] = flags["center"].float().mean()