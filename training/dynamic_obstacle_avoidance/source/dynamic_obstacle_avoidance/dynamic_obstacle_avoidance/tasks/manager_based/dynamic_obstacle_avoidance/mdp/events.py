from __future__ import annotations

import math

import torch
from .path_dataset import Nav2PathDataset
from .nav2_map import Nav2OccupancyMap
from isaaclab.managers import SceneEntityCfg

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
    """Load one real Nav2 global path per reset and place robot at path start."""

    _ensure_nav2_path_buffers(env, max_path_points=max_path_points)

    robot = env.scene[asset_cfg.name]

    starts, goals, paths, valid_counts, path_lengths = env.nav2_path_dataset.sample_batch(env_ids)

    env.navrl_start_pose[env_ids] = starts
    env.navrl_goal_pose[env_ids] = goals
    env.navrl_global_path_xy[env_ids] = paths
    env.navrl_path_valid_count[env_ids] = valid_counts
    env.navrl_final_goal_xy[env_ids] = goals[:, :2]

    root_state = robot.data.default_root_state[env_ids].clone()

    root_state[:, 0] = starts[:, 0] + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = starts[:, 1] + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = 0.05

    yaw = starts[:, 2]
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

    validate_nav2_path_against_map(
        env=env,
        env_ids=env_ids,
        max_bad_fraction=0.02,
    )



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

    _ensure_nav2_map(env)

    draw = _get_debug_draw()
    _clear_debug_draw(draw)

    robot = env.scene[asset_cfg.name]

    # If env_ids is empty, draw all envs.
    if env_ids is None or len(env_ids) == 0:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # DebugDraw can become heavy with many envs.
    # For now, draw all envs passed by EventManager.
    draw_env_ids = env_ids

    # --------------------------------------------------
    # Draw map/path only for first env to avoid clutter
    # --------------------------------------------------
    first_env_id = int(draw_env_ids[0].item())
    # first_origin = env.scene.env_origins[first_env_id, :2]

    # --------------------------------------------------
    # Draw occupied map for ALL envs
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
    # Draw Nav2 global path for ALL envs
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

    ray_angles = torch.linspace(-math.pi, math.pi, num_rays, device=env.device)

    for eid_t in draw_env_ids:
        eid = int(eid_t.item())

        origin = env.scene.env_origins[eid, :2]

        robot_xy_world = robot.data.root_pos_w[eid, :2]
        robot_xy = robot_xy_world - origin

        q = robot.data.root_quat_w[eid]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        scan_norm = env.nav2_occupancy_map.raycast_scan(
            robot_xy=robot_xy.unsqueeze(0),
            robot_yaw=yaw.unsqueeze(0),
            num_rays=num_rays,
            max_range=max_range,
            step_size=step_size,
            unknown_is_occupied=True,
        )[0]

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