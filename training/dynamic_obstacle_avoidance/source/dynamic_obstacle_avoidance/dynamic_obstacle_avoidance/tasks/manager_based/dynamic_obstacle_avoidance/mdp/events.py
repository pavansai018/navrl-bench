from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


CELL_SIZE = 0.50
HIDDEN_Z = -10.0


MAP_TEMPLATES = {
    "corridor_crossing": [
        "########################",
        "#..........D...........#",
        "#..........D...........#",
        "#####.############.#####",
        "#S....................G#",
        "#####.############.#####",
        "#..........D...........#",
        "#..........D...........#",
        "########################",
    ],
    "doorway_blockage": [
        "########################",
        "#S.........#...........#",
        "#..........#...........#",
        "#..........#...........#",
        "#####.######.######.####",
        "#..........D..........G#",
        "#####.######.######.####",
        "#..........#...........#",
        "#..........#...........#",
        "########################",
    ],
    "t_junction_crossing": [
        "########################",
        "#..........D...........#",
        "#..........D...........#",
        "###########.############",
        "#S....................G#",
        "###########.############",
        "#..........D...........#",
        "#..........D...........#",
        "########################",
    ],
}


def _ensure_navrl_buffers(env):
    device = env.device

    if not hasattr(env, "navrl_final_goal_xy"):
        env.navrl_final_goal_xy = torch.zeros(env.num_envs, 2, device=device)

    if not hasattr(env, "navrl_path_points"):
        env.navrl_path_points = torch.zeros(env.num_envs, 32, 2, device=device)

    if not hasattr(env, "navrl_lookahead_xy"):
        env.navrl_lookahead_xy = torch.zeros(env.num_envs, 2, device=device)

    if not hasattr(env, "navrl_previous_goal_distance"):
        env.navrl_previous_goal_distance = torch.zeros(env.num_envs, device=device)

    if not hasattr(env, "navrl_previous_path_progress"):
        env.navrl_previous_path_progress = torch.zeros(env.num_envs, device=device)

    if not hasattr(env, "navrl_dynamic_xy"):
        env.navrl_dynamic_xy = torch.zeros(env.num_envs, 8, 2, device=device)

    if not hasattr(env, "navrl_dynamic_vel_xy"):
        env.navrl_dynamic_vel_xy = torch.zeros(env.num_envs, 8, 2, device=device)

    if not hasattr(env, "navrl_static_xy"):
        env.navrl_static_xy = torch.zeros(env.num_envs, 12, 2, device=device)

    if not hasattr(env, "navrl_wall_xy"):
        env.navrl_wall_xy = torch.zeros(env.num_envs, 96, 2, device=device)

    if not hasattr(env, "navrl_active_dynamic_count"):
        env.navrl_active_dynamic_count = 0

    if not hasattr(env, "navrl_active_static_count"):
        env.navrl_active_static_count = 0

    if not hasattr(env, "navrl_active_wall_count"):
        env.navrl_active_wall_count = 0


def _env_origins(env, env_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(env.scene, "env_origins"):
        return env.scene.env_origins[env_ids, :2]
    return torch.zeros(len(env_ids), 2, device=env.device)


def _grid_to_xy(row: int, col: int, rows: int, cols: int):
    x = (col - (cols - 1) / 2.0) * CELL_SIZE
    y = ((rows - 1) / 2.0 - row) * CELL_SIZE
    return x, y


def _parse_template(template_name: str):
    grid = MAP_TEMPLATES[template_name]
    rows = len(grid)
    cols = len(grid[0])

    wall_xy = []
    free_xy = []
    dynamic_lane_xy = []
    start_xy = None
    goal_xy = None

    for r, line in enumerate(grid):
        for c, char in enumerate(line):
            x, y = _grid_to_xy(r, c, rows, cols)

            if char == "#":
                wall_xy.append((x, y))
            elif char == "D":
                dynamic_lane_xy.append((x, y))
                free_xy.append((x, y))
            elif char == "S":
                start_xy = (x, y)
                free_xy.append((x, y))
            elif char == "G":
                goal_xy = (x, y)
                free_xy.append((x, y))
            else:
                free_xy.append((x, y))

    return wall_xy, free_xy, dynamic_lane_xy, start_xy, goal_xy


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros(len(yaw), 4, device=yaw.device)
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat


def _collection_pose(env, env_ids: torch.Tensor, xy: torch.Tensor, z: float, max_count: int):
    pose = torch.zeros(len(env_ids), max_count, 7, device=env.device)
    pose[..., 2] = HIDDEN_Z
    pose[..., 3] = 1.0

    if xy.numel() == 0:
        return pose

    origins = _env_origins(env, env_ids)
    count = xy.shape[1]

    pose[:, :count, 0:2] = xy + origins[:, None, :]
    pose[:, :count, 2] = z
    pose[:, :count, 3] = 1.0

    return pose


def _write_collection_pose(collection, pose: torch.Tensor, env_ids: torch.Tensor):
    if hasattr(collection, "write_object_pose_to_sim"):
        collection.write_object_pose_to_sim(pose, env_ids=env_ids)
    else:
        state = torch.zeros(*pose.shape[:-1], 13, device=pose.device)
        state[..., 0:7] = pose
        collection.write_object_state_to_sim(state, env_ids=env_ids)


def _write_marker_pose(marker, env, env_ids: torch.Tensor, xy: torch.Tensor, z: float):
    origins = _env_origins(env, env_ids)

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, 0:2] = xy + origins
    pose[:, 2] = z
    pose[:, 3] = 1.0

    marker.write_root_pose_to_sim(pose, env_ids=env_ids)


def _robot_xy(env, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor):
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[env_ids, :2] - _env_origins(env, env_ids)


def reset_runtime_map(
    env,
    env_ids: torch.Tensor,
    wall_asset_cfg: SceneEntityCfg,
    static_asset_cfg: SceneEntityCfg,
    dynamic_asset_cfg: SceneEntityCfg,
    template_name: str = "random",
    num_static_obstacles: int = 6,
    num_dynamic_obstacles: int = 4,
):
    _ensure_navrl_buffers(env)

    if template_name == "random":
        names = list(MAP_TEMPLATES.keys())
        template_name = names[torch.randint(0, len(names), (1,), device=env.device).item()]

    wall_xy, free_xy, dynamic_lane_xy, _, _ = _parse_template(template_name)

    walls = env.scene[wall_asset_cfg.name]
    static_obstacles = env.scene[static_asset_cfg.name]
    dynamic_obstacles = env.scene[dynamic_asset_cfg.name]

    max_walls = walls.num_objects
    max_static = static_obstacles.num_objects
    max_dynamic = dynamic_obstacles.num_objects

    wall_xy = torch.tensor(wall_xy[:max_walls], device=env.device, dtype=torch.float32)
    wall_xy = wall_xy.unsqueeze(0).repeat(len(env_ids), 1, 1)

    wall_pose = _collection_pose(env, env_ids, wall_xy, z=0.25, max_count=max_walls)
    _write_collection_pose(walls, wall_pose, env_ids)

    env.navrl_wall_xy[env_ids, : wall_xy.shape[1]] = wall_xy
    env.navrl_active_wall_count = wall_xy.shape[1]

    free_xy = torch.tensor(free_xy, device=env.device, dtype=torch.float32)
    dynamic_lane_xy = torch.tensor(dynamic_lane_xy, device=env.device, dtype=torch.float32)

    static_pose_xy = torch.zeros(len(env_ids), num_static_obstacles, 2, device=env.device)
    for i in range(len(env_ids)):
        ids = torch.randint(0, free_xy.shape[0], (num_static_obstacles,), device=env.device)
        static_pose_xy[i] = free_xy[ids]

    static_pose = _collection_pose(env, env_ids, static_pose_xy, z=0.225, max_count=max_static)
    _write_collection_pose(static_obstacles, static_pose, env_ids)

    env.navrl_static_xy[env_ids, :num_static_obstacles] = static_pose_xy
    env.navrl_active_static_count = num_static_obstacles

    if dynamic_lane_xy.shape[0] == 0:
        dynamic_lane_xy = free_xy

    dynamic_pose_xy = torch.zeros(len(env_ids), num_dynamic_obstacles, 2, device=env.device)
    for i in range(len(env_ids)):
        ids = torch.randint(0, dynamic_lane_xy.shape[0], (num_dynamic_obstacles,), device=env.device)
        dynamic_pose_xy[i] = dynamic_lane_xy[ids]

    dynamic_pose = _collection_pose(env, env_ids, dynamic_pose_xy, z=0.325, max_count=max_dynamic)
    _write_collection_pose(dynamic_obstacles, dynamic_pose, env_ids)

    env.navrl_dynamic_xy[env_ids, :num_dynamic_obstacles] = dynamic_pose_xy
    env.navrl_active_dynamic_count = num_dynamic_obstacles


def reset_robot_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range=(-4.5, -3.5),
    y_range=(-0.6, 0.6),
    yaw_range=(-0.15, 0.15),
):
    robot = env.scene[asset_cfg.name]
    origins = _env_origins(env, env_ids)

    x = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
    y = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(*yaw_range)

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, 0:2] = torch.stack([x, y], dim=-1) + origins
    pose[:, 2] = 0.05
    pose[:, 3:7] = _quat_from_yaw(yaw)

    velocity = torch.zeros(len(env_ids), 6, device=env.device)

    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def reset_path_command(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | None = None,
    final_goal_marker_cfg: SceneEntityCfg | None = None,
    lookahead_marker_cfg: SceneEntityCfg | None = None,
    start_x_range=(-4.5, -3.5),
    start_y_range=(-0.6, 0.6),
    goal_x_range=(3.5, 4.5),
    goal_y_range=(-0.8, 0.8),
    lookahead_distance: float = 0.8,
):
    _ensure_navrl_buffers(env)

    start_x = torch.empty(len(env_ids), device=env.device).uniform_(*start_x_range)
    start_y = torch.empty(len(env_ids), device=env.device).uniform_(*start_y_range)
    goal_x = torch.empty(len(env_ids), device=env.device).uniform_(*goal_x_range)
    goal_y = torch.empty(len(env_ids), device=env.device).uniform_(*goal_y_range)

    start = torch.stack([start_x, start_y], dim=-1)
    goal = torch.stack([goal_x, goal_y], dim=-1)

    env.navrl_final_goal_xy[env_ids] = goal

    num_points = env.navrl_path_points.shape[1]
    t = torch.linspace(0.0, 1.0, num_points, device=env.device).view(1, num_points, 1)
    path = start[:, None, :] * (1.0 - t) + goal[:, None, :] * t
    env.navrl_path_points[env_ids] = path

    direction = goal - start
    distance = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
    lookahead = start + direction / distance * lookahead_distance
    env.navrl_lookahead_xy[env_ids] = lookahead

    env.navrl_previous_goal_distance[env_ids] = torch.norm(goal - start, dim=-1)
    env.navrl_previous_path_progress[env_ids] = torch.zeros(len(env_ids), device=env.device)

    if final_goal_marker_cfg is not None:
        final_marker = env.scene[final_goal_marker_cfg.name]
        _write_marker_pose(final_marker, env, env_ids, goal, z=0.16)

    if lookahead_marker_cfg is not None:
        lookahead_marker = env.scene[lookahead_marker_cfg.name]
        _write_marker_pose(lookahead_marker, env, env_ids, lookahead, z=0.10)


def reset_dynamic_obstacles(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.35,
    max_speed: float = 1.20,
    num_active: int = 4,
):
    _ensure_navrl_buffers(env)

    angles = torch.rand(len(env_ids), num_active, device=env.device) * 2.0 * math.pi
    speeds = torch.empty(len(env_ids), num_active, device=env.device).uniform_(min_speed, max_speed)

    env.navrl_dynamic_vel_xy[env_ids, :num_active, 0] = speeds * torch.cos(angles)
    env.navrl_dynamic_vel_xy[env_ids, :num_active, 1] = speeds * torch.sin(angles)
    env.navrl_active_dynamic_count = num_active


def move_dynamic_obstacles(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_limit: float = 5.5,
    y_limit: float = 3.0,
):
    _ensure_navrl_buffers(env)

    n = env.navrl_active_dynamic_count
    if n <= 0:
        return

    dynamic_obstacles = env.scene[asset_cfg.name]
    max_dynamic = dynamic_obstacles.num_objects

    dt = env.step_dt

    xy = env.navrl_dynamic_xy[env_ids, :n]
    vel = env.navrl_dynamic_vel_xy[env_ids, :n]

    xy = xy + vel * dt

    hit_x = torch.abs(xy[..., 0]) > x_limit
    hit_y = torch.abs(xy[..., 1]) > y_limit

    vel[..., 0] = torch.where(hit_x, -vel[..., 0], vel[..., 0])
    vel[..., 1] = torch.where(hit_y, -vel[..., 1], vel[..., 1])

    xy[..., 0] = torch.clamp(xy[..., 0], -x_limit, x_limit)
    xy[..., 1] = torch.clamp(xy[..., 1], -y_limit, y_limit)

    env.navrl_dynamic_xy[env_ids, :n] = xy
    env.navrl_dynamic_vel_xy[env_ids, :n] = vel

    pose = _collection_pose(env, env_ids, xy, z=0.325, max_count=max_dynamic)
    _write_collection_pose(dynamic_obstacles, pose, env_ids)


def update_lookahead_target(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    lookahead_marker_cfg: SceneEntityCfg | None = None,
    lookahead_distance: float = 0.8,
):
    _ensure_navrl_buffers(env)

    robot_xy = _robot_xy(env, asset_cfg, env_ids)
    path = env.navrl_path_points[env_ids]

    distances = torch.norm(path - robot_xy[:, None, :], dim=-1)
    closest_ids = torch.argmin(distances, dim=-1)

    lookahead_ids = closest_ids.clone()

    for i in range(len(env_ids)):
        distance_sum = 0.0
        j = int(closest_ids[i].item())

        while j < path.shape[1] - 1 and distance_sum < lookahead_distance:
            segment_length = torch.norm(path[i, j + 1] - path[i, j])
            distance_sum += float(segment_length.item())
            j += 1

        lookahead_ids[i] = j

    lookahead = path[torch.arange(len(env_ids), device=env.device), lookahead_ids]
    env.navrl_lookahead_xy[env_ids] = lookahead

    if lookahead_marker_cfg is not None:
        marker = env.scene[lookahead_marker_cfg.name]
        _write_marker_pose(marker, env, env_ids, lookahead, z=0.10)