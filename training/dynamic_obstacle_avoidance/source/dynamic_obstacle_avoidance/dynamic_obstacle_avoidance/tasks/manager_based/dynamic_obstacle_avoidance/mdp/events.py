from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg

CELL_SIZE = 0.6
HIDDEN_Z = -10.0

SCENARIOS = [
    'narrow_corridor_crossing',
    'doorway_human_crossing',
    'blind_corner_intercept',
    'static_dynamic',
    'crossing_stream'
]



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


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros(len(yaw), 4, device=yaw.device)
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat



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
    num_static_obstacles: int = 5,
    num_dynamic_obstacles: int = 3,
):
    _ensure_navrl_buffers(env)

    if template_name == "random":
        scenario_id = torch.randint(0, len(SCENARIOS), (1,), device=env.device).item()
        template_name = SCENARIOS[scenario_id]

    wall_segments, static_obstacles, dynamic_lanes, start, goal = _get_scenario_layout(template_name)

    walls = env.scene[wall_asset_cfg.name]
    static_obs = env.scene[static_asset_cfg.name]
    dynamic_obs = env.scene[dynamic_asset_cfg.name]

    max_walls = walls.num_objects
    max_static = static_obs.num_objects
    max_dynamic = dynamic_obs.num_objects

    # ----------------------------
    # Walls
    # ----------------------------
    wall_pose = torch.zeros(len(env_ids), max_walls, 7, device=env.device)
    wall_pose[..., 2] = HIDDEN_Z
    wall_pose[..., 3] = 1.0

    env.navrl_wall_xy[env_ids] = 999.0

    active_wall_count = min(len(wall_segments), max_walls)

    origins = _env_origins(env, env_ids)

    for j in range(active_wall_count):
        x, y, sx, sy = wall_segments[j]

        wall_pose[:, j, 0] = x + origins[:, 0]
        wall_pose[:, j, 1] = y + origins[:, 1]
        wall_pose[:, j, 2] = 0.25
        wall_pose[:, j, 3] = 1.0

        env.navrl_wall_xy[env_ids, j, 0] = x
        env.navrl_wall_xy[env_ids, j, 1] = y

    _write_collection_pose(walls, wall_pose, env_ids)

    env.navrl_active_wall_count = active_wall_count

    # ----------------------------
    # Static obstacles
    # ----------------------------
    static_pose = torch.zeros(len(env_ids), max_static, 7, device=env.device)
    static_pose[..., 2] = HIDDEN_Z
    static_pose[..., 3] = 1.0

    env.navrl_static_xy[env_ids] = 999.0

    active_static_count = min(num_static_obstacles, len(static_obstacles), max_static)

    for j in range(active_static_count):
        x, y, sx, sy = static_obstacles[j]

        static_pose[:, j, 0] = x + origins[:, 0]
        static_pose[:, j, 1] = y + origins[:, 1]
        static_pose[:, j, 2] = 0.225
        static_pose[:, j, 3] = 1.0

        env.navrl_static_xy[env_ids, j, 0] = x
        env.navrl_static_xy[env_ids, j, 1] = y

    _write_collection_pose(static_obs, static_pose, env_ids)

    env.navrl_active_static_count = active_static_count

    # ----------------------------
    # Dynamic obstacles
    # ----------------------------
    dynamic_pose = torch.zeros(len(env_ids), max_dynamic, 7, device=env.device)
    dynamic_pose[..., 2] = HIDDEN_Z
    dynamic_pose[..., 3] = 1.0

    env.navrl_dynamic_xy[env_ids] = 999.0

    active_dynamic_count = min(num_dynamic_obstacles, len(dynamic_lanes), max_dynamic)

    if not hasattr(env, "navrl_dynamic_lane_start_xy"):
        env.navrl_dynamic_lane_start_xy = torch.zeros(env.num_envs, max_dynamic, 2, device=env.device)
    if not hasattr(env, "navrl_dynamic_lane_end_xy"):
        env.navrl_dynamic_lane_end_xy = torch.zeros(env.num_envs, max_dynamic, 2, device=env.device)
    if not hasattr(env, "navrl_dynamic_lane_direction"):
        env.navrl_dynamic_lane_direction = torch.ones(env.num_envs, max_dynamic, device=env.device)

    for j in range(active_dynamic_count):
        x1, y1, x2, y2 = dynamic_lanes[j]

        phase = torch.rand(len(env_ids), device=env.device)
        x = x1 + phase * (x2 - x1)
        y = y1 + phase * (y2 - y1)

        dynamic_pose[:, j, 0] = x + origins[:, 0]
        dynamic_pose[:, j, 1] = y + origins[:, 1]
        dynamic_pose[:, j, 2] = 0.325
        dynamic_pose[:, j, 3] = 1.0

        env.navrl_dynamic_xy[env_ids, j, 0] = x
        env.navrl_dynamic_xy[env_ids, j, 1] = y

        env.navrl_dynamic_lane_start_xy[env_ids, j, 0] = x1
        env.navrl_dynamic_lane_start_xy[env_ids, j, 1] = y1
        env.navrl_dynamic_lane_end_xy[env_ids, j, 0] = x2
        env.navrl_dynamic_lane_end_xy[env_ids, j, 1] = y2

        direction = torch.where(torch.rand(len(env_ids), device=env.device) > 0.5, 1.0, -1.0)
        env.navrl_dynamic_lane_direction[env_ids, j] = direction

    _write_collection_pose(dynamic_obs, dynamic_pose, env_ids)

    env.navrl_active_dynamic_count = active_dynamic_count

    # Store scenario start/goal for reset_path_command
    if not hasattr(env, "navrl_scenario_start_xy"):
        env.navrl_scenario_start_xy = torch.zeros(env.num_envs, 2, device=env.device)
    if not hasattr(env, "navrl_scenario_goal_xy"):
        env.navrl_scenario_goal_xy = torch.zeros(env.num_envs, 2, device=env.device)

    env.navrl_scenario_start_xy[env_ids, 0] = start[0]
    env.navrl_scenario_start_xy[env_ids, 1] = start[1]
    env.navrl_scenario_goal_xy[env_ids, 0] = goal[0]
    env.navrl_scenario_goal_xy[env_ids, 1] = goal[1]


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

    if hasattr(env, "navrl_scenario_start_xy") and hasattr(env, "navrl_scenario_goal_xy"):
        start = env.navrl_scenario_start_xy[env_ids]
        goal = env.navrl_scenario_goal_xy[env_ids]
    else:
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

    dynamic_obs = env.scene[asset_cfg.name]
    max_dynamic = dynamic_obs.num_objects
    origins = _env_origins(env, env_ids)

    dt = env.step_dt

    xy = env.navrl_dynamic_xy[env_ids, :n]
    start = env.navrl_dynamic_lane_start_xy[env_ids, :n]
    end = env.navrl_dynamic_lane_end_xy[env_ids, :n]
    direction_sign = env.navrl_dynamic_lane_direction[env_ids, :n]

    lane_vec = end - start
    lane_len = torch.norm(lane_vec, dim=-1).clamp_min(1e-6)
    lane_dir = lane_vec / lane_len.unsqueeze(-1)

    speed = 0.65

    xy = xy + lane_dir * direction_sign.unsqueeze(-1) * speed * dt

    from_start = xy - start
    progress = torch.sum(from_start * lane_dir, dim=-1)

    hit_start = progress < 0.0
    hit_end = progress > lane_len
    hit = hit_start | hit_end

    direction_sign = torch.where(hit, -direction_sign, direction_sign)

    # Correct tensor-safe clamp:
    progress = torch.maximum(progress, torch.zeros_like(progress))
    progress = torch.minimum(progress, lane_len)

    xy = start + lane_dir * progress.unsqueeze(-1)

    env.navrl_dynamic_xy[env_ids, :n] = xy
    env.navrl_dynamic_lane_direction[env_ids, :n] = direction_sign

    pose = torch.zeros(len(env_ids), max_dynamic, 7, device=env.device)
    pose[..., 2] = HIDDEN_Z
    pose[..., 3] = 1.0

    pose[:, :n, 0:2] = xy + origins[:, None, :]
    pose[:, :n, 2] = 0.325
    pose[:, :n, 3] = 1.0

    _write_collection_pose(dynamic_obs, pose, env_ids)
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


def _get_scenario_layout(name: str):
    """Return wall segments, static obstacles, dynamic obstacle spawn lines, start, goal.

    Coordinates are local to each environment.

    wall_segments:
        list of (x, y, sx, sy)

    static_obstacles:
        list of (x, y, sx, sy)

    dynamic_lanes:
        list of (x1, y1, x2, y2)
        dynamic obstacles move along/cross these lanes.

    start:
        (x, y)

    goal:
        (x, y)
    """

    if name == "narrow_corridor_crossing":
        # Long corridor where moving obstacles cross the robot path.
        # Classical controllers often stop until the crossing object fully clears.
        wall_segments = [
            (0.0,  1.25, 10.0, 0.25),   # upper corridor wall
            (0.0, -1.25, 10.0, 0.25),   # lower corridor wall
            (-5.0, 0.0, 0.25, 2.75),    # left boundary
            (5.0,  0.0, 0.25, 2.75),    # right boundary
        ]

        static_obstacles = [
            (-1.4, 0.65, 0.45, 0.45),
            (1.6, -0.65, 0.45, 0.45),
        ]

        dynamic_lanes = [
            (-2.0, -1.0, -2.0, 1.0),
            (0.0, 1.0, 0.0, -1.0),
            (2.0, -1.0, 2.0, 1.0),
        ]

        start = (-4.3, 0.0)
        goal = (4.3, 0.0)

    elif name == "doorway_human_crossing":
        # Doorway bottleneck. Robot must time the doorway crossing.
        # Classical controllers usually freeze at the doorway.
        wall_segments = [
            (-2.5, 1.5, 5.0, 0.25),
            (-2.5, -1.5, 5.0, 0.25),
            (2.5, 1.5, 5.0, 0.25),
            (2.5, -1.5, 5.0, 0.25),

            (0.0, 1.05, 0.25, 0.90),    # top doorway wall piece
            (0.0, -1.05, 0.25, 0.90),   # bottom doorway wall piece
        ]

        static_obstacles = [
            (-3.2, 0.7, 0.55, 0.55),
            (3.2, -0.7, 0.55, 0.55),
        ]

        dynamic_lanes = [
            (0.0, -1.3, 0.0, 1.3),      # human crosses doorway
            (-0.5, 1.2, 0.5, -1.2),
        ]

        start = (-4.2, 0.0)
        goal = (4.2, 0.0)

    elif name == "blind_corner_intercept":
        # L-shaped turn. Dynamic obstacle appears from blind side.
        # Tests anticipation, not just reactive stopping.
        wall_segments = [
            (-2.5, 1.25, 5.0, 0.25),
            (-2.5, -1.25, 5.0, 0.25),
            (0.0, -1.25, 0.25, 3.0),
            (1.5, 0.25, 3.0, 0.25),
            (1.5, 2.25, 3.0, 0.25),
            (3.0, 1.25, 0.25, 2.0),
        ]

        static_obstacles = [
            (-1.0, 0.65, 0.45, 0.45),
            (1.4, 1.6, 0.45, 0.45),
        ]

        dynamic_lanes = [
            (1.8, 2.0, 1.8, 0.4),       # obstacle comes from blind branch
            (2.6, 0.5, 0.6, 0.5),
        ]

        start = (-4.2, 0.0)
        goal = (2.4, 1.5)

    elif name == "static_dynamic":
        # Static clutter plus moving obstacles.
        # Tests whether RL can keep moving instead of conservative stop-and-go.
        wall_segments = [
            (0.0,  2.0, 10.0, 0.25),
            (0.0, -2.0, 10.0, 0.25),
            (-5.0, 0.0, 0.25, 4.0),
            (5.0,  0.0, 0.25, 4.0),
        ]

        static_obstacles = [
            (-3.0, 0.7, 0.55, 0.55),
            (-1.6, -0.7, 0.55, 0.55),
            (-0.2, 0.7, 0.55, 0.55),
            (1.2, -0.7, 0.55, 0.55),
            (2.6, 0.7, 0.55, 0.55),
        ]

        dynamic_lanes = [
            (-2.3, -1.6, -2.3, 1.6),
            (0.5, 1.6, 0.5, -1.6),
            (3.2, -1.6, 3.2, 1.6),
        ]

        start = (-4.4, 0.0)
        goal = (4.4, 0.0)

    elif name == "crossing_stream":
        # Multiple moving obstacles crossing the path at different phases.
        # This directly targets dynamic obstacle benchmarking.
        wall_segments = [
            (0.0,  1.8, 10.0, 0.25),
            (0.0, -1.8, 10.0, 0.25),
            (-5.0, 0.0, 0.25, 3.6),
            (5.0,  0.0, 0.25, 3.6),
        ]

        static_obstacles = [
            (-3.5, -0.9, 0.45, 0.45),
            (3.5, 0.9, 0.45, 0.45),
        ]

        dynamic_lanes = [
            (-3.0, -1.5, -3.0, 1.5),
            (-1.5, 1.5, -1.5, -1.5),
            (0.0, -1.5, 0.0, 1.5),
            (1.5, 1.5, 1.5, -1.5),
            (3.0, -1.5, 3.0, 1.5),
        ]

        start = (-4.4, 0.0)
        goal = (4.4, 0.0)

    else:
        raise ValueError(f"Unknown scenario: {name}")

    return wall_segments, static_obstacles, dynamic_lanes, start, goal
