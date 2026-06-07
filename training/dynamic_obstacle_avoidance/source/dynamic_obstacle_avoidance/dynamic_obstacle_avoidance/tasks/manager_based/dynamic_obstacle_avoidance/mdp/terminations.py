from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy, map_collision_flag, dynamic_obstacle_collision_flag


def final_goal_reached(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30,) -> torch.Tensor:
    robot_xy = _robot_xy(env, asset_cfg.name)

    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    distance = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)

    return distance < threshold

def map_collision_termination(env, asset_cfg: SceneEntityCfg, radius: float = 0.22,) -> torch.Tensor:
    return map_collision_flag(env, radius=radius, num_points=16,)

def dynamic_obstacle_collision_termination(env, asset_cfg: SceneEntityCfg, robot_radius: float = 0.22) -> torch.Tensor:
    return dynamic_obstacle_collision_flag(env, robot_radius=robot_radius)

def stuck_termination(
    env,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float = 0.02,
    time_window_s: float = 2.0,
    grace_period_s: float = 2.0,
) -> torch.Tensor:
    """
    Terminate the episode if the robot has been moving below speed_threshold
    for longer than time_window_s, and is not near the final goal.

    Why this helps:
        Without this, the robot can learn that "waiting for obstacles to pass"
        is a valid strategy. This is not ideal because:
            1. It wastes time (episode_length_s is finite)
            2. It does not generalize to dense obstacle scenarios
               where there is never a clear moment to pass

        By terminating "stuck" episodes, the PPO algorithm learns that
        "waiting" is always a losing strategy and must find a lateral bypass.

    grace_period_s:
        Do not apply stuck detection for the first N seconds of the episode.
        This prevents false positives at episode start when the robot is
        being placed at the path start position.

    Buffers:
        env.navrl_stuck_timer: [num_envs] float, seconds spent below threshold
        env.navrl_episode_time: [num_envs] float, total time in current episode
    """
    robot = env.scene[asset_cfg.name]

    # Get physics dt and decimation to compute step time in seconds
    # step_dt is the time elapsed per RL step (physics_dt * decimation)
    dt = float(getattr(env, "step_dt", env.physics_dt * env.cfg.decimation))

    # Initialize buffers on first call
    if not hasattr(env, "navrl_stuck_timer"):
        env.navrl_stuck_timer = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "navrl_episode_time"):
        env.navrl_episode_time = torch.zeros(env.num_envs, device=env.device)

    # Advance episode time
    env.navrl_episode_time += dt

    # Compute current speed
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)

    # Increment stuck timer for slow envs, reset for moving envs
    is_slow = speed < speed_threshold
    env.navrl_stuck_timer = torch.where(
        is_slow,
        env.navrl_stuck_timer + dt,
        torch.zeros_like(env.navrl_stuck_timer),
    )

    # Check if stuck for long enough
    stuck = env.navrl_stuck_timer > time_window_s

    # Do not apply during grace period
    past_grace = env.navrl_episode_time > grace_period_s
    stuck = stuck & past_grace

    # Do not apply if near the final goal
    if hasattr(env, "navrl_final_goal_xy"):
        robot_xy = _robot_xy(env, asset_cfg.name)
        dist_to_goal = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
        not_at_goal = dist_to_goal > 0.5
        stuck = stuck & not_at_goal

    return stuck


def reset_stuck_buffers(
    env,
    env_ids: torch.Tensor,
):
    """
    Reset stuck detection buffers on episode reset.
    Call this as an Event in mode='reset'.
    """
    if hasattr(env, "navrl_stuck_timer"):
        env.navrl_stuck_timer[env_ids] = 0.0

    if hasattr(env, "navrl_episode_time"):
        env.navrl_episode_time[env_ids] = 0.0