from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .rewards import dynamic_obstacle_collision_penalty, map_collision_penalty


def dynamic_collision(env, asset_cfg: SceneEntityCfg, robot_radius: float = 0.22):
    return dynamic_obstacle_collision_penalty(env, asset_cfg, robot_radius=robot_radius).bool()


def map_collision(env, asset_cfg: SceneEntityCfg, radius: float = 0.22):
    return map_collision_penalty(env, asset_cfg, radius=radius).bool()


def final_goal_reached(env, asset_cfg: SceneEntityCfg, threshold: float = 0.30):
    if not hasattr(env, "navrl_final_goal_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    dist = torch.norm(env.navrl_final_goal_xy - robot_xy, dim=-1)
    return dist < threshold

def stuck(
    env,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float = 0.05,
    time_window_s: float = 1.0,
    grace_period_s: float = 2.0,
):
    robot = env.scene[asset_cfg.name]

    speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)

    dt = float(env.step_dt)
    window_steps = max(1, int(time_window_s / dt))
    grace_steps = max(1, int(grace_period_s / dt))

    if not hasattr(env, "stuck_counter"):
        env.stuck_counter = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    moving = speed > speed_threshold

    env.stuck_counter = torch.where(
        moving,
        torch.zeros_like(env.stuck_counter),
        env.stuck_counter + 1,
    )

    in_grace = env.episode_length_buf < grace_steps

    return (env.stuck_counter >= window_steps) & (~in_grace)