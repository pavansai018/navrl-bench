import argparse
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Dynamic-Obstacle-Avoidance-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=6000)

# Force curriculum level so you can test specific obstacle stages.
# 0 = no dynamic obstacles
# 1 = first obstacle stage
# 2 = second obstacle stage
# etc.
parser.add_argument("--fixed_level", type=int, default=1)

# Teacher MPPI params for testing.
parser.add_argument("--mppi_samples", type=int, default=96)
parser.add_argument("--mppi_horizon", type=int, default=20)
parser.add_argument("--mppi_chunk", type=int, default=64)

parser.add_argument("--debug_draw", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

import dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance  # noqa: F401

from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp.teacher_mppi import (
    compute_mppi_teacher_action,
)
from dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance.mdp.observations import (
    _robot_xy,
    map_collision_flag,
    dynamic_obstacle_collision_flag,
)


def normalize_teacher_action(env, teacher_action_physical: torch.Tensor) -> torch.Tensor:
    """Teacher returns physical [vx, vy, wz].
    env.step expects normalized action [-1, 1].
    """
    action_cfg = env.unwrapped.cfg.actions.base_velocity

    scale = torch.tensor(
        [
            float(action_cfg.max_vx),
            float(action_cfg.max_vy),
            float(action_cfg.max_wz),
        ],
        device=teacher_action_physical.device,
        dtype=teacher_action_physical.dtype,
    )

    return torch.clamp(teacher_action_physical / scale.clamp_min(1.0e-6), -1.0, 1.0)


def print_env_state(env, step_i: int, teacher_physical: torch.Tensor, teacher_normalized: torch.Tensor):
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    robot_xy = _robot_xy(unwrapped)
    root_pos = robot.data.root_pos_w[:, :3]

    processed = unwrapped.action_manager._terms["base_velocity"].processed_actions

    if hasattr(unwrapped, "navrl_final_goal_xy"):
        goal_dist = torch.norm(unwrapped.navrl_final_goal_xy - robot_xy, dim=-1)
    else:
        goal_dist = torch.zeros(unwrapped.num_envs, device=unwrapped.device)

    map_col = map_collision_flag(unwrapped).float()

    if hasattr(unwrapped, "dyn_obs_xy"):
        dyn_col = dynamic_obstacle_collision_flag(unwrapped).float()
    else:
        dyn_col = torch.zeros(unwrapped.num_envs, device=unwrapped.device)

    if hasattr(unwrapped, "navrl_path_blocked"):
        path_blocked = unwrapped.navrl_path_blocked.float()
    else:
        path_blocked = torch.zeros(unwrapped.num_envs, device=unwrapped.device)

    env_id = 0

    print(
        f"step={step_i:04d} | "
        f"pos=({root_pos[env_id,0].item(): .3f}, {root_pos[env_id,1].item(): .3f}) | "
        f"goal_dist={goal_dist[env_id].item(): .3f} | "
        f"teacher_phys=[{teacher_physical[env_id,0].item(): .3f}, "
        f"{teacher_physical[env_id,1].item(): .3f}, "
        f"{teacher_physical[env_id,2].item(): .3f}] | "
        f"teacher_norm=[{teacher_normalized[env_id,0].item(): .3f}, "
        f"{teacher_normalized[env_id,1].item(): .3f}, "
        f"{teacher_normalized[env_id,2].item(): .3f}] | "
        f"processed=[{processed[env_id,0].item(): .3f}, "
        f"{processed[env_id,1].item(): .3f}, "
        f"{processed[env_id,2].item(): .3f}] | "
        f"path_blocked={path_blocked[env_id].item():.0f} | "
        f"map_col={map_col[env_id].item():.0f} | "
        f"dyn_col={dyn_col[env_id].item():.0f}"
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env_cfg.fixed_curriculum_level = args_cli.fixed_level

    env_cfg.debug_draw_nav2 = args_cli.debug_draw
    env_cfg.debug_draw_path = args_cli.debug_draw
    env_cfg.debug_draw_lidar = args_cli.debug_draw
    env_cfg.debug_draw_dynamic_obstacles = args_cli.debug_draw

    env_cfg.mppi_teacher["num_samples"] = args_cli.mppi_samples
    env_cfg.mppi_teacher["horizon"] = args_cli.mppi_horizon
    env_cfg.mppi_teacher["env_chunk_size"] = args_cli.mppi_chunk
    env_cfg.mppi_teacher["recompute_interval"] = 1

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    print("\n========== TORCH MPPI TEACHER TEST ==========")
    print(f"task                 : {args_cli.task}")
    print(f"num_envs             : {args_cli.num_envs}")
    print(f"fixed_curriculum_level: {args_cli.fixed_level}")
    print(f"mppi_samples          : {args_cli.mppi_samples}")
    print(f"mppi_horizon          : {args_cli.mppi_horizon}")
    print("=============================================\n")

    for i in range(args_cli.steps):
        teacher_physical = compute_mppi_teacher_action(env.unwrapped)
        teacher_normalized = normalize_teacher_action(env, teacher_physical)

        obs, rew, terminated, truncated, info = env.step(teacher_normalized)

        if i % 10 == 0:
            print_env_state(env, i, teacher_physical, teacher_normalized)

        done = terminated | truncated
        if torch.any(done):
            print(f"\nReset triggered at step {i}. done_envs={torch.nonzero(done).flatten().tolist()}")
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()