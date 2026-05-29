import argparse
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="M3-Flat-Movement-Test-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import dynamic_obstacle_avoidance.tasks.manager_based.dynamic_obstacle_avoidance # type: ignore

def run_action(env, action, steps, label):
    print(f"\n========== {label} ==========")

    action_tensor = torch.tensor(action, device=env.unwrapped.device, dtype=torch.float32).repeat(
        env.unwrapped.num_envs, 1
    )

    for i in range(steps):
        obs, rew, terminated, truncated, info = env.step(action_tensor)

        robot = env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w[0, :3].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()

        if i % 30 == 0:
            print(
                f"step={i:04d} | "
                f"x={pos[0]: .3f}, y={pos[1]: .3f}, z={pos[2]: .3f} | "
                f"quat={quat}"
            )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # action = [vx, vy, wz], normalized [-1, 1]

    # run_action(env, [1.0, 0.0, 0.0], 180, "FORWARD TEST: action [1, 0, 0]")
    # env.reset()

    # run_action(env, [-1.0, 0.0, 0.0], 180, "BACKWARD TEST: action [-1, 0, 0]")
    # env.reset()

    # run_action(env, [-0.0, 0.0, 0.0], 180, "STAY STILL: action [0, 0, 0]")
    # env.reset()

    # run_action(env, [0.0, 1.0, 0.0], 180, "LEFT/RIGHT STRAFE TEST: action [0, 1, 0]")
    # env.reset()

    # run_action(env, [0.0, -1.0, 0.0], 180, "OPPOSITE STRAFE TEST: action [0, -1, 0]")
    # env.reset()

    # run_action(env, [0.0, 0.0, 1.0], 180, "ROTATION TEST: action [0, 0, 1]")
    # env.reset()

    # run_action(env, [0.0, 0.0, -1.0], 180, "OPPOSITE ROTATION TEST: action [0, 0, -1]")

    # env.close()
    # tests = [
    #     ([1.0, 0.0, 0.0, 0.0], "lwheel1 only +"),
    #     ([-1.0, 0.0, 0.0, 0.0], "lwheel1 only -"),

    #     ([0.0, 1.0, 0.0, 0.0], "lwheel2 only +"),
    #     ([0.0, -1.0, 0.0, 0.0], "lwheel2 only -"),

    #     ([0.0, 0.0, 1.0, 0.0], "rwheel1 only +"),
    #     ([0.0, 0.0, -1.0, 0.0], "rwheel1 only -"),

    #     ([0.0, 0.0, 0.0, 1.0], "rwheel2 only +"),
    #     ([0.0, 0.0, 0.0, -1.0], "rwheel2 only -"),

    #     ([1.0, 1.0, 1.0, 1.0], "all wheels +"),
    #     ([-1.0, -1.0, -1.0, -1.0], "all wheels -"),

    #     ([1.0, 1.0, -1.0, -1.0], "left + right -"),
    #     ([-1.0, -1.0, 1.0, 1.0], "left - right +"),

    #     ([1.0, -1.0, -1.0, 1.0], "diagonal pattern A"),
    #     ([-1.0, 1.0, 1.0, -1.0], "diagonal pattern B"),

    #     ([1.0, -1.0, 1.0, -1.0], "diagonal pattern C"),
    #     ([-1.0, 1.0, -1.0, 1.0], "diagonal pattern D"),
    # ]
    # tests = [
    #     ([1.0, 1.0, -1.0, -1.0], "FORWARD RAW WHEEL TEST"),
    #     ([-1.0, -1.0, 1.0, 1.0], "BACKWARD RAW WHEEL TEST"),
    #     ([0.0, 0.0, 0.0, 0.0], "ZERO RAW WHEEL TEST"),
    # ]
    tests = [
        ([1.0, 0.0, 0.0], "FORWARD BASE VELOCITY TEST"),
        ([-1.0, 0.0, 0.0], "BACKWARD BASE VELOCITY TEST"),
        ([0.0, 0.0, 0.0], "ZERO BASE VELOCITY TEST"),
    ]

    tests = [
        ([1.0, 0.0, 0.0], "FORWARD vx +"),
        ([-1.0, 0.0, 0.0], "BACKWARD vx -"),

        ([0.0, 1.0, 0.0], "STRAFE vy +"),
        ([0.0, -1.0, 0.0], "STRAFE vy -"),

        ([0.0, 0.0, 1.0], "ROTATE wz +"),
        ([0.0, 0.0, -1.0], "ROTATE wz -"),

        ([0.0, 0.0, 0.0], "ZERO"),
    ]

    for action, label in tests:
        env.reset()
        run_action(env, action, 120, label)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()