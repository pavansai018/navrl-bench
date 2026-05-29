# Dynamic Obstacle Avoidance

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Manager-Based IsaacLab RL Training for ROSMASTER M3 Dynamic Obstacle Avoidance**

This repository contains an IsaacLab extension for training a reinforcement-learning-based local navigation policy for **ROSMASTER M3**. The project focuses on goal-directed navigation in environments with static obstacles and fast-moving dynamic obstacles.

The target behavior is simple:

> Given a goal, the robot should avoid fast-moving obstacles and reach the goal safely.

The project uses the **Manager-Based IsaacLab workflow** and trains a PPO-based policy that can later be integrated with a ROS 2 / Nav2 local-control pipeline for benchmarking against classical controllers such as DWB, MPPI, and Regulated Pure Pursuit.

---

## Project Requirement

The environment should support training a mobile robot policy for dynamic obstacle avoidance.

The training world should include:

- ROSMASTER M3 robot model or a compatible proxy during early development
- Goal-directed navigation task
- Static map layout generated inside IsaacLab
- Static obstacles such as walls, boxes, and narrow passages
- Dynamic obstacles moving with randomized speed and direction
- Reset randomization for obstacle placement, goal position, and dynamic obstacle motion
- Continuous velocity command output suitable for `/cmd_vel`
- Reward terms for goal progress, collision avoidance, smooth motion, and dynamic obstacle clearance

---

## Main Training Objective

The RL policy should learn to:

- Move toward a given goal
- Avoid static obstacles
- Avoid fast-moving dynamic obstacles
- Reduce unnecessary stopping
- Maintain safe obstacle distance
- Reach the goal without collision
- Produce smooth velocity commands

---

## Training Approach

The project uses **PPO** as the primary reinforcement learning algorithm.

PPO is selected because:

- It is well supported in IsaacLab through RSL-RL
- It is stable for continuous-control tasks
- It works well with parallel simulation environments
- It is easier to debug than many off-policy algorithms
- It is suitable for early-stage sim-to-real robotics experiments

---

## Environment Design

The IsaacLab environment should be generated at runtime using manager-based configuration.

The planned environment contains:

```text
Robot
  ↓
Static map layout
  ↓
Static obstacles
  ↓
Dynamic obstacles
  ↓
Goal marker
  ↓
Observation, reward, termination, and reset managers
```

The environment should be built using reusable objects such as wall blocks, obstacle objects, and moving obstacle actors. Map layouts can be generated procedurally or selected from predefined templates.

---

## Planned Map Templates

The project may use the following training and benchmark map layouts:

### 1. Corridor Crossing

A straight corridor where a dynamic obstacle crosses the robot path.

### 2. Doorway Blockage

Two rooms connected by a doorway where a moving obstacle temporarily blocks the doorway.

### 3. T-Junction Crossing

A T-shaped intersection where dynamic obstacles cross the robot path near the junction.

### 4. Cluttered Indoor Navigation

An indoor space with static obstacles and moving obstacles in the free space.

### 5. Open Arena Crossing

An open area where multiple moving obstacles cross between the robot and the goal.

---

## Planned Observation Space

The policy observation may include:

- Goal distance
- Goal direction relative to the robot
- Robot linear velocity
- Robot angular velocity
- Laser scan or ray-caster obstacle distances
- Previous action
- Optional local path direction
- Optional nearest dynamic obstacle information

---

## Planned Action Space

The policy action is planned as:

```text
linear velocity
angular velocity
```

The action output should be limited by ROSMASTER M3 safety limits before being published as a valid velocity command.

---

## Reward Design

The reward function should encourage safe and efficient navigation.

### Positive Rewards

- Progress toward the goal
- Reaching the goal
- Maintaining useful forward motion
- Safe obstacle clearance

### Penalties

- Collision with static obstacles
- Collision with dynamic obstacles
- Getting too close to obstacles
- Excessive stopping
- Moving away from the goal
- Jerky or unstable velocity commands
- Timeout before reaching the goal

---

## Benchmark Direction

After training, the learned policy will be evaluated against classical Nav2 local controllers under the same test conditions.

Controllers planned for comparison:

- DWB
- MPPI
- Regulated Pure Pursuit
- PPO-trained RL policy

The benchmark principle is:

> Same robot, same map, same goal, same obstacle setup, same robot limits. Only the local controller changes.

---

## Evaluation Metrics

The project will evaluate controller performance using:

- Success rate
- Time to goal
- Path length
- Number of stops
- Collision count
- Minimum obstacle distance
- Dynamic obstacle clearance
- Velocity smoothness

---

## Sim-to-Real Focus

Training in simulation is only the first stage. The major technical challenge is transferring the learned policy safely to the real ROSMASTER M3.

Important sim-to-real concerns include:

- Sensor noise
- Odometry drift
- Wheel slip
- Command delay
- Velocity and acceleration limits
- Observation mismatch between simulation and ROS 2
- Safe filtering of velocity commands

---

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/dynamic_obstacle_avoidance
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```

        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

---

## Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file `.python.env` in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

---

## Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

---

## Code Formatting

We have a pre-commit template to automatically format your code.

To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

---

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/dynamic_obstacle_avoidance"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`.

Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*",
"<path-to-isaac-sim>/extscache/omni.kit.*",
"<path-to-isaac-sim>/extscache/omni.graph.*",
"<path-to-isaac-sim>/extscache/omni.services.*"
```

---

## Project Status

This repository is currently in the initial setup stage.

Immediate focus:

- Create the manager-based IsaacLab task
- Define the ROSMASTER M3 robot asset or proxy model
- Generate runtime training maps
- Add static and dynamic obstacles
- Define PPO observation, action, reward, and termination terms
- Validate the environment with zero-action and random-action agents
- Start PPO training after the environment is stable

---

