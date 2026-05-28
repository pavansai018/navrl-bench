# NavRL Bench
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**RL-Based Dynamic Obstacle Avoidance for ROSMASTER M3 Navigation**

NavRL Bench is a robotics project focused on goal-directed navigation in dynamic environments. The project trains an RL-based local controller for **ROSMASTER M3** using **IsaacLab / Isaac Sim** and benchmarks the trained policy against classical **Nav2** local controllers such as **DWB**, **MPPI**, and **Regulated Pure Pursuit**.

The final goal is simple:

> Given a goal, the robot should avoid fast-moving obstacles and reach the goal safely.

---

## Project Motivation

Classical Nav2 controllers are reliable and widely used for mobile robot navigation. However, when fast-moving obstacles repeatedly cross the robot path, they can become conservative, stop frequently, or struggle to maintain smooth progress toward the goal.

NavRL Bench investigates whether a reinforcement-learning-based local controller trained from scratch can produce more adaptive dynamic-obstacle avoidance behavior while still reaching the assigned goal safely.

The project does not assume that RL will always perform better. The goal is to measure where classical controllers perform well, where they struggle, and whether the trained RL policy provides better adaptability in fast-moving dynamic-obstacle scenarios.

---

## Core Idea

The global planner remains unchanged. Only the local controller block is compared.

```text
Map + Goal
   ↓
Nav2 Global Planner
   ↓
Global Path
   ↓
Local Controller
   ├── DWB
   ├── MPPI
   ├── Regulated Pure Pursuit
   └── Trained RL Policy
   ↓
/cmd_vel
   ↓
ROSMASTER M3
```

---

## Project Objectives

- Train an RL-based local navigation policy from scratch.
- Use IsaacLab / Isaac Sim for simulation-based RL training.
- Integrate the trained policy into a ROS2 / Nav2 local-control pipeline.
- Benchmark the RL policy against classical Nav2 controllers.
- Focus on fast-moving dynamic-obstacle avoidance.
- Evaluate all controllers under controlled and repeatable test conditions.

---

## Robot Platform

The project uses **ROSMASTER M3** as the mobile robot platform for navigation, obstacle avoidance, benchmarking, and sim-to-real testing.

---

## Tech Stack

| Component | Tool / Framework |
|---|---|
| Robot Platform | ROSMASTER M3 |
| Middleware | ROS 2 |
| Navigation Stack | Nav2 |
| Classical Controllers | DWB, MPPI, Regulated Pure Pursuit |
| RL Training | IsaacLab / Isaac Sim |
| RL Algorithm | PPO |
| Simulation / Visualization | Isaac Sim, Gazebo, RViz |
| Programming | Python |

---

## Controllers Compared

### DWB Controller

A classical Nav2 local controller based on the Dynamic Window Approach. It samples possible velocity commands and evaluates them using critic functions.

### MPPI Controller

A sampling-based predictive controller that evaluates candidate trajectories and selects the best control action based on cost.

### Regulated Pure Pursuit

A path-tracking controller that follows the global path while regulating speed based on curvature and collision constraints.

### Trained RL Policy

A reinforcement-learning-based local controller trained from scratch in IsaacLab / Isaac Sim. The policy outputs velocity commands compatible with the ROS2 navigation pipeline.

---

## RL Training Pipeline

```text
Create ROSMASTER M3 training environment
   ↓
Define observation space
   ↓
Define velocity action space
   ↓
Design reward function
   ↓
Train PPO policy from scratch
   ↓
Validate policy in simulation
   ↓
Export trained policy
   ↓
Integrate into ROS2 / Nav2 local-control pipeline
```

---

## Planned RL Observation Space

The RL policy is expected to use local navigation information such as:

- Goal direction relative to the robot
- Goal distance
- Robot linear velocity
- Robot angular velocity
- Laser scan or compact local obstacle representation
- Previous action for smoother behavior
- Optional local path information from Nav2

---

## Planned RL Action Space

The policy outputs velocity commands:

```text
linear velocity
angular velocity
```

These commands are constrained by ROSMASTER M3 safety limits and converted into valid `/cmd_vel` messages.

---

## Reward Design

The reward function is designed to encourage safe and useful navigation behavior.

### Positive Rewards

- Progress toward the goal
- Reaching the goal
- Maintaining useful forward motion
- Safe local obstacle avoidance

### Penalties

- Collision with obstacles
- Getting too close to obstacles
- Unnecessary stopping
- Moving away from the goal
- Jerky or unstable velocity commands

The reward design will be refined through training experiments.

---

## Benchmark Scenarios

### 1. Fast Dynamic Obstacle Crossing

A moving obstacle crosses the robot path at higher speed while the robot attempts to reach the goal.

### 2. Temporary Path Blockage

A moving obstacle briefly blocks the route, forcing the robot to slow down, wait, or locally avoid.

### 3. Goal Reaching with Moving Obstacles

The robot must reach the assigned goal while reacting to moving obstacles along the route.

### 4. Static Obstacle Navigation

A baseline scenario to verify normal local navigation behavior around fixed obstacles.

### 5. Narrow Passage Navigation

The robot navigates through a narrow passage where local control decisions are important.

### 6. Repeated Controlled Trials

Each controller is tested multiple times under the same scenario setup for fair comparison.

---

## Evaluation Metrics

Controller performance will be evaluated using:

- Success rate
- Time to goal
- Path length
- Number of stops
- Collision count
- Minimum obstacle distance
- Velocity smoothness
- Dynamic obstacle clearance

---

## Benchmark Principle

The benchmark follows one main rule:

> Same robot, same map, same start-goal pairs, same dynamic-obstacle setup, same robot limits.

Only the local controller changes.

This keeps the comparison fair between:

```text
DWB
MPPI
Regulated Pure Pursuit
RL Policy
```

---

## Sim-to-Real Focus

Training an RL policy in simulation is only the first step. The major technical challenge is transferring that policy safely and reliably to the real ROSMASTER M3.

Key sim-to-real concerns include:

- Sensor noise
- Wheel slip
- Command delay
- Odometry drift
- Velocity and acceleration limits
- Observation mismatch between simulation and ROS2
- Safe `/cmd_vel` output filtering

The project will first validate the policy in simulation and then move toward real-robot testing.

---

## Repository Structure

```text
navrl-bench/
│
├── docs/
│   ├── index.html
│   ├── rl-training.html
│   ├── style.css
│   └── script.js
│
├── .github/
│   └── workflows/
│       └── pr-check.yml
│
├── training/
│   └── README.md
│
├── nav2_configs/
│   └── README.md
│
├── ros2_nodes/
│   └── README.md
│
├── results/
│   └── README.md
│
└── README.md
```

The structure may evolve as the project implementation progresses.

---

## Project Website


<a href="https://pavansai018.github.io/navrl-bench" target="_blank">https://pavansai018.github.io/navrl-bench</a>

---

## Source Repository


<a href="https://github.com/pavansai018/navrl-bench" target="_blank">https://github.com/pavansai018/navrl-bench</a>


---

## Current Status

This project is under active development.

Current focus:

- Finalizing website and documentation
- Setting up ROSMASTER M3 project structure
- Preparing Nav2 baseline configuration
- Designing IsaacLab RL training workflow
- Defining benchmark scenarios for dynamic-obstacle navigation

---

## Team

**Team Name:** Shift+Delete

**Contributors:**

- Pavan Sai
- Siva

---

## References

```
@InProceedings{macenski2020marathon2,
author = {Macenski, Steven and Martin, Francisco and White, Ruffin and Ginés Clavero, Jonatan},
title = {The Marathon 2: A Navigation System},
booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
year = {2020}
}
```
```
@article{macenski2023survey,
      title={From the desks of ROS maintainers: A survey of modern & capable mobile robotics algorithms in the robot operating system 2},
      author={S. Macenski, T. Moore, DV Lu, A. Merzlyakov, M. Ferguson},
      year={2023},
      journal = {Robotics and Autonomous Systems}
}
```
```
@article{macenski2023regulated,
      title={Regulated Pure Pursuit for Robot Path Tracking},
      author={Steve Macenski and Shrijit Singh and Francisco Martin and Jonatan Gines},
      year={2023},
      journal = {Autonomous Robots}
}
```

```
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```
```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```
---

## License

This project is licensed under the BSD 3-Clause License – see the [LICENSE](LICENSE) file for details.
