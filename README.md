# NavRL Bench
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**RL-Based Dynamic Obstacle Avoidance for ROSMASTER M3 Navigation**

NavRL Bench trains a PPO policy with a **ScanHistoryTransformerActor** from scratch in IsaacLab / Isaac Sim and benchmarks the trained policy against classical Nav2 local controllers — **DWB**, **MPPI**, and **Regulated Pure Pursuit** — on the **ROSMASTER M3** mecanum-wheel robot.

> Given a goal, the robot should avoid fast-moving obstacles and reach the goal safely.

---
## Demo Videos

<table>
<tr>
<td align="center">

**Scenario 1**

<video src="https://github.com/user-attachments/assets/9a30fd73-8a11-4826-9b1c-b1f284a6e100" controls width="320"></video>

</td>
<td align="center">

**Scenario 2**

<video src="https://github.com/user-attachments/assets/d883cb59-62cb-4328-8aa7-2c36251ee528" controls width="320"></video>

</td>
<td align="center">

**Scenario 3**

<video src="https://github.com/user-attachments/assets/78cb38ee-0746-4bb8-9f2d-af424923a461" controls width="320"></video>

</td>
</tr>
</table>

---

## Project Motivation

Classical Nav2 controllers are reliable and widely used for mobile robot navigation. However, when fast-moving obstacles repeatedly cross the robot path, they can become conservative, stop frequently, or struggle to maintain smooth progress toward the goal.

ROSMASTER M3 is a mecanum-wheel platform capable of lateral (strafe) motion. Classical controllers rarely exploit lateral velocity for obstacle avoidance. An RL controller trained with a 3D velocity action space — linear x, linear y, and angular z — can learn adaptive lateral bypass maneuvers that classical baselines do not produce.

NavRL Bench investigates whether a trained scan-transformer policy can learn more adaptive dynamic-obstacle avoidance while still reaching the assigned goal safely.

---

## Core Idea

The global planner remains unchanged. Only the local controller block is replaced.

```text
Map + Goal
   ↓
Nav2 Global Planner
   ↓
Global Path → 8 lookahead waypoints extracted by RL bridge node
   ↓
Local Controller
   ├── DWB
   ├── MPPI
   ├── Regulated Pure Pursuit
   └── PPO + ScanHistoryTransformerActor  ← trained by this project
   ↓
/cmd_vel  →  [vx, vy, wz]
   ↓
ROSMASTER M3 (mecanum wheels)
```

---

## Tech Stack

| Component | Tool / Framework |
|---|---|
| Robot Platform | ROSMASTER M3 (mecanum wheels) |
| Middleware | ROS 2 |
| Navigation Stack | Nav2 |
| Classical Controllers | DWB, MPPI, Regulated Pure Pursuit |
| RL Training Simulator | IsaacLab / Isaac Sim |
| RL Algorithm | PPO (RSL-RL) |
| Policy Architecture | ScanHistoryTransformerActor (PyTorch TransformerEncoder) |
| Simulation / Visualization | Isaac Sim, Gazebo, RViz |
| Programming | Python, PyTorch |

---

## Policy Architecture — ScanHistoryTransformerActor

Instead of processing a single lidar snapshot, the actor treats each of the **144 lidar rays as a transformer token** whose value is an **8-frame temporal history**. This lets the network learn ray-wise closing motion — detecting which angular sectors contain approaching obstacles — without ever receiving explicit obstacle positions or velocities.

```text
Policy obs (1176-dim)
   ├── path [18]        8 Nav2 waypoints in robot frame + heading error + cross-track error
   ├── scan_flat [1152] 8 frames × 144 rays  →  reshape [B, 8, 144]
   └── motion [6]       vx, vy, wz + previous action

Per-ray feature extraction  (20 features per ray)
   8 normalized history values | 7 temporal deltas | current range
   min range over history | closing rate | sin(θ), cos(θ)
   ↓
ray_proj: Linear(20→128) → LayerNorm → GELU → Linear(128→128)
+ learned positional embedding [1, 144, 128]
   ↓
TransformerEncoder  (d_model=128, nhead=4, ff_dim=256, norm_first=True)
   ↓
Multi-pooling
   ├── attention pool [128]   learned importance weights over rays
   ├── min pool      [128]   worst-case obstacle sectors
   └── mean pool     [128]   global scene context

path_encoder:   Linear(18→128) → LN → GELU → Linear → [128]
motion_encoder: Linear(6→128)  → LN → GELU → Linear → [128]
   ↓
concat → [640]
   ↓
actor_head: Linear(640→256) → LN → GELU → Linear(256→128) → LN → GELU → Linear(128→3)
   ↓
Output: [vx, vy, wz]
```

The **critic** receives a privileged 1209-dim observation that includes ground-truth positions and velocities of the 4 nearest dynamic obstacles, enabling accurate value estimation during training. This information is never passed to the deployed policy.

---

## Observation Space

### Policy Observations — 1176 dimensions

| Term | Dim | Description |
|---|---|---|
| `local_path_window` | 16 | 8 future Nav2 waypoints in robot frame (x, y per point), normalized by 4 m |
| `nav2_heading_error` | 1 | Angular error between robot yaw and path tangent, wrapped to [−π, π] rad |
| `nav2_cross_track_error` | 1 | Signed lateral distance from robot to nearest path point, in metres |
| `scan_history` | 1152 | 8-frame lidar history × 144 rays; 360° scan, 10 m range. Flattened as [T, R] |
| `base_lin_vel` | 2 | Robot linear velocity (vx, vy) in m/s |
| `base_ang_vel` | 1 | Robot yaw rate wz in rad/s |
| `previous_action` | 3 | Last velocity command normalized to [−1, 1]: (vx, vy, wz) |
| **Total** | **1176** | |

### Additional Critic Observations — 33 extra dims → 1209 total

| Term | Dim | Description |
|---|---|---|
| `dynamic_obstacles` | 24 | 4 nearest obstacles × [x, y, vx, vy, radius, active] in robot frame |
| `path_blocked` | 1 | Binary: obstacle overlaps forward path corridor |
| `time_to_closest_approach` | 4 | 2 obstacles × [TCA, DCA] |
| `distance_to_goal` | 1 | Euclidean distance to final goal waypoint |
| `progress_fraction` | 1 | Fraction of Nav2 path completed [0, 1] |
| `map_collision` | 1 | Binary: robot footprint in static map |
| `dynamic_collision` | 1 | Binary: robot in contact with dynamic obstacle |
| **Total** | **1209** | |

---

## Action Space

The policy outputs a **3D mecanum velocity command**:

| Axis | Limit | Description |
|---|---|---|
| `vx` | ±0.5 m/s | Forward / backward |
| `vy` | ±0.5 m/s | Lateral strafe |
| `wz` | ±1.5 rad/s | Yaw rotation |

Actions are clipped to [−1, 1] by the actor head, scaled by the limits above, then processed through a rate limiter (smooth acceleration) and a static map action shield (blocks commands that would collide with known walls).

---

## Reward Design

The reward function has 20+ terms covering path following, obstacle avoidance, goal reaching, and behavioral shaping.

### Path Following

| Term | Weight | Description |
|---|---|---|
| `progress` | +35.0 | Arc-length progress along Nav2 path per step |
| `goal_approach` | +5.0 | Decrease in distance to final goal per step |
| `cross_track` | −4.0 | Lateral distance from path |
| `path_rejoin` | +15.0 | Reward returning to path when CTE > 0.2 m |
| `heading_alignment` | +8.0 | cos(path tangent − robot yaw), 4-step lookahead |
| `path_velocity` | +6.0 | Velocity projected onto path tangent |

### Dynamic Obstacle Avoidance

| Term | Weight | Description |
|---|---|---|
| `dynamic_collision` | −100.0 | Direct contact — terminates episode |
| `dynamic_clearance` | −20.0 | Within robot_radius + 0.25 m safety margin |
| `dynamic_ttc` | −15.0 | Time-to-collision below threshold within 3 s horizon |
| `dynamic_yield_no_side_space` | +8.0 | Reward slowing when both forward path and sides are blocked |
| `wall_aware_dynamic_bypass` | +20.0 | Reward lateral bypass when side wall clearance exists |
| `dynamic_forward_blocked` | −12.0 | Penalty for moving forward into blocked corridor |
| `dynamic_bad_lateral` | −8.0 | Penalty for strafing toward a wall with insufficient clearance |

### Static Map Avoidance

| Term | Weight | Description |
|---|---|---|
| `map_collision` | −150.0 | Footprint in occupancy map — terminates episode; heaviest penalty |
| `static_velocity_clearance` | −15.0 | Moving toward walls within 0.3 m in ±45° forward sector |

### Goal Reaching

| Term | Weight | Description |
|---|---|---|
| `final_goal` | +120.0 | Sparse bonus for reaching within 0.3 m of goal — terminates episode |

### Action Regularization and Behavioral Shaping

| Term | Weight | Description |
|---|---|---|
| `action_smoothness` | −0.06 | L2 norm of action change per step |
| `yaw_rate` | −0.05 | Absolute yaw rate — discourages spinning |
| `lateral_oscillation` | −0.35 | Sign flip in vy between steps — discourages side-to-side oscillation |
| `time` | −0.06 | Per-step time penalty |
| `no_wait` | −2.0 | Near-zero speed when path is not blocked — prevents waiting as a strategy |
| `start_speed` | −8.0 | Slow speed during first 1.5 s of episode — prevents hesitation at start |

---

## Training Pipeline

```text
Import ROSMASTER M3 URDF into IsaacLab (4 mecanum wheels, 0.035 m radius)
   ↓
Load real Nav2 path dataset — AWS warehouse map, up to 600 points per episode
   ↓
Define 1176-dim policy observation / 1209-dim privileged critic observation
   ↓
Define 3D mecanum velocity action space
   ↓
Train PPO with ScanHistoryTransformerActor — parallel envs, adaptive curriculum
   ↓
Adaptive curriculum: obstacle difficulty first, then domain randomization
   ↓
Validate policy in simulation
   ↓
Export checkpoint → load inside Nav2 RL local controller node
```

---

## Curriculum Design

The curriculum follows an **obstacles_first** strategy. The policy masters obstacle avoidance before sensor noise and actuator imperfections are introduced.

### Track 1 — Obstacle Difficulty

| Level | Scenario |
|---|---|
| L0 | Open field — pure goal-reaching |
| L1 | Tiny stationary obstacle on path (r 0.08–0.11 m) |
| L2 | Tiny stationary obstacle, lateral offset |
| L3 | Small stationary obstacle on path (r 0.10–0.14 m) |
| L4 | Small stationary obstacle, lateral offset |
| L5 | Medium stationary obstacle (r 0.14–0.18 m) |
| L6 | Slow crossing obstacle (speed 0.04–0.10 m/s, perpendicular to path) |

### Track 2 — Domain Randomization

| Stage | Randomization |
|---|---|
| DR0 | Lidar ray count degradation |
| DR1 | Gaussian scan noise |
| DR2 | Random ray dropout |
| DR3 | Action delay (1–3 step buffer) |
| DR4 | Per-wheel motor strength scaling |
| DR5–9 | Mass, CoM shift, wheel radius mismatch, wheel slip, combined |

### Promotion / Demotion

| Metric | Promote | Demote |
|---|---|---|
| Success Rate | > 70% | < 10% |
| Map Collision Rate | < 20% | > 50% |
| Dynamic Collision Rate | < 20% | > 50% |
| Timeout Rate | < 15% | > 50% |

---

## PPO Configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO (RSL-RL) |
| LR schedule | Adaptive (KL-based) |
| Target KL | 0.01 |
| Clip parameter ε | 0.2 |
| Discount factor γ | 0.99 |
| GAE λ | 0.95 |
| Max gradient norm | 1.0 |
| Clipped value loss | Yes |
| Entropy coefficient | 0.00 |

---

## Termination Conditions

| Condition | Outcome |
|---|---|
| Distance to goal < 0.3 m | Success |
| Episode time limit exceeded | Failure |
| Robot footprint in static map | Failure |
| Contact with dynamic obstacle | Failure |
| Speed < 0.02 m/s for > 2.0 s (not near goal) | Failure — stuck |

---

## Controllers Compared

### DWB Controller

A classical Nav2 local controller based on the Dynamic Window Approach. Samples possible velocity commands and evaluates them using configurable critic functions.

### MPPI Controller

A sampling-based predictive controller that evaluates thousands of candidate trajectories and selects the optimal control action by cost-weighted averaging.

### Regulated Pure Pursuit

A path-tracking controller that follows the global path while regulating speed based on curvature and collision constraints.

### PPO + ScanHistoryTransformerActor

A PPO policy with a transformer-based actor trained from scratch in IsaacLab. Outputs a 3D mecanum velocity command including lateral strafe, integrated into Nav2 via a custom local controller plugin node.

---

## Benchmark Scenarios

### 1. Fast Dynamic Obstacle Crossing

A moving obstacle crosses the robot path at higher speed while the robot attempts to reach the goal.

### 2. Temporary Path Blockage

A moving obstacle briefly blocks the route, forcing the robot to slow down, wait, or locally avoid.

### 3. Goal Reaching with Moving Obstacles

The robot must reach the assigned goal while reacting to multiple moving obstacles along the route.

### 4. Static Obstacle Navigation

A baseline scenario to verify normal local navigation behavior around fixed obstacles without dynamic elements.

### 5. Narrow Passage Navigation

The robot navigates through a narrow passage where precise lateral positioning and local control decisions matter.

### 6. Repeated Controlled Trials

Each controller is tested multiple times under the same scenario setup for statistically fair comparison.

---

## Evaluation Metrics

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

> Same robot, same map, same start-goal pairs, same dynamic-obstacle setup, same robot limits.

Only the local controller changes. Classical controllers use their standard Nav2 implementations. The RL policy uses the trained checkpoint produced by this project's training pipeline. All are evaluated under the same ROSMASTER M3 benchmark scenarios.

---

## Sim-to-Real

Training an RL policy in simulation is only the first step. Domain randomization during training covers:

- Lidar ray degradation, Gaussian scan noise, random ray dropout
- Action delay (1–3 step command buffer)
- Per-wheel motor strength scaling, wheel slip, wheel radius mismatch
- Robot mass variation, center-of-mass shift

The policy is validated in simulation before real-robot testing. Key sim-to-real concerns include odometry drift, command latency, and ensuring the 1176-dim observation can be assembled correctly from live ROS 2 topics.

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
├── training/
│   └── dynamic_obstacle_avoidance/
│       ├── source/   ← IsaacLab environment, MDP (obs, actions, rewards, curriculum)
│       └── scripts/  ← RSL-RL PPO config, ScanHistoryTransformerActor
│
├── .github/
│   └── workflows/
│       └── pr-check.yml
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

- RL training with ScanHistoryTransformerActor and adaptive curriculum
- Validating trained policy in simulation
- Preparing Nav2 RL local controller plugin for policy deployment
- Setting up benchmark evaluation pipeline

---

## Team

**Team Name:** Shift+Delete

**Contributors:**

- Pavan Sai
- Siva

---

## References

```bibtex
@InProceedings{macenski2020marathon2,
author = {Macenski, Steven and Martin, Francisco and White, Ruffin and Ginés Clavero, Jonatan},
title = {The Marathon 2: A Navigation System},
booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
year = {2020}
}
```
```bibtex
@article{macenski2023survey,
      title={From the desks of ROS maintainers: A survey of modern & capable mobile robotics algorithms in the robot operating system 2},
      author={S. Macenski, T. Moore, DV Lu, A. Merzlyakov, M. Ferguson},
      year={2023},
      journal = {Robotics and Autonomous Systems}
}
```
```bibtex
@article{macenski2023regulated,
      title={Regulated Pure Pursuit for Robot Path Tracking},
      author={Steve Macenski and Shrijit Singh and Francisco Martin and Jonatan Gines},
      year={2023},
      journal = {Autonomous Robots}
}
```

```bibtex
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```
```bibtex
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
