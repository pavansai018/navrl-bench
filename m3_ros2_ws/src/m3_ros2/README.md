# NavRL Bench ROS 2 Package

**ROS 2 integration package for ROSMASTER M3 dynamic obstacle avoidance and Nav2 benchmarking**

This package is the ROS 2 side of the **NavRL Bench / Dynamic Obstacle Avoidance** project.  
The IsaacLab side trains the RL policy. This ROS 2 package is responsible for loading the trained policy, connecting it with ROSMASTER M3 navigation topics, and supporting benchmarking against classical Nav2 controllers.

The final goal is simple:

> Given a goal, ROSMASTER M3 should avoid fast-moving dynamic obstacles and reach the goal safely.

---

## Purpose of this Package

This ROS 2 package supports:

- ROSMASTER M3 real-robot or simulation integration
- Nav2 baseline controller testing
- RL policy inference inside ROS 2
- Dynamic-obstacle avoidance experiments
- Benchmarking between DWB, MPPI, Regulated Pure Pursuit, and the trained RL policy
- Logging metrics such as success rate, time to goal, number of stops, collision events, and dynamic-obstacle clearance

---

## System Architecture

```text
IsaacLab Training Side
    ↓
Trained PPO Policy
    ↓
Policy Export
    ↓
ROS 2 Inference Node
    ↓
/cmd_vel
    ↓
ROSMASTER M3
```

The global planner remains unchanged. The main focus is the **local-control layer**.

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
   └── RL Policy Node
   ↓
/cmd_vel
   ↓
ROSMASTER M3
```

---

## Planned Package Responsibilities

### 1. RL Policy Inference

- Load exported PPO policy
- Subscribe to observation topics
- Build the same observation vector used during IsaacLab training
- Run policy inference
- Publish safe `/cmd_vel`

### 2. Observation Processing

- Read laser scan or local obstacle data
- Read robot odometry
- Read goal or local path information
- Convert ROS 2 data into the RL observation format

### 3. Safety Filtering

- Clamp linear and angular velocity
- Prevent unsafe commands
- Stop robot if observations are stale
- Stop robot if policy output is invalid

### 4. Benchmark Logging

- Track time to goal
- Track number of stops
- Track path length
- Track collision or near-collision events
- Track minimum obstacle distance
- Store experiment results

### 5. Nav2 Baseline Support

- Launch DWB baseline
- Launch MPPI baseline
- Launch Regulated Pure Pursuit baseline
- Keep test conditions consistent across all controllers

---

## Suggested Package Structure

```text
m3_ros2/
│
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── ros2_pkg
│
├── m3_ros2/
│   ├── __init__.py
│   ├── rl_policy_node.py
│   ├── observation_builder.py
│   ├── safety_filter.py
│   ├── benchmark_logger.py
│   └── utils.py
│
├── config/
│   ├── rl_policy.yaml
│   ├── benchmark.yaml
│   ├── nav2_dwb.yaml
│   ├── nav2_mppi.yaml
│   └── nav2_rpp.yaml
│
├── launch/
│   ├── rl_controller.launch.py
│   ├── benchmark_dwb.launch.py
│   ├── benchmark_mppi.launch.py
│   ├── benchmark_rpp.launch.py
│   └── benchmark_rl.launch.py
│
├── policies/
│   └── README.md
│
├── results/
│   └── README.md
│
└── README.md
```

---

## Main Nodes

### `rl_policy_node.py`

Loads the trained RL policy and publishes velocity commands.

Expected subscriptions:

```text
/scan
/odom
/goal_pose or /navigate_to_pose goal
/nav2_global_path or local path topic
```

Expected publication:

```text
/cmd_vel
```

Main responsibility:

```text
ROS 2 observations → RL observation vector → policy inference → safe /cmd_vel
```

---

### `observation_builder.py`

Builds the observation vector expected by the trained policy.

Planned observation inputs:

- Goal distance
- Goal direction relative to robot
- Robot linear velocity
- Robot angular velocity
- Laser scan or local obstacle distances
- Previous action
- Optional local path direction
- Optional nearest dynamic obstacle information

Important rule:

> The ROS 2 observation format must match the IsaacLab training observation format.

---

### `safety_filter.py`

Applies safety checks before publishing velocity commands.

Planned checks:

- Clamp linear velocity
- Clamp angular velocity
- Reject NaN or invalid policy output
- Stop if `/scan` is stale
- Stop if `/odom` is stale
- Stop if obstacle is dangerously close
- Enforce ROSMASTER M3 velocity limits

---

### `benchmark_logger.py`

Records experiment metrics.

Planned metrics:

- Success rate
- Time to goal
- Path length
- Number of stops
- Collision count
- Minimum obstacle distance
- Dynamic obstacle clearance
- Velocity smoothness

---

## RL Policy Input / Output

### Input

The RL policy should receive a fixed-size observation vector built from ROS 2 topics.

Example:

```text
[laser_scan, goal_distance, goal_angle, robot_v, robot_w, previous_action]
```

### Output

The policy outputs:

```text
linear_velocity
angular_velocity
```

The output is converted into a valid ROS 2 `Twist` message:

```text
geometry_msgs/msg/Twist
```

Published to:

```text
/cmd_vel
```

---

## Benchmark Controllers

The package supports comparison between:

- DWB
- MPPI
- Regulated Pure Pursuit
- PPO-trained RL policy

The benchmark principle is:

> Same robot, same map, same goal, same dynamic-obstacle setup, same robot limits. Only the local controller changes.

---

## Benchmark Scenarios

### 1. Fast Dynamic Obstacle Crossing

A moving obstacle crosses the robot path while the robot attempts to reach the goal.

### 2. Temporary Path Blockage

A moving obstacle briefly blocks the route, forcing the robot to slow down, wait, or locally avoid.

### 3. Goal Reaching with Moving Obstacles

The robot must reach the assigned goal while reacting to moving obstacles.

### 4. Static Obstacle Baseline

A simple baseline scenario with fixed obstacles only.

### 5. Narrow Passage Navigation

A constrained path where local-control behavior is clearly visible.

---

## Installation

Create a ROS 2 workspace:

```bash
mkdir -p ~/m3_ros2_ws/src
cd ~/m3_ros2_ws/src
```

Clone this package into `src`:

```bash
git clone <ROS2_PACKAGE_REPO_URL>
```

Build the workspace:

```bash
cd ~/m3_ros2_ws
colcon build
```

Source the workspace:

```bash
source install/setup.bash
```

---

## Example Launch Commands

Launch RL controller node:

```bash
ros2 launch m3_ros2 rl_controller.launch.py
```

Launch benchmark with DWB:

```bash
ros2 launch m3_ros2 benchmark_dwb.launch.py
```

Launch benchmark with MPPI:

```bash
ros2 launch m3_ros2 benchmark_mppi.launch.py
```

Launch benchmark with Regulated Pure Pursuit:

```bash
ros2 launch m3_ros2 benchmark_rpp.launch.py
```

Launch benchmark with RL policy:

```bash
ros2 launch m3_ros2 benchmark_rl.launch.py
```

---

## Configuration Files

### `config/rl_policy.yaml`

```yaml
policy_path: "policies/policy.onnx"
use_onnx: true
max_linear_velocity: 0.6
max_angular_velocity: 1.8
scan_topic: "/scan"
odom_topic: "/odom"
cmd_vel_topic: "/cmd_vel"
```

### `config/benchmark.yaml`

```yaml
goal_tolerance: 0.25
timeout_seconds: 120
stop_velocity_threshold: 0.03
minimum_safe_distance: 0.30
log_results: true
```

---

## Sim-to-Real Considerations

The ROS 2 side must handle the sim-to-real gap carefully.

Important concerns:

- Real laser scan noise
- Odometry drift
- Wheel slip
- Command delay
- Difference between IsaacLab observations and ROS 2 observations
- Velocity and acceleration limits
- Safety filtering before `/cmd_vel`

The RL policy should not directly control the robot without safety checks.

---

## Current Status

This package is in the initial setup stage.

Immediate tasks:

- Create ROS 2 package structure
- Add RL policy inference node
- Add observation builder
- Add safety filter
- Add benchmark logger
- Add Nav2 controller launch files
- Connect exported IsaacLab policy
- Test first in simulation before real robot deployment

---

