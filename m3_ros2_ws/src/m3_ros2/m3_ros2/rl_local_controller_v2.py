#!/usr/bin/env python3

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

import tf2_ros

from ament_index_python.packages import get_package_share_directory

from scan_transformer_actor_critic import (
    ActorCriticScanTransformer,
    ScanHistoryTransformerActor,
    POLICY_OBS_DIM,
    CRITIC_OBS_DIM,
)


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class PTPolicyWrapper:
    def __init__(
        self,
        pt_path: str,
        device: str,
        logger,
    ):
        self.pt_path = pt_path
        self.device = torch.device(device)
        self.logger = logger
        self.model = None
        self.model_kind = None

        if not os.path.exists(self.pt_path):
            raise FileNotFoundError(f"Policy file not found: {self.pt_path}")

        torch.set_num_threads(1)

        # 1. Try TorchScript first.
        try:
            self.model = torch.jit.load(self.pt_path, map_location=self.device)
            self.model.eval()
            self.model_kind = "torchscript"
            self.logger.info(f"Loaded TorchScript policy: {self.pt_path}")
            return
        except Exception as exc:
            self.logger.warn(f"torch.jit.load failed, trying torch.load checkpoint: {exc}")

        # 2. Load PyTorch checkpoint / full module.
        try:
            ckpt = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.pt_path, map_location=self.device)

        if isinstance(ckpt, nn.Module):
            self.model = ckpt.to(self.device)
            self.model.eval()
            self.model_kind = "nn_module"
            self.logger.info(f"Loaded full nn.Module policy: {self.pt_path}")
            return

        state_dict = self._extract_state_dict(ckpt)
        if state_dict is None:
            raise RuntimeError(
                "Could not find state_dict in .pt file. "
                "Expected keys like model_state_dict, actor_critic_state_dict, state_dict, model, actor, or policy."
            )

        state_dict = self._clean_state_dict_prefixes(state_dict)

        # 3. Try loading as full ActorCriticScanTransformer first.
        full_model = ActorCriticScanTransformer(
            num_actor_obs=POLICY_OBS_DIM,
            num_critic_obs=CRITIC_OBS_DIM,
            num_actions=3,
        ).to(self.device)

        missing, unexpected = full_model.load_state_dict(state_dict, strict=False)

        full_loaded_actor_keys = any(k.startswith("actor.") for k in state_dict.keys())

        if full_loaded_actor_keys:
            full_model.eval()
            self.model = full_model
            self.model_kind = "actor_critic"
            self.logger.info(f"Loaded checkpoint into ActorCriticScanTransformer: {self.pt_path}")
            self.logger.info(f"Missing keys count: {len(missing)}, unexpected keys count: {len(unexpected)}")
            return

        # 4. Fallback: actor-only state dict.
        actor_model = ScanHistoryTransformerActor(
            num_actions=3,
            scan_history_len=8,
            num_rays=144,
            d_model=128,
            nhead=4,
            num_layers=2,
            ff_dim=256,
            scan_max_range=10.0,
        ).to(self.device)

        actor_missing, actor_unexpected = actor_model.load_state_dict(state_dict, strict=False)

        actor_model.eval()
        self.model = actor_model
        self.model_kind = "actor_only"

        self.logger.info(f"Loaded checkpoint into ScanHistoryTransformerActor: {self.pt_path}")
        self.logger.info(
            f"Actor missing keys count: {len(actor_missing)}, "
            f"unexpected keys count: {len(actor_unexpected)}"
        )

    def _extract_state_dict(self, ckpt):
        if not isinstance(ckpt, dict):
            return None

        candidate_keys = [
            "model_state_dict",
            "actor_critic_state_dict",
            "actor_critic",
            "state_dict",
            "model",
            "actor",
            "policy",
        ]

        for key in candidate_keys:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

        return None

    def _clean_state_dict_prefixes(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(state_dict)

        # Repeatedly remove common wrappers.
        changed = True
        while changed:
            changed = False
            keys = list(out.keys())

            for prefix in [
                "module.",
                "model.",
                "policy.",
                "student.",
                "actor_critic.",
            ]:
                if len(keys) > 0 and all(k.startswith(prefix) for k in keys):
                    out = {k[len(prefix):]: v for k, v in out.items()}
                    changed = True
                    break

        return out

    @torch.no_grad()
    def infer(self, obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs_np).float().reshape(1, -1).to(self.device)

        if obs_t.shape[1] != POLICY_OBS_DIM:
            raise RuntimeError(f"Policy expected obs dim {POLICY_OBS_DIM}, got {obs_t.shape[1]}")

        if self.model_kind == "torchscript":
            out = self.model(obs_t)

        elif self.model_kind == "actor_critic":
            out = self.model.act_inference({"policy": obs_t})

        elif self.model_kind == "actor_only":
            out = self.model(obs_t)

        else:
            # Generic full nn.Module fallback.
            if hasattr(self.model, "act_inference"):
                try:
                    out = self.model.act_inference({"policy": obs_t})
                except Exception:
                    out = self.model.act_inference(obs_t)
            elif hasattr(self.model, "actor"):
                out = self.model.actor(obs_t)
            else:
                out = self.model(obs_t)

        if isinstance(out, dict):
            for key in ["actions", "action", "mean", "mu", "loc"]:
                if key in out:
                    out = out[key]
                    break

        if isinstance(out, (tuple, list)):
            out = out[0]

        if not torch.is_tensor(out):
            raise RuntimeError(f"Policy output is not tensor: {type(out)}")

        action = out.detach().cpu().numpy().reshape(-1)

        if action.shape[0] < 3:
            raise RuntimeError(f"Policy output has less than 3 actions: {action.shape}")

        action = action[:3].astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        return action


class RLLocalControllerScanTransformerPT(Node):
    def __init__(self):
        super().__init__("rl_local_controller_scan_transformer_pt")

        try:
            pkg_share = get_package_share_directory("m3_ros2")
            default_pt_path = os.path.join(pkg_share, "rl_policies", "policy.pt")
        except Exception:
            default_pt_path = "policy.pt"

        self.declare_parameter("pt_path", default_pt_path)
        self.declare_parameter("torch_device", "cpu")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/plan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # Must match training.
        self.declare_parameter("num_path_points", 8)
        self.declare_parameter("path_step", 8)
        self.declare_parameter("path_window_normalization_m", 4.0)

        self.declare_parameter("scan_history_len", 8)
        self.declare_parameter("num_scan_rays", 144)
        self.declare_parameter("scan_max_range", 10.0)

        # Must match config.py ACTIONS.
        self.declare_parameter("max_vx", 0.5)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_wz", 1.5)

        self.declare_parameter("max_delta_vx", 0.025)
        self.declare_parameter("max_delta_vy", 0.025)
        self.declare_parameter("max_delta_wz", 0.08)

        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("enable_action_rate_limit", True)
        self.declare_parameter("enable_motion", False)

        self.declare_parameter("emergency_stop_scan_m", 0.18)
        self.declare_parameter("goal_stop_m", 0.30)
        self.declare_parameter("goal_slowdown_m", 0.50)

        self.declare_parameter("debug_log_period_s", 2.0)

        self.pt_path = str(self.get_parameter("pt_path").value)
        self.torch_device = str(self.get_parameter("torch_device").value)

        self.num_path_points = int(self.get_parameter("num_path_points").value)
        self.path_step = int(self.get_parameter("path_step").value)
        self.path_norm_m = float(self.get_parameter("path_window_normalization_m").value)

        self.scan_history_len = int(self.get_parameter("scan_history_len").value)
        self.num_scan_rays = int(self.get_parameter("num_scan_rays").value)
        self.scan_max_range = float(self.get_parameter("scan_max_range").value)

        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self.max_delta = np.array(
            [
                float(self.get_parameter("max_delta_vx").value),
                float(self.get_parameter("max_delta_vy").value),
                float(self.get_parameter("max_delta_wz").value),
            ],
            dtype=np.float32,
        )

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.enable_action_rate_limit = bool(self.get_parameter("enable_action_rate_limit").value)
        self.enable_motion = bool(self.get_parameter("enable_motion").value)

        self.emergency_stop_scan_m = float(self.get_parameter("emergency_stop_scan_m").value)
        self.goal_stop_m = float(self.get_parameter("goal_stop_m").value)
        self.goal_slowdown_m = float(self.get_parameter("goal_slowdown_m").value)

        self.debug_log_period_s = float(self.get_parameter("debug_log_period_s").value)
        self.last_debug_log_time = self.get_clock().now()

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        if self.scan_history_len != 8 or self.num_scan_rays != 144:
            raise RuntimeError(
                f"Training architecture expects scan_history_len=8 and num_scan_rays=144. "
                f"Got history={self.scan_history_len}, rays={self.num_scan_rays}"
            )

        self.policy = PTPolicyWrapper(
            pt_path=self.pt_path,
            device=self.torch_device,
            logger=self.get_logger(),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_scan_norm: Optional[np.ndarray] = None
        self.latest_scan_m: Optional[np.ndarray] = None
        self.scan_history: Optional[np.ndarray] = None

        self.latest_odom: Optional[Odometry] = None
        self.latest_path: Optional[np.ndarray] = None

        self.previous_action = np.zeros(3, dtype=np.float32)
        self.applied_cmd = np.zeros(3, dtype=np.float32)

        self._last_tf_warn_ns = 0

        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            10,
        )

        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            10,
        )

        self.create_subscription(
            Path,
            str(self.get_parameter("path_topic").value),
            self.path_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_loop,
        )

        self.get_logger().info(f"Loaded PT policy: {self.pt_path}")
        self.get_logger().info(f"Policy obs dim: {POLICY_OBS_DIM}")
        self.get_logger().info(
            "Observation layout: "
            "local_path_window(16) + heading(1) + cte(1) + "
            "scan_history(8x144=1152) + base_lin_vel(2) + base_ang_vel(1) + previous_action(3)"
        )
        self.get_logger().info(
            f"Action scale: max_vx={self.max_vx}, max_vy={self.max_vy}, max_wz={self.max_wz}"
        )
        self.get_logger().info(f"enable_motion={self.enable_motion}")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=self.scan_max_range,
        )

        ranges = np.clip(ranges, 0.0, self.scan_max_range)

        old_angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment

        # Match training scan convention:
        # torch.linspace(-pi, pi, num_rays + 1)[:-1]
        target_angles = np.linspace(
            -math.pi,
            math.pi,
            self.num_scan_rays + 1,
            dtype=np.float32,
        )[:-1]

        order = np.argsort(old_angles)
        old_angles = old_angles[order]
        ranges = ranges[order]

        ranges_resampled = np.interp(
            target_angles,
            old_angles,
            ranges,
            left=self.scan_max_range,
            right=self.scan_max_range,
        ).astype(np.float32)

        ranges_m = np.clip(ranges_resampled, 0.0, self.scan_max_range)
        scan_norm = np.clip(ranges_m / self.scan_max_range, 0.0, 1.0).astype(np.float32)

        self.latest_scan_norm = scan_norm
        self.latest_scan_m = ranges_m.astype(np.float32)

        if self.scan_history is None:
            self.scan_history = np.repeat(
                scan_norm.reshape(1, self.num_scan_rays),
                self.scan_history_len,
                axis=0,
            ).astype(np.float32)
        else:
            self.scan_history[:-1, :] = self.scan_history[1:, :]
            self.scan_history[-1, :] = scan_norm

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return

        self.latest_path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )

    def get_robot_pose_in_map(self) -> Optional[Tuple[np.ndarray, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.20),
            )
        except Exception as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_tf_warn_ns > int(2.0 * 1e9):
                self.get_logger().warn(
                    f"TF lookup failed {self.map_frame}->{self.base_frame}: {exc}"
                )
                self._last_tf_warn_ns = now_ns
            return None

        robot_xy = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
            ],
            dtype=np.float32,
        )

        robot_yaw = yaw_from_quat(tf.transform.rotation)

        return robot_xy, robot_yaw

    def build_local_path_obs(self, robot_xy: np.ndarray, robot_yaw: float):
        path = self.latest_path

        if path is None or len(path) < 2:
            return (
                np.zeros(self.num_path_points * 2, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )

        d = np.linalg.norm(path - robot_xy[None, :], axis=1)
        nearest_idx = int(np.argmin(d))

        c = math.cos(-robot_yaw)
        s = math.sin(-robot_yaw)

        local_points = []

        for k in range(self.num_path_points):
            idx = nearest_idx + (k + 1) * self.path_step
            idx = min(idx, len(path) - 1)

            rel = path[idx] - robot_xy

            x_b = c * rel[0] - s * rel[1]
            y_b = s * rel[0] + c * rel[1]

            local_points.extend([x_b, y_b])

        local_path_window = np.clip(
            np.array(local_points, dtype=np.float32) / self.path_norm_m,
            -2.0,
            2.0,
        ).astype(np.float32)

        next_idx = min(nearest_idx + 4, len(path) - 1)

        p0 = path[nearest_idx]
        p1 = path[next_idx]

        path_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        heading_error = np.array(
            [wrap_to_pi(path_yaw - robot_yaw)],
            dtype=np.float32,
        )

        cross_track = np.array(
            [float(np.min(d))],
            dtype=np.float32,
        )

        return local_path_window, heading_error, cross_track

    def reached_path_end(self, robot_xy: np.ndarray) -> bool:
        if self.latest_path is None or len(self.latest_path) < 2:
            return False

        goal_xy = self.latest_path[-1]
        dist_to_goal = float(np.linalg.norm(goal_xy - robot_xy))

        return dist_to_goal <= self.goal_stop_m

    def build_observation(self) -> Optional[np.ndarray]:
        if self.scan_history is None:
            return None

        if self.latest_odom is None:
            return None

        if self.latest_path is None:
            return None

        pose_result = self.get_robot_pose_in_map()
        if pose_result is None:
            return None

        robot_xy, robot_yaw = pose_result
        odom = self.latest_odom

        local_path_window, heading_error, cross_track = self.build_local_path_obs(
            robot_xy,
            robot_yaw,
        )

        base_lin_vel = np.array(
            [
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
            ],
            dtype=np.float32,
        )

        base_ang_vel = np.array(
            [odom.twist.twist.angular.z],
            dtype=np.float32,
        )

        scan_history_flat = self.scan_history.reshape(-1).astype(np.float32)

        obs = np.concatenate(
            [
                local_path_window,      # 0:16
                heading_error,          # 16:17
                cross_track,            # 17:18
                scan_history_flat,      # 18:1170
                base_lin_vel,           # 1170:1172
                base_ang_vel,           # 1172:1173
                self.previous_action,   # 1173:1176
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != POLICY_OBS_DIM:
            self.get_logger().error(
                f"Observation dim mismatch: built={obs.shape[0]}, expected={POLICY_OBS_DIM}"
            )
            return None

        return obs

    def publish_stop(self):
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        self.previous_action[:] = 0.0
        self.applied_cmd[:] = 0.0

    def control_loop(self):
        obs = self.build_observation()
        if obs is None:
            return

        pose_result = self.get_robot_pose_in_map()
        if pose_result is None:
            return

        robot_xy, _ = pose_result

        goal_dist = float("nan")
        if self.latest_path is not None and len(self.latest_path) >= 2:
            goal_xy = self.latest_path[-1]
            goal_dist = float(np.linalg.norm(goal_xy - robot_xy))

        if self.reached_path_end(robot_xy):
            self.publish_stop()
            return

        try:
            raw_action = self.policy.infer(obs)
        except Exception as exc:
            self.get_logger().error(f"PT policy inference failed: {exc}")
            self.publish_stop()
            return

        raw_action = np.clip(raw_action.astype(np.float32), -1.0, 1.0)

        target_cmd = np.array(
            [
                raw_action[0] * self.max_vx,
                raw_action[1] * self.max_vy,
                raw_action[2] * self.max_wz,
            ],
            dtype=np.float32,
        )

        if self.latest_path is not None and len(self.latest_path) >= 2:
            goal_xy = self.latest_path[-1]
            goal_dist = float(np.linalg.norm(goal_xy - robot_xy))

            if goal_dist < self.goal_slowdown_m:
                scale = max(0.15, goal_dist / self.goal_slowdown_m)
                target_cmd *= scale

        if self.enable_action_rate_limit:
            delta = np.clip(
                target_cmd - self.applied_cmd,
                -self.max_delta,
                self.max_delta,
            )
            self.applied_cmd = self.applied_cmd + delta
        else:
            self.applied_cmd = target_cmd

        if self.latest_scan_m is not None:
            if float(np.min(self.latest_scan_m)) < self.emergency_stop_scan_m:
                self.applied_cmd[:] = 0.0

        self.previous_action = np.array(
            [
                self.applied_cmd[0] / max(self.max_vx, 1.0e-6),
                self.applied_cmd[1] / max(self.max_vy, 1.0e-6),
                self.applied_cmd[2] / max(self.max_wz, 1.0e-6),
            ],
            dtype=np.float32,
        )

        self.previous_action = np.clip(self.previous_action, -1.0, 1.0)

        cmd = Twist()
        cmd.linear.x = float(self.applied_cmd[0])
        cmd.linear.y = float(self.applied_cmd[1])
        cmd.angular.z = float(self.applied_cmd[2])

        if self.enable_motion:
            self.cmd_pub.publish(cmd)

        now = self.get_clock().now()
        elapsed = (now - self.last_debug_log_time).nanoseconds * 1.0e-9

        if elapsed >= self.debug_log_period_s:
            scan_min = (
                float(np.min(self.latest_scan_m))
                if self.latest_scan_m is not None
                else float("nan")
            )

            self.get_logger().info(
                f"obs_dim={POLICY_OBS_DIM}, "
                f"goal_dist={goal_dist:.3f}, "
                f"scan_min_m={scan_min:.3f}, "
                f"raw_action=[{raw_action[0]:+.3f}, {raw_action[1]:+.3f}, {raw_action[2]:+.3f}], "
                f"cmd=[{cmd.linear.x:+.3f}, {cmd.linear.y:+.3f}, {cmd.angular.z:+.3f}], "
                f"prev=[{self.previous_action[0]:+.3f}, {self.previous_action[1]:+.3f}, {self.previous_action[2]:+.3f}], "
                f"motion={self.enable_motion}"
            )

            self.last_debug_log_time = now


def main():
    rclpy.init()
    node = RLLocalControllerScanTransformerPT()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
