import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return qz, qw


def load_map_free_space(map_yaml_path: str, min_wall_distance_m: float):
    with open(map_yaml_path, "r") as f:
        info = yaml.safe_load(f)

    image_path = info["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    resolution = float(info["resolution"])
    origin = info["origin"]
    negate = int(info.get("negate", 0))
    occupied_thresh = float(info.get("occupied_thresh", 0.65))
    free_thresh = float(info.get("free_thresh", 0.196))

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load map image: {image_path}")

    img = img.astype(np.float32) / 255.0

    if negate == 0:
        occ_prob = 1.0 - img
    else:
        occ_prob = img

    occupied = occ_prob > occupied_thresh
    free = occ_prob < free_thresh

    # Keep only free cells with clearance from walls.
    free_uint8 = free.astype(np.uint8)
    dist_px = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
    min_dist_px = min_wall_distance_m / resolution
    safe_free = free & (dist_px > min_dist_px)

    ys, xs = np.where(safe_free)

    return {
        "resolution": resolution,
        "origin": origin,
        "height": img.shape[0],
        "width": img.shape[1],
        "safe_cells": np.stack([xs, ys], axis=1),
        "occupied": occupied,
        "free": free,
        "safe_free": safe_free,
    }


def map_cell_to_world(cell_xy, map_info):
    x_cell, y_cell = cell_xy
    res = map_info["resolution"]
    origin_x, origin_y, _ = map_info["origin"]
    height = map_info["height"]

    # ROS map origin is bottom-left. Image y is top-down.
    x_world = origin_x + (x_cell + 0.5) * res
    y_world = origin_y + ((height - y_cell - 1) + 0.5) * res

    return float(x_world), float(y_world)


def path_to_xy(path_msg):
    pts = []
    for pose_stamped in path_msg.poses:
        pts.append([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
        ])
    return np.asarray(pts, dtype=np.float32)


def path_length(path_xy):
    if len(path_xy) < 2:
        return 0.0
    d = np.diff(path_xy, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


class Nav2PathDatasetCollector(Node):
    def __init__(self, args):
        super().__init__("nav2_path_dataset_collector")

        self.args = args
        self.client = ActionClient(self, ComputePathToPose, args.action_name)

        self.get_logger().info(f"Waiting for Nav2 action: {args.action_name}")
        self.client.wait_for_server()
        self.get_logger().info("Connected to Nav2 planner action server.")

        self.map_info = load_map_free_space(args.map, args.min_wall_distance)
        self.rng = np.random.default_rng(args.seed)

        Path(args.out).mkdir(parents=True, exist_ok=True)

    def make_pose(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.args.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        qz, qw = yaw_to_quat(yaw)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def sample_safe_pose(self):
        safe_cells = self.map_info["safe_cells"]
        idx = self.rng.integers(0, len(safe_cells))
        x, y = map_cell_to_world(safe_cells[idx], self.map_info)
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        return x, y, yaw

    def compute_path(self, start_pose, goal_pose):
        goal = ComputePathToPose.Goal()
        goal.start = start_pose
        goal.goal = goal_pose
        goal.use_start = True
        goal.planner_id = self.args.planner_id

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            return None

        return result.result.path

    def collect(self):
        saved = 0
        attempts = 0

        while saved < self.args.num_paths and attempts < self.args.max_attempts:
            attempts += 1

            sx, sy, syaw = self.sample_safe_pose()
            gx, gy, gyaw = self.sample_safe_pose()

            straight_dist = math.hypot(gx - sx, gy - sy)
            if straight_dist < self.args.min_start_goal_distance:
                continue

            start_pose = self.make_pose(sx, sy, syaw)
            goal_pose = self.make_pose(gx, gy, gyaw)

            path_msg = self.compute_path(start_pose, goal_pose)
            if path_msg is None or len(path_msg.poses) < 5:
                continue

            path_xy = path_to_xy(path_msg)
            plen = path_length(path_xy)

            if plen < self.args.min_path_length:
                continue

            if plen > self.args.max_path_length:
                continue

            out_file = Path(self.args.out) / f"path_{saved:06d}.npz"

            np.savez_compressed(
                out_file,
                start=np.asarray([sx, sy, syaw], dtype=np.float32),
                goal=np.asarray([gx, gy, gyaw], dtype=np.float32),
                path_xy=path_xy,
                path_length=np.asarray([plen], dtype=np.float32),
            )

            saved += 1

            if saved % 10 == 0:
                self.get_logger().info(
                    f"Saved {saved}/{self.args.num_paths} paths. Attempts: {attempts}"
                )

        self.get_logger().info(f"Finished. Saved {saved} paths from {attempts} attempts.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", required=True, help="Path to Nav2 map yaml.")
    parser.add_argument("--out", required=True, help="Output dataset folder.")
    parser.add_argument("--num_paths", type=int, default=1000)
    parser.add_argument("--max_attempts", type=int, default=10000)

    parser.add_argument("--action_name", default="/compute_path_to_pose")
    parser.add_argument("--planner_id", default="GridBased")
    parser.add_argument("--frame_id", default="map")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_wall_distance", type=float, default=0.35)
    parser.add_argument("--min_start_goal_distance", type=float, default=3.0)
    parser.add_argument("--min_path_length", type=float, default=3.0)
    parser.add_argument("--max_path_length", type=float, default=40.0)

    args = parser.parse_args()

    rclpy.init()
    node = Nav2PathDatasetCollector(args)
    node.collect()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()