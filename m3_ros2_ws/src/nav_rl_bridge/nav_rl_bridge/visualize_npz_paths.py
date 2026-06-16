import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_map(map_yaml_path):
    with open(map_yaml_path, "r") as f:
        info = yaml.safe_load(f)

    image_path = info["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read map image: {image_path}")

    resolution = float(info["resolution"])
    origin = info["origin"]
    negate = int(info.get("negate", 0))
    occupied_thresh = float(info.get("occupied_thresh", 0.65))
    free_thresh = float(info.get("free_thresh", 0.196))

    img_norm = img.astype(np.float32) / 255.0

    if negate == 0:
        occ_prob = 1.0 - img_norm
    else:
        occ_prob = img_norm

    occupied = occ_prob > occupied_thresh
    free = occ_prob < free_thresh
    unknown_or_blocked = ~free

    free_uint8 = free.astype(np.uint8)
    dist_px = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
    dist_m = dist_px * resolution

    return {
        "img": img,
        "resolution": resolution,
        "origin": origin,
        "height": img.shape[0],
        "width": img.shape[1],
        "occupied": occupied,
        "free": free,
        "unknown_or_blocked": unknown_or_blocked,
        "dist_m": dist_m,
    }


def world_to_pixel(x_world, y_world, map_info):
    res = map_info["resolution"]
    origin_x, origin_y, origin_yaw = map_info["origin"]
    height = map_info["height"]

    dx = x_world - origin_x
    dy = y_world - origin_y

    # Inverse rotation by origin_yaw.
    c = math.cos(-origin_yaw)
    s = math.sin(-origin_yaw)

    mx = c * dx - s * dy
    my = s * dx + c * dy

    x_cell = int(math.floor(mx / res))
    y_from_bottom = int(math.floor(my / res))

    y_cell = height - 1 - y_from_bottom

    return x_cell, y_cell


def draw_base_map(map_info):
    h, w = map_info["height"], map_info["width"]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Unknown / blocked = gray
    canvas[:, :] = (120, 120, 120)

    # Free = white
    canvas[map_info["free"]] = (245, 245, 245)

    # Occupied = black
    canvas[map_info["occupied"]] = (0, 0, 0)

    return canvas


def line_cells_safe(p0, p1, free_mask):
    x0, y0 = p0
    x1, y1 = p1

    h, w = free_mask.shape

    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    if n <= 1:
        n = 2

    xs = np.linspace(x0, x1, n).round().astype(np.int32)
    ys = np.linspace(y0, y1, n).round().astype(np.int32)

    if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
        return False

    return bool(np.all(free_mask[ys, xs]))


def check_path(path_xy, map_info, required_clearance):
    pixels = []

    for x, y in path_xy:
        px, py = world_to_pixel(float(x), float(y), map_info)
        pixels.append((px, py))

    h, w = map_info["height"], map_info["width"]
    free = map_info["free"]
    dist_m = map_info["dist_m"]

    valid = True
    min_clearance = float("inf")
    bad_reason = ""

    for px, py in pixels:
        if px < 0 or px >= w or py < 0 or py >= h:
            valid = False
            bad_reason = "outside_map"
            min_clearance = 0.0
            break

        if not free[py, px]:
            valid = False
            bad_reason = "point_in_occupied_or_unknown"
            min_clearance = 0.0
            break

        min_clearance = min(min_clearance, float(dist_m[py, px]))

    if valid:
        for a, b in zip(pixels[:-1], pixels[1:]):
            if not line_cells_safe(a, b, free):
                valid = False
                bad_reason = "segment_crosses_occupied_or_unknown"
                break

    clearance_ok = min_clearance >= required_clearance

    if valid and not clearance_ok:
        bad_reason = "low_clearance"

    return {
        "pixels": pixels,
        "valid": bool(valid and clearance_ok),
        "geometric_valid": bool(valid),
        "min_clearance": float(min_clearance),
        "clearance_ok": bool(clearance_ok),
        "bad_reason": bad_reason,
    }


def path_length(path_xy):
    if len(path_xy) < 2:
        return 0.0
    d = np.diff(path_xy, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def visualize_one(npz_path, map_info, out_path, required_clearance):
    data = np.load(npz_path)

    path_xy = data["path_xy"]
    start = data["start"]
    goal = data["goal"]

    result = check_path(path_xy, map_info, required_clearance)

    canvas = draw_base_map(map_info)

    pixels = result["pixels"]

    # Draw required clearance warning points.
    for px, py in pixels:
        if 0 <= px < map_info["width"] and 0 <= py < map_info["height"]:
            clearance = map_info["dist_m"][py, px]
            if clearance < required_clearance:
                cv2.circle(canvas, (px, py), 2, (0, 165, 255), -1)

    # Draw path.
    for a, b in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, a, b, (0, 0, 255), 2)

    # Draw start and goal from saved values.
    spx, spy = world_to_pixel(float(start[0]), float(start[1]), map_info)
    gpx, gpy = world_to_pixel(float(goal[0]), float(goal[1]), map_info)

    cv2.circle(canvas, (spx, spy), 5, (255, 0, 0), -1)   # start = blue
    cv2.circle(canvas, (gpx, gpy), 5, (0, 255, 0), -1)   # goal = green

    status = "PASS" if result["valid"] else "FAIL"

    text = (
        f"{npz_path.name} | {status} | "
        f"len={path_length(path_xy):.2f}m | "
        f"clearance={result['min_clearance']:.2f}m | "
        f"{result['bad_reason']}"
    )

    cv2.rectangle(canvas, (5, 5), (min(canvas.shape[1] - 5, 1050), 35), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), canvas)

    return {
        "file": npz_path.name,
        "status": status,
        "geometric_valid": result["geometric_valid"],
        "clearance_ok": result["clearance_ok"],
        "min_clearance_m": result["min_clearance"],
        "path_length_m": path_length(path_xy),
        "num_points": len(path_xy),
        "bad_reason": result["bad_reason"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", required=True, help="Path to map.yaml")
    parser.add_argument("--dataset", required=True, help="Folder containing path_*.npz")
    parser.add_argument("--out", required=True, help="Output visualization folder")

    parser.add_argument("--num", type=int, default=100, help="Number of paths to visualize")
    parser.add_argument("--every", type=int, default=1, help="Visualize every Nth path")
    parser.add_argument("--required_clearance", type=float, default=0.30)

    args = parser.parse_args()

    map_info = load_map(args.map)

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dataset_dir.glob("path_*.npz"))
    files = files[::args.every][:args.num]

    if len(files) == 0:
        raise RuntimeError(f"No path_*.npz files found in {dataset_dir}")

    rows = []

    for i, npz_path in enumerate(files):
        out_png = out_dir / f"{npz_path.stem}.png"

        row = visualize_one(
            npz_path=npz_path,
            map_info=map_info,
            out_path=out_png,
            required_clearance=args.required_clearance,
        )

        rows.append(row)

        print(
            f"[{i + 1}/{len(files)}] {npz_path.name}: "
            f"{row['status']} | "
            f"clearance={row['min_clearance_m']:.3f} m | "
            f"length={row['path_length_m']:.3f} m | "
            f"{row['bad_reason']}"
        )

    csv_path = out_dir / "path_validation_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "status",
                "geometric_valid",
                "clearance_ok",
                "min_clearance_m",
                "path_length_m",
                "num_points",
                "bad_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = len(rows) - passed

    print("")
    print(f"Finished visualizing {len(rows)} paths.")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()