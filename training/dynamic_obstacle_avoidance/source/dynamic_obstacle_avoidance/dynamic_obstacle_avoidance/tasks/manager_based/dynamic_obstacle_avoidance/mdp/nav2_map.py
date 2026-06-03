from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


class Nav2OccupancyMap:
    """Shared Nav2 occupancy-grid map for IsaacLab.

    This is NOT a PhysX object.
    It is a tensor map used for:
      - map-based lidar scan
      - map collision checking
      - debug visualization only
    """

    def __init__(
        self,
        map_yaml_path: str,
        device: str,
        occupied_threshold: float | None = None,
        free_threshold: float | None = None,
        inflation_radius_m: float = 0.12,
    ):
        self.map_yaml_path = Path(map_yaml_path)
        self.device = device

        if not self.map_yaml_path.exists():
            raise FileNotFoundError(f"Map yaml not found: {self.map_yaml_path}")

        with open(self.map_yaml_path, "r") as f:
            info = yaml.safe_load(f)

        self.resolution = float(info["resolution"])
        self.origin = info["origin"]
        self.origin_x = float(self.origin[0])
        self.origin_y = float(self.origin[1])
        self.origin_yaw = float(self.origin[2]) if len(self.origin) > 2 else 0.0

        self.negate = int(info.get("negate", 0))
        self.occupied_thresh = float(
            occupied_threshold if occupied_threshold is not None else info.get("occupied_thresh", 0.65)
        )
        self.free_thresh = float(
            free_threshold if free_threshold is not None else info.get("free_thresh", 0.196)
        )

        image_path = Path(info["image"])
        if not image_path.is_absolute():
            image_path = self.map_yaml_path.parent / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Map image not found: {image_path}")

        img = Image.open(image_path).convert("L")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        self.height, self.width = img_np.shape

        # Nav2 map convention:
        # negate=0 means black occupied, white free.
        if self.negate == 0:
            occ_prob = 1.0 - img_np
        else:
            occ_prob = img_np

        occupied_np = occ_prob > self.occupied_thresh
        free_np = occ_prob < self.free_thresh
        unknown_np = ~(occupied_np | free_np)

        occupied = torch.tensor(occupied_np, dtype=torch.bool, device=device)
        self.free = torch.tensor(free_np, dtype=torch.bool, device=device)
        self.unknown = torch.tensor(unknown_np, dtype=torch.bool, device=device)

        # Inflate occupied cells so the map behaves like a real robot footprint / costmap.
        # This closes tiny one-pixel gaps and makes scan/collision more realistic.
        inflation_cells = max(0, int(math.ceil(inflation_radius_m / self.resolution)))

        if inflation_cells > 0:
            kernel_size = 2 * inflation_cells + 1
            occ_float = occupied.float().unsqueeze(0).unsqueeze(0)

            occ_inflated = F.max_pool2d(
                occ_float,
                kernel_size=kernel_size,
                stride=1,
                padding=inflation_cells,
            )

            occupied = occ_inflated.squeeze(0).squeeze(0).bool()

        self.occupied = occupied

        occupied_np_final = self.occupied.detach().cpu().numpy()
        ys, xs = np.where(occupied_np_final)

        self.occupied_cells_xy = torch.tensor(
            np.stack([xs, ys], axis=-1),
            dtype=torch.long,
            device=device,
        )

    def world_to_cell(self, xy_world: torch.Tensor) -> torch.Tensor:
        """Convert map-frame world xy to image cell xy."""
        x = xy_world[..., 0]
        y = xy_world[..., 1]

        cell_x = torch.floor((x - self.origin_x) / self.resolution).long()

        cell_y_from_bottom = torch.floor((y - self.origin_y) / self.resolution).long()
        cell_y = (self.height - 1) - cell_y_from_bottom

        return torch.stack([cell_x, cell_y], dim=-1)

    def cell_to_world(self, cell_xy: torch.Tensor) -> torch.Tensor:
        """Convert image cell xy to map-frame world xy at cell center."""
        cell_x = cell_xy[..., 0].float()
        cell_y = cell_xy[..., 1].float()

        x = self.origin_x + (cell_x + 0.5) * self.resolution

        cell_y_from_bottom = (self.height - 1) - cell_y
        y = self.origin_y + (cell_y_from_bottom + 0.5) * self.resolution

        return torch.stack([x, y], dim=-1)

    def in_bounds_cell(self, cell_xy: torch.Tensor) -> torch.Tensor:
        x = cell_xy[..., 0]
        y = cell_xy[..., 1]

        return (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
        )

    def is_occupied_world(
        self,
        xy_world: torch.Tensor,
        unknown_is_occupied: bool = True,
    ) -> torch.Tensor:
        """Check occupancy from map-frame xy.

        This is tensor collision, not PhysX collision.
        """
        cell_xy = self.world_to_cell(xy_world)
        in_bounds = self.in_bounds_cell(cell_xy)

        safe_x = torch.clamp(cell_xy[..., 0], 0, self.width - 1)
        safe_y = torch.clamp(cell_xy[..., 1], 0, self.height - 1)

        occupied = self.occupied[safe_y, safe_x]

        if unknown_is_occupied:
            unknown = self.unknown[safe_y, safe_x]
            occupied = occupied | unknown

        occupied = occupied | (~in_bounds)
        return occupied


    def raycast_scan(
        self,
        robot_xy: torch.Tensor,
        robot_yaw: torch.Tensor,
        num_rays: int = 72,
        max_range: float = 4.0,
        step_size: float = 0.05,
        unknown_is_occupied: bool = True,
    ) -> torch.Tensor:
        """Map-only lidar scan.

        Important:
        - It sees only the occupancy grid.
        - It does not see other robots.
        - It does not see path markers.
        - It does not see goal markers.
        - It does not depend on PhysX objects.
        """
        device = robot_xy.device
        num_envs = robot_xy.shape[0]

        ray_angles = torch.linspace(-math.pi, math.pi, num_rays, device=device)
        world_angles = robot_yaw[:, None] + ray_angles[None, :]

        distances = torch.arange(
            0.0,
            max_range + step_size,
            step_size,
            device=device,
        )

        xs = robot_xy[:, None, None, 0] + torch.cos(world_angles[:, :, None]) * distances[None, None, :]
        ys = robot_xy[:, None, None, 1] + torch.sin(world_angles[:, :, None]) * distances[None, None, :]

        pts = torch.stack([xs, ys], dim=-1)

        occupied = self.is_occupied_world(
            pts.reshape(-1, 2),
            unknown_is_occupied=unknown_is_occupied,
        ).reshape(num_envs, num_rays, -1)

        hit_any = occupied.any(dim=-1)
        first_hit_idx = torch.argmax(occupied.float(), dim=-1)

        scan = distances[first_hit_idx]
        scan = torch.where(hit_any, scan, torch.ones_like(scan) * max_range)

        return torch.clamp(scan / max_range, 0.0, 1.0)
    
    def path_occupancy_report(
        self,
        path_xy: torch.Tensor,
        unknown_is_occupied: bool = True,
    ):
        """Check how many path points fall inside occupied/unknown map cells.

        path_xy:
            [N, 2] map-frame path points
        """
        if path_xy.numel() == 0:
            return {
                "num_points": 0,
                "num_occupied": 0,
                "occupied_fraction": 0.0,
                "first_bad_points": [],
            }

        occupied = self.is_occupied_world(
            path_xy,
            unknown_is_occupied=unknown_is_occupied,
        )

        bad_ids = torch.nonzero(occupied, as_tuple=False).squeeze(-1)
        first_bad = []

        for i in bad_ids[:10]:
            p = path_xy[i]
            c = self.world_to_cell(p)
            first_bad.append(
                {
                    "path_index": int(i.item()),
                    "world_xy": [float(p[0].item()), float(p[1].item())],
                    "cell_xy": [int(c[0].item()), int(c[1].item())],
                }
            )

        return {
            "num_points": int(path_xy.shape[0]),
            "num_occupied": int(occupied.sum().item()),
            "occupied_fraction": float(occupied.float().mean().item()),
            "first_bad_points": first_bad,
        }

    def debug_occupied_world_points(
        self,
        stride: int = 1,
        max_points: int = 30000,
    ) -> torch.Tensor:
        """Dense occupied map points for DebugDraw only.

        This does not create PhysX objects.
        """
        cells = self.occupied_cells_xy

        if stride > 1:
            cells = cells[::stride]

        if cells.shape[0] > max_points:
            # deterministic enough for debug
            ids = torch.linspace(
                0,
                cells.shape[0] - 1,
                max_points,
                device=self.device,
            ).long()
            cells = cells[ids]

        return self.cell_to_world(cells)