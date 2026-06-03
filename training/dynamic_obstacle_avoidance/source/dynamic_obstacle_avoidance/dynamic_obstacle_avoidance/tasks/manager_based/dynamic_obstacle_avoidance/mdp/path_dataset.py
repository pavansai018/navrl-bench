from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class Nav2PathDataset:
    """Loads Nav2-generated global paths saved as .npz files."""

    def __init__(
        self,
        dataset_dir: str,
        device: str,
        max_path_points: int = 600,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.device = device
        self.max_path_points = max_path_points

        self.files = sorted(self.dataset_dir.glob("path_*.npz"))

        if len(self.files) == 0:
            raise RuntimeError(f"No path_*.npz files found in {self.dataset_dir}")

    def __len__(self):
        return len(self.files)

    def sample_batch(self, env_ids: torch.Tensor):
        """Sample one stored Nav2 path per env_id."""

        num_envs = len(env_ids)

        starts = torch.zeros(num_envs, 3, device=self.device)
        goals = torch.zeros(num_envs, 3, device=self.device)
        paths = torch.zeros(num_envs, self.max_path_points, 2, device=self.device)
        path_lengths = torch.zeros(num_envs, device=self.device)
        valid_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        file_indices = torch.randint(
            low=0,
            high=len(self.files),
            size=(num_envs,),
            device=self.device,
        ).cpu().numpy()

        for i, file_idx in enumerate(file_indices):
            data = np.load(self.files[file_idx])

            start = data["start"].astype(np.float32)
            goal = data["goal"].astype(np.float32)
            path_xy = data["path_xy"].astype(np.float32)

            n = min(len(path_xy), self.max_path_points)

            starts[i] = torch.tensor(start, device=self.device)
            goals[i] = torch.tensor(goal, device=self.device)
            paths[i, :n] = torch.tensor(path_xy[:n], device=self.device)
            valid_counts[i] = n

            if "path_length" in data:
                path_lengths[i] = float(data["path_length"][0])
            else:
                diff = path_xy[1:] - path_xy[:-1]
                path_lengths[i] = float(np.linalg.norm(diff, axis=1).sum())

        return starts, goals, paths, valid_counts, path_lengths