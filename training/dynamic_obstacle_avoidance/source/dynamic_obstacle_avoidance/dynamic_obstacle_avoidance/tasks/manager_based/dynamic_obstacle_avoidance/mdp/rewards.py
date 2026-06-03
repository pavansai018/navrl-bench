from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import _robot_xy



def constant_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)
