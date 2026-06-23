from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


POLICY_OBS_DIM = 1176
CRITIC_OBS_DIM = 1209


def get_activation(name: str) -> nn.Module:
    name = str(name).lower()

    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()

    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_dims = [int(v) for v in hidden_dims]

    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(get_activation(activation))
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def to_int_list(x, default: list[int]) -> list[int]:
    if x is None:
        return list(default)

    if isinstance(x, list):
        return [int(v) for v in x]

    if isinstance(x, tuple):
        return [int(v) for v in x]

    try:
        return [int(v) for v in list(x)]
    except Exception as exc:
        raise TypeError(f"Cannot convert hidden dims to int list: {x}") from exc


def select_obs(observations, group_name: str) -> torch.Tensor:
    """Select policy/critic tensor from TensorDict, dict, or already-flat tensor."""

    if isinstance(observations, torch.Tensor):
        return observations

    if hasattr(observations, "keys") and hasattr(observations, "__getitem__"):
        keys = list(observations.keys())

        if group_name in keys:
            value = observations[group_name]
            if isinstance(value, torch.Tensor):
                return value
            return select_obs(value, group_name)

        if "obs" in keys:
            return select_obs(observations["obs"], group_name)

        if "observations" in keys:
            return select_obs(observations["observations"], group_name)

    raise TypeError(
        f"Cannot select {group_name} observations from type {type(observations)}: {observations}"
    )


def infer_num_actions(num_actions, fallback=None) -> int:
    if num_actions is not None:
        if isinstance(num_actions, torch.Tensor):
            return int(num_actions.item())
        return int(num_actions)

    if fallback is not None:
        if isinstance(fallback, torch.Tensor):
            return int(fallback.item())
        if isinstance(fallback, int):
            return int(fallback)

    raise TypeError(
        f"Cannot infer num_actions. num_actions={num_actions}, fallback={fallback}"
    )


class ScanHistoryTransformerActor(nn.Module):
    """Transformer actor for policy obs layout:

    local_path_window:      16
    nav2_heading_error:      1
    nav2_cross_track_error:  1
    scan_history:        1152
    base_lin_vel:           2
    base_angle_vel:         1
    previous_action:        3

    Total: 1176
    """

    def __init__(
        self,
        num_actions: int,
        history_len: int = 8,
        num_rays: int = 144,
        path_dim: int = 18,
        motion_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        activation: str = "elu",
        actor_hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        self.history_len = int(history_len)
        self.num_rays = int(num_rays)
        self.path_dim = int(path_dim)
        self.motion_dim = int(motion_dim)
        self.scan_dim = self.history_len * self.num_rays
        self.expected_actor_obs_dim = self.path_dim + self.scan_dim + self.motion_dim

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]

        self.scan_proj = nn.Linear(self.num_rays, d_model)
        self.path_proj = nn.Linear(self.path_dim, d_model)
        self.motion_proj = nn.Linear(self.motion_dim, d_model)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.history_len + 2, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.actor_head = build_mlp(
            input_dim=d_model,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.expected_actor_obs_dim:
            raise RuntimeError(
                f"Actor obs dim mismatch. Expected {self.expected_actor_obs_dim}, "
                f"got {obs.shape[-1]}. Expected layout: "
                f"18 path + {self.scan_dim} scan_history + 6 motion."
            )

        path = obs[:, 0:self.path_dim]

        scan_start = self.path_dim
        scan_end = scan_start + self.scan_dim

        scan_hist = obs[:, scan_start:scan_end].reshape(
            obs.shape[0],
            self.history_len,
            self.num_rays,
        )

        motion = obs[:, scan_end:scan_end + self.motion_dim]

        path_token = self.path_proj(path).unsqueeze(1)
        motion_token = self.motion_proj(motion).unsqueeze(1)
        scan_tokens = self.scan_proj(scan_hist)

        tokens = torch.cat(
            [
                path_token,
                motion_token,
                scan_tokens,
            ],
            dim=1,
        )

        tokens = tokens + self.pos_embed

        encoded = self.encoder(tokens)

        feature = encoded[:, 0, :]

        return self.actor_head(feature)


class ActorCriticScanTransformer(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs=None,
        num_actions=None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.3,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()

        actor_obs_dim = POLICY_OBS_DIM
        critic_obs_dim = CRITIC_OBS_DIM

        if hasattr(num_actor_obs, "keys") and hasattr(num_actor_obs, "__getitem__"):
            try:
                actor_obs_dim = int(select_obs(num_actor_obs, "policy").shape[-1])
            except Exception:
                actor_obs_dim = POLICY_OBS_DIM

            try:
                critic_obs_dim = int(select_obs(num_actor_obs, "critic").shape[-1])
            except Exception:
                critic_obs_dim = CRITIC_OBS_DIM

        num_actions = infer_num_actions(num_actions, fallback=num_critic_obs)

        actor_hidden_dims = to_int_list(actor_hidden_dims, [256, 128])
        critic_hidden_dims = to_int_list(critic_hidden_dims, [256, 256, 128])

        self.num_actor_obs = actor_obs_dim
        self.num_critic_obs = critic_obs_dim
        self.num_actions = num_actions
        self.noise_std_type = str(noise_std_type)

        self.actor = ScanHistoryTransformerActor(
            num_actions=num_actions,
            history_len=8,
            num_rays=144,
            path_dim=18,
            motion_dim=6,
            d_model=128,
            nhead=4,
            num_layers=2,
            ff_dim=256,
            activation=activation,
            actor_hidden_dims=actor_hidden_dims,
        )

        self.critic = build_mlp(
            input_dim=critic_obs_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
        )

        if self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(torch.ones(num_actions) * float(init_noise_std))
            )
            self.std = None
        else:
            self.std = nn.Parameter(
                torch.ones(num_actions) * float(init_noise_std)
            )
            self.log_std = None

        self.distribution: Normal | None = None

    def reset(self, dones=None):
        pass

    def update_normalization(self, observations):
        # Required by this RSL-RL version.
        # This custom actor-critic is not using internal observation normalizers.
        return

    def get_normalization_state(self):
        # Required for checkpoint compatibility in some RSL-RL versions.
        return {}

    def load_normalization_state(self, state):
        # Required for checkpoint compatibility in some RSL-RL versions.
        return

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _get_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "log":
            std = torch.exp(self.log_std)
        else:
            std = torch.clamp(self.std, min=1.0e-6)

        return std.expand_as(mean)

    def update_distribution(self, observations):
        policy_obs = select_obs(observations, "policy")
        mean = self.actor(policy_obs)
        std = self._get_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations) -> torch.Tensor:
        policy_obs = select_obs(observations, "policy")
        return self.actor(policy_obs)

    def evaluate(self, critic_observations, **kwargs) -> torch.Tensor:
        critic_obs = select_obs(critic_observations, "critic")
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)