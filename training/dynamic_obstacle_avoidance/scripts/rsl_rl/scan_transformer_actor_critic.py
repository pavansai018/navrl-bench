from __future__ import annotations
import math
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
    """
    Scan-only dynamic-obstacle actor.

    Observation layout:
      local_path_window      16
      heading_error           1
      cross_track_error       1
      scan_history       8 * 144 = 1152
      base_lin_vel            2
      base_ang_vel            1
      previous_action         3

    Total = 1176

    Main change from previous version:
      Previous: 8 temporal tokens, each token = full 144-ray scan.
      New: 144 ray tokens, each token = 8-frame temporal history of one ray.

    This lets the actor learn ray-wise closing motion from scan history without
    explicit obstacle states.
    """

    def __init__(
        self,
        num_actions: int,
        scan_history_len: int = 8,
        num_rays: int = 144,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
    ):
        super().__init__()

        self.path_dim = 18
        self.scan_history_len = scan_history_len
        self.num_rays = num_rays
        self.scan_dim = scan_history_len * num_rays
        self.motion_dim = 6

        self.expected_actor_obs_dim = self.path_dim + self.scan_dim + self.motion_dim

        # Per-ray input:
        # 8 history values
        # 7 temporal deltas
        # current range
        # min range over history
        # closing rate over full history
        # sin(theta), cos(theta)
        ray_feature_dim = scan_history_len + (scan_history_len - 1) + 1 + 1 + 1 + 2

        self.ray_proj = nn.Sequential(
            nn.Linear(ray_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned ray positional embedding: ray index / bearing information.
        self.ray_pos_embed = nn.Parameter(torch.zeros(1, num_rays, d_model))
        nn.init.trunc_normal_(self.ray_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ray_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Path and robot-motion encoders.
        self.path_encoder = nn.Sequential(
            nn.Linear(self.path_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.motion_encoder = nn.Sequential(
            nn.Linear(self.motion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Attention pooling over ray tokens.
        self.ray_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Final actor head.
        # ray_attn_pool + ray_min_pool + ray_mean_pool + path + motion
        fused_dim = d_model * 5

        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            obs = obs.view(obs.shape[0], -1)

        if obs.shape[1] != self.expected_actor_obs_dim:
            raise RuntimeError(
                f"Actor expected obs dim {self.expected_actor_obs_dim}, got {obs.shape[1]}"
            )

        path = obs[:, :18]
        scan_flat = obs[:, 18 : 18 + self.scan_dim]
        motion = obs[:, 18 + self.scan_dim : 18 + self.scan_dim + self.motion_dim]

        batch_size = obs.shape[0]

        # scan_hist: [B, T, R]
        scan_hist = scan_flat.view(batch_size, self.scan_history_len, self.num_rays)

        # Clamp to stable range. Your scan max is usually 4.0.
        scan_hist = torch.clamp(scan_hist, 0.0, 4.0)

        # Normalize range to roughly [0, 1].
        scan_norm = scan_hist / 4.0

        # Temporal deltas per ray.
        # Positive closing means obstacle/range is getting closer.
        # If old range=3.0 and new range=2.0, old-new=+1.0.
        scan_delta = scan_norm[:, :-1, :] - scan_norm[:, 1:, :]  # [B, 7, R]

        current_scan = scan_norm[:, -1:, :]                      # [B, 1, R]
        min_scan = torch.min(scan_norm, dim=1, keepdim=True).values  # [B, 1, R]

        closing_full = scan_norm[:, 0:1, :] - scan_norm[:, -1:, :]   # [B, 1, R]
        closing_full = torch.clamp(closing_full, -1.0, 1.0)

        # Ray bearing encoding. Assumes 360-degree scan from -pi to pi.
        # Shape: [1, 2, R], then expanded by batch.
        ray_angles = torch.linspace(
            -math.pi,
            math.pi,
            self.num_rays,
            device=obs.device,
            dtype=obs.dtype,
        )
        angle_feat = torch.stack(
            [torch.sin(ray_angles), torch.cos(ray_angles)],
            dim=0,
        ).view(1, 2, self.num_rays)
        angle_feat = angle_feat.expand(batch_size, -1, -1)

        # Build per-ray features: [B, F, R]
        ray_features = torch.cat(
            [
                scan_norm,       # [B, 8, R]
                scan_delta,      # [B, 7, R]
                current_scan,    # [B, 1, R]
                min_scan,        # [B, 1, R]
                closing_full,    # [B, 1, R]
                angle_feat,      # [B, 2, R]
            ],
            dim=1,
        )

        # [B, F, R] -> [B, R, F]
        ray_features = ray_features.transpose(1, 2).contiguous()

        # [B, R, D]
        ray_tokens = self.ray_proj(ray_features)
        ray_tokens = ray_tokens + self.ray_pos_embed

        ray_tokens = self.ray_transformer(ray_tokens)

        # Attention pooling: focus on important obstacle/free-space sectors.
        scores = self.ray_score(ray_tokens)              # [B, R, 1]
        weights = torch.softmax(scores, dim=1)
        ray_attn_pool = torch.sum(weights * ray_tokens, dim=1)

        # Global scan summaries.
        ray_mean_pool = torch.mean(ray_tokens, dim=1)
        ray_min_pool = torch.min(ray_tokens, dim=1).values

        path_feat = self.path_encoder(path)
        motion_feat = self.motion_encoder(motion)

        fused = torch.cat(
            [
                ray_attn_pool,
                ray_min_pool,
                ray_mean_pool,
                path_feat,
                motion_feat,
            ],
            dim=-1,
        )

        return self.actor_head(fused)


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