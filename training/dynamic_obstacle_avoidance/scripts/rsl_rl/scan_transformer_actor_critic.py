from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    layers: list[nn.Module] = []
    last_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(get_activation(activation))
        last_dim = int(hidden_dim)

    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def to_int_list(x, default: list[int]) -> list[int]:
    if x is None:
        return list(default)
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, tuple):
        return [int(v) for v in x]
    return [int(v) for v in list(x)]


def select_obs(observations, group_name: str) -> torch.Tensor:
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
        f"Cannot select {group_name} observations from type {type(observations)}"
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
    Actor observation layout:

    local_path_window       16
    heading_error            1
    cross_track_error        1
    scan_history       8 * 144 = 1152
    base_lin_vel             2
    base_ang_vel             1
    previous_action          3

    Total = 1176

    Actor input remains scan/path/velocity/previous-action only.
    Auxiliary heads are training-only.
    """

    def __init__(
        self,
        num_actions: int,
        scan_history_len: int = 8,
        num_rays: int = 144,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
    ):
        super().__init__()

        self.path_dim = 18
        self.scan_history_len = int(scan_history_len)
        self.num_rays = int(num_rays)
        self.scan_dim = self.scan_history_len * self.num_rays
        self.motion_dim = 6

        self.expected_actor_obs_dim = self.path_dim + self.scan_dim + self.motion_dim

        ray_feature_dim = (
            self.scan_history_len
            + (self.scan_history_len - 1)
            + 1
            + 1
            + 1
            + 2
        )

        self.ray_proj = nn.Sequential(
            nn.Linear(ray_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ray_pos_embed = nn.Parameter(torch.zeros(1, self.num_rays, d_model))
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

        self.ray_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        fused_dim = d_model * 5
        self.feature_dim = fused_dim

        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_actions),
        )

        self.aux_path_blocked_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.aux_dynamic_risk_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            obs = obs.reshape(obs.shape[0], -1)

        if obs.shape[1] != self.expected_actor_obs_dim:
            raise RuntimeError(
                f"Actor expected obs dim {self.expected_actor_obs_dim}, got {obs.shape[1]}"
            )

        path = obs[:, :18]
        scan_flat = obs[:, 18 : 18 + self.scan_dim]
        motion = obs[:, 18 + self.scan_dim : 18 + self.scan_dim + self.motion_dim]

        batch_size = obs.shape[0]

        scan_hist = scan_flat.reshape(
            batch_size,
            self.scan_history_len,
            self.num_rays,
        )

        scan_hist = torch.clamp(scan_hist, 0.0, 4.0)
        scan_norm = scan_hist / 4.0

        scan_delta = scan_norm[:, :-1, :] - scan_norm[:, 1:, :]
        current_scan = scan_norm[:, -1:, :]
        min_scan = torch.min(scan_norm, dim=1, keepdim=True).values
        closing_full = torch.clamp(scan_norm[:, 0:1, :] - scan_norm[:, -1:, :], -1.0, 1.0)

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
        ).reshape(1, 2, self.num_rays)

        angle_feat = angle_feat.expand(batch_size, -1, -1)

        ray_features = torch.cat(
            [
                scan_norm,
                scan_delta,
                current_scan,
                min_scan,
                closing_full,
                angle_feat,
            ],
            dim=1,
        )

        ray_features = ray_features.transpose(1, 2).contiguous()

        ray_tokens = self.ray_proj(ray_features)
        ray_tokens = ray_tokens + self.ray_pos_embed
        ray_tokens = self.ray_transformer(ray_tokens)

        scores = self.ray_score(ray_tokens)
        weights = torch.softmax(scores, dim=1)

        ray_attn_pool = torch.sum(weights * ray_tokens, dim=1)
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

        return fused

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature = self.encode(obs)
        return self.actor_head(feature)

    def auxiliary_logits(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encode(obs)
        path_blocked_logits = self.aux_path_blocked_head(feature)
        dynamic_risk_logits = self.aux_dynamic_risk_head(feature)
        return path_blocked_logits, dynamic_risk_logits


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

        num_actions = infer_num_actions(num_actions, fallback=None)

        actor_hidden_dims = to_int_list(actor_hidden_dims, [256, 128])
        critic_hidden_dims = to_int_list(critic_hidden_dims, [256, 256, 128])

        self.num_actor_obs = actor_obs_dim
        self.num_critic_obs = critic_obs_dim
        self.num_actions = num_actions
        self.noise_std_type = str(noise_std_type)

        self.actor = ScanHistoryTransformerActor(
            num_actions=num_actions,
            scan_history_len=8,
            num_rays=144,
            d_model=128,
            nhead=4,
            num_layers=2,
            ff_dim=256,
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
        self.aux_call_count = 0

    def reset(self, dones=None):
        pass

    def update_normalization(self, observations):
        return

    def get_normalization_state(self):
        return {}

    def load_normalization_state(self, state):
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

    @staticmethod
    def _balanced_bce_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float().clamp(0.0, 1.0)

        with torch.no_grad():
            positive_rate = target.mean().clamp(0.02, 0.98)
            pos_weight = ((1.0 - positive_rate) / positive_rate).reshape(1)

        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
        )

    def _aux_targets_from_critic_obs(
        self,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Critic layout:

        0:1176      policy obs
        1176:1200   dynamic_obstacles, 24
        1200:1201   path_blocked
        1201:1205   time_to_closest_approach: [tca0, dca0, tca1, dca1]
        1205:1206   distance_to_goal
        1206:1207   progress_fraction
        1207:1208   map_collision
        1208:1209   dynamic_collision
        """
        critic_obs = critic_obs.detach()

        path_blocked = critic_obs[:, 1200:1201].float().clamp(0.0, 1.0)

        ttc_dca = critic_obs[:, 1201:1205].float()
        ttc_dca = torch.nan_to_num(ttc_dca, nan=1.0, posinf=1.0, neginf=0.0)

        tca_norm = ttc_dca[:, 0::2].clamp(0.0, 1.0)
        dca_norm = ttc_dca[:, 1::2].clamp(0.0, 1.0)

        # TCA is normalized by horizon_s.
        # DCA is normalized by max_range.
        # Risk target: obstacle soon + predicted closest distance small.
        soon_risk = torch.clamp((0.45 - tca_norm) / 0.45, 0.0, 1.0)
        close_risk = torch.clamp((0.14 - dca_norm) / 0.14, 0.0, 1.0)

        dynamic_risk = soon_risk * close_risk
        dynamic_risk = torch.max(dynamic_risk, dim=-1, keepdim=True).values

        dynamic_collision = critic_obs[:, 1208:1209].float().clamp(0.0, 1.0)
        dynamic_risk = torch.maximum(dynamic_risk, dynamic_collision)

        return path_blocked, dynamic_risk

    def auxiliary_loss(
        self,
        actor_observations,
        critic_observations,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        policy_obs = select_obs(actor_observations, "policy")
        critic_obs = select_obs(critic_observations, "critic")

        path_blocked_target, dynamic_risk_target = self._aux_targets_from_critic_obs(
            critic_obs
        )

        path_blocked_logits, dynamic_risk_logits = self.actor.auxiliary_logits(policy_obs)

        blocked_loss = self._balanced_bce_with_logits(
            path_blocked_logits,
            path_blocked_target,
        )

        risk_loss = self._balanced_bce_with_logits(
            dynamic_risk_logits,
            dynamic_risk_target,
        )

        aux_loss = blocked_loss + 0.5 * risk_loss

        self.aux_call_count += 1
        if self.aux_call_count % 500 == 0:
            with torch.no_grad():
                print(
                    "[AUX]",
                    "calls=", self.aux_call_count,
                    "loss=", float(aux_loss.detach().cpu()),
                    "blocked_pred=", float(torch.sigmoid(path_blocked_logits).mean().detach().cpu()),
                    "blocked_target=", float(path_blocked_target.mean().detach().cpu()),
                    "risk_pred=", float(torch.sigmoid(dynamic_risk_logits).mean().detach().cpu()),
                    "risk_target=", float(dynamic_risk_target.mean().detach().cpu()),
                    flush=True,
                )

        info = {
            "aux_loss": float(aux_loss.detach().cpu()),
            "aux_blocked_loss": float(blocked_loss.detach().cpu()),
            "aux_risk_loss": float(risk_loss.detach().cpu()),
            "aux_blocked_target": float(path_blocked_target.mean().detach().cpu()),
            "aux_risk_target": float(dynamic_risk_target.mean().detach().cpu()),
        }

        return aux_loss, info

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