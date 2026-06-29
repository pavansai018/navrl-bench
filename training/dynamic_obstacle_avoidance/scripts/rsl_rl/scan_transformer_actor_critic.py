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

    path block:
        local_path_window       16
        heading_error            1
        cross_track_error        1

    scan:
        scan_history       8 * 144 = 1152

    motion:
        base_lin_vel             2
        base_ang_vel             1
        previous_action          3

    Total = 1176

    This version fixes:
      - hard-coded 4m scan scaling
      - scan pooling before path fusion
      - weak crossing-obstacle representation
      - no path-conditioned ray attention
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
        scan_max_range: float = 10.0,
    ):
        super().__init__()

        self.path_dim = 18
        self.scan_history_len = int(scan_history_len)
        self.num_rays = int(num_rays)
        self.scan_dim = self.scan_history_len * self.num_rays
        self.motion_dim = 6
        self.scan_max_range = float(scan_max_range)

        self.expected_actor_obs_dim = self.path_dim + self.scan_dim + self.motion_dim

        # Per-ray features:
        # scan history                  T
        # temporal deltas               T - 1
        # current range                 1
        # min range                     1
        # full-history closing          1
        # ray angle sin/cos             2
        # path-ray alignment            1
        # current scan spatial grad     2
        # near obstacle score           1
        ray_feature_dim = (
            self.scan_history_len
            + (self.scan_history_len - 1)
            + 1
            + 1
            + 1
            + 2
            + 1
            + 2
            + 1
        )

        self.ray_proj = nn.Sequential(
            nn.Linear(ray_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ray_pos_embed = nn.Parameter(torch.zeros(1, self.num_rays, d_model))
        nn.init.trunc_normal_(self.ray_pos_embed, std=0.02)

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

        # Path + motion condition injected into every ray token.
        self.context_film = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
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

        self.ray_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Path-conditioned cross attention:
        # query = path/motion context
        # key/value = ray tokens
        self.context_query = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.path_ray_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True,
        )

        # Extra learned score after ray tokens are already path-conditioned.
        self.ray_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        fused_dim = d_model * 6
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

    def _normalize_scan(self, scan_hist: torch.Tensor) -> torch.Tensor:
        """
        Handles both cases:

        Case A:
            scan_history already normalized [0, 1]

        Case B:
            scan_history is raw meters [0, scan_max_range]

        Output:
            scan_norm in [0, 1], where smaller = closer obstacle.
        """

        scan_hist = torch.nan_to_num(
            scan_hist,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )

        # Batch-level detection is okay because the observation convention
        # is fixed during a run.
        max_val = scan_hist.detach().amax()

        if max_val <= 1.5:
            return torch.clamp(scan_hist, 0.0, 1.0)

        scan_hist = torch.clamp(scan_hist, 0.0, self.scan_max_range)
        return scan_hist / self.scan_max_range

    def _path_ray_alignment(
        self,
        path: torch.Tensor,
        ray_angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes how relevant each ray direction is to the local path corridor.

        path[:, :16] is assumed to be 8 local path xy points.
        Output shape: [B, 1, R]
        """

        batch_size = path.shape[0]

        path_xy = path[:, :16].reshape(batch_size, 8, 2)
        path_norm = torch.norm(path_xy, dim=-1, keepdim=True)

        valid = path_norm > 1.0e-4
        path_dir = path_xy / path_norm.clamp_min(1.0e-6)

        ray_dirs = torch.stack(
            [torch.cos(ray_angles), torch.sin(ray_angles)],
            dim=-1,
        )  # [R, 2]

        # [B, 8, R]
        alignment = torch.einsum("bpd,rd->bpr", path_dir, ray_dirs)

        # Invalid path points should not dominate.
        alignment = torch.where(
            valid,
            alignment,
            torch.full_like(alignment, -1.0),
        )

        # Best local-path alignment per ray.
        alignment = torch.max(alignment, dim=1, keepdim=True).values

        # Convert [-1, 1] -> [0, 1]
        alignment = 0.5 * (alignment + 1.0)

        return alignment.clamp(0.0, 1.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            obs = obs.reshape(obs.shape[0], -1)

        if obs.shape[1] != self.expected_actor_obs_dim:
            raise RuntimeError(
                f"Actor expected obs dim {self.expected_actor_obs_dim}, got {obs.shape[1]}"
            )

        batch_size = obs.shape[0]

        path = obs[:, :18]
        scan_flat = obs[:, 18 : 18 + self.scan_dim]
        motion = obs[:, 18 + self.scan_dim : 18 + self.scan_dim + self.motion_dim]

        scan_hist = scan_flat.reshape(
            batch_size,
            self.scan_history_len,
            self.num_rays,
        )

        scan_norm = self._normalize_scan(scan_hist)

        # Temporal motion per ray.
        scan_delta = scan_norm[:, :-1, :] - scan_norm[:, 1:, :]
        current_scan = scan_norm[:, -1:, :]
        min_scan = torch.min(scan_norm, dim=1, keepdim=True).values

        closing_full = torch.clamp(
            scan_norm[:, 0:1, :] - scan_norm[:, -1:, :],
            -1.0,
            1.0,
        )

        # Spatial gradients help crossing obstacles that move across ray indices.
        current = current_scan
        current_left = torch.roll(current, shifts=1, dims=-1)
        current_right = torch.roll(current, shifts=-1, dims=-1)

        spatial_grad_left = torch.clamp(current - current_left, -1.0, 1.0)
        spatial_grad_right = torch.clamp(current_right - current, -1.0, 1.0)

        # Smaller range = nearer obstacle. This gives an explicit closeness signal.
        near_score = 1.0 - current_scan

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

        path_alignment = self._path_ray_alignment(
            path=path,
            ray_angles=ray_angles,
        )

        ray_features = torch.cat(
            [
                scan_norm,
                scan_delta,
                current_scan,
                min_scan,
                closing_full,
                angle_feat,
                path_alignment,
                spatial_grad_left,
                spatial_grad_right,
                near_score,
            ],
            dim=1,
        )

        ray_features = ray_features.transpose(1, 2).contiguous()

        path_feat = self.path_encoder(path)
        motion_feat = self.motion_encoder(motion)
        context = torch.cat([path_feat, motion_feat], dim=-1)

        ray_tokens = self.ray_proj(ray_features)
        ray_tokens = ray_tokens + self.ray_pos_embed

        # Inject path/motion into ray tokens before transformer.
        gamma_beta = self.context_film(context)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)

        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)

        ray_tokens = ray_tokens * (1.0 + gamma) + beta

        ray_tokens = self.ray_transformer(ray_tokens)

        # Path-conditioned cross attention.
        query = self.context_query(context).unsqueeze(1)

        path_ray_pool, _ = self.path_ray_attention(
            query=query,
            key=ray_tokens,
            value=ray_tokens,
            need_weights=False,
        )

        path_ray_pool = path_ray_pool.squeeze(1)

        # Learned path-conditioned ray pooling.
        scores = self.ray_score(ray_tokens)
        weights = torch.softmax(scores, dim=1)
        ray_attn_pool = torch.sum(weights * ray_tokens, dim=1)

        ray_mean_pool = torch.mean(ray_tokens, dim=1)

        # Focused near-obstacle pool.
        near_weights = torch.softmax(near_score.transpose(1, 2) * 8.0, dim=1)
        ray_near_pool = torch.sum(near_weights * ray_tokens, dim=1)

        fused = torch.cat(
            [
                path_ray_pool,
                ray_attn_pool,
                ray_near_pool,
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
            scan_max_range=10.0,
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

        if critic_obs.shape[1] > 1209:
            future_path_blocked = critic_obs[:, 1209:1210].float().clamp(0.0, 1.0)
        else:
            future_path_blocked = path_blocked
        future_danger = torch.maximum(dynamic_risk, future_path_blocked)
        return path_blocked, future_danger

    def auxiliary_loss(
        self,
        actor_observations,
        critic_observations,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Training-only auxiliary supervision.

        Head 1:
            predict path blocked now

        Head 2:
            predict future danger =
            max(old TTC/DCA dynamic risk, future_path_blocked_1s)

        Actor input is unchanged.
        """

        policy_obs = select_obs(actor_observations, "policy")
        critic_obs = select_obs(critic_observations, "critic")

        path_blocked_now_target, future_danger_target = self._aux_targets_from_critic_obs(
            critic_obs
        )

        path_blocked_logits, future_danger_logits = self.actor.auxiliary_logits(policy_obs)

        blocked_loss = self._balanced_bce_with_logits(
            path_blocked_logits,
            path_blocked_now_target,
        )

        future_danger_loss = self._balanced_bce_with_logits(
            future_danger_logits,
            future_danger_target,
        )

        aux_loss = blocked_loss + 0.5 * future_danger_loss

        self.aux_call_count += 1

        if self.aux_call_count % 500 == 0:
            with torch.no_grad():
                print(
                    "[AUX FUTURE]",
                    "calls=", self.aux_call_count,
                    "loss=", float(aux_loss.detach().cpu()),
                    "blocked_pred=", float(torch.sigmoid(path_blocked_logits).mean().detach().cpu()),
                    "blocked_target=", float(path_blocked_now_target.mean().detach().cpu()),
                    "future_pred=", float(torch.sigmoid(future_danger_logits).mean().detach().cpu()),
                    "future_target=", float(future_danger_target.mean().detach().cpu()),
                    flush=True,
                )

        info = {
            "aux_loss": float(aux_loss.detach().cpu()),
            "aux_blocked_loss": float(blocked_loss.detach().cpu()),
            "aux_future_danger_loss": float(future_danger_loss.detach().cpu()),
            "aux_blocked_target": float(path_blocked_now_target.mean().detach().cpu()),
            "aux_future_danger_target": float(future_danger_target.mean().detach().cpu()),
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