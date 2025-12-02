"""
Offline SAC prior wrapper for MPPI.

Load an exported SAC checkpoint (e.g., from `train_sac_push_online.py`) and
produce velocity commands compatible with `MPPIisaacPlanner`.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


def default_push_observation(sim, goal_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Build observations for all parallel envs:
    [q(6), qd(6), block pos(3), block ori(4), goal pos(3), goal ori(4)]
    """
    if goal_pose is None:
        goal_pose = torch.tensor([0.7, 0.2, 0.5, 0.0, 0.0, 0.258819, 0.9659258], device=sim.root_state.device)

    q = sim.dof_state[:, 0::2]
    qd = sim.dof_state[:, 1::2]

    block_idx = sim.get_actor_index("obj_to_push")
    block_state = sim.root_state[:, block_idx]
    block_pos = block_state[:, :3]
    block_ori = block_state[:, 3:7]

    goal = goal_pose.to(sim.root_state.device).view(1, -1).repeat(sim.num_envs, 1)
    goal_pos = goal[:, :3]
    goal_ori = goal[:, 3:7]

    return torch.cat([q, qd, block_pos, block_ori, goal_pos, goal_ori], dim=-1)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * act_dim),
        )
        self.act_limit = act_limit

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        return self.act_limit * torch.tanh(pi)

    def act_deterministic(self, obs):
        mu_logstd = self.net(obs)
        mu, _ = torch.chunk(mu_logstd, 2, dim=-1)
        return self.act_limit * torch.tanh(mu)


@dataclass
class PriorConfig:
    checkpoint: str
    obs_dim: int = 26
    act_dim: int = 6
    act_limit: float = 0.2
    hidden: int = 128
    device: Optional[str] = None
    goal_pose: Optional[torch.Tensor] = None
    obs_extractor: Optional[Callable] = None


class OfflineSACPrior:
    """
    Simple wrapper so MPPI can call prior.compute_command(sim).
    """

    def __init__(self, cfg: PriorConfig):
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.obs_extractor = cfg.obs_extractor or default_push_observation

        ckpt = torch.load(cfg.checkpoint, map_location=self.device)
        obs_mean = ckpt.get("obs_mean", None)
        obs_std = ckpt.get("obs_std", None)
        self.obs_mean = obs_mean.to(self.device) if obs_mean is not None else None
        self.obs_std = obs_std.to(self.device) if obs_std is not None else None

        ckpt_cfg = ckpt.get("config", {})
        hidden = ckpt_cfg.get("hidden", cfg.hidden)
        act_limit = ckpt_cfg.get("act_limit", cfg.act_limit)
        obs_dim = ckpt_cfg.get("obs_dim", cfg.obs_dim)
        act_dim = ckpt_cfg.get("act_dim", cfg.act_dim)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden, act_limit).to(self.device)

        state_dict = ckpt["actor_state_dict"]
        # Handle checkpoints saved from SACActor (net.net.*) by flattening one level.
        if any(k.startswith("net.net.") for k in state_dict.keys()):
            state_dict = {k.replace("net.net.", "net."): v for k, v in state_dict.items()}
        self.actor.load_state_dict(state_dict, strict=False)
        self.actor.eval()

        if cfg.goal_pose is None:
            self.goal_pose = None
        else:
            self.goal_pose = torch.as_tensor(cfg.goal_pose, device=self.device, dtype=torch.float32)

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_mean is None or self.obs_std is None:
            return obs
        return (obs - self.obs_mean) / (self.obs_std + 1e-6)

    def compute_command(self, sim) -> torch.Tensor:
        obs = self.obs_extractor(sim, goal_pose=self.goal_pose)
        obs = obs.to(self.device)
        obs = self._norm(obs)
        with torch.no_grad():
            action = self.actor.act_deterministic(obs)
        return action
