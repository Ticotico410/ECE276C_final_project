"""
Online SAC training for the push task using Isaac Gym.

This script:
- builds the same push environment as the MPPI demo,
- collects data on-the-fly (no offline dataset needed),
- trains a small SAC policy,
- exports both a checkpoint usable as MPPI prior and a dataset npz.

Defaults are intentionally small for fast iteration on limited compute.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Important: import isaacgym before torch to avoid import ordering issues.
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper, IsaacGymConfig
from mppiisaac.prior.offline_sac_prior import default_push_observation

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from yaml.loader import SafeLoader
from torch.utils.tensorboard import SummaryWriter
from mppiisaac.utils.push_objective import compute_push_cost

def to_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.actions = torch.zeros((capacity, act_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
        self.ptr = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.next_obs[idx] = next_obs
        self.dones[idx] = done
        self.ptr = int((self.ptr + n) % self.capacity)
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.ptr
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def to_npz(self, path: str):
        max_idx = self.capacity if self.full else self.ptr
        np.savez(
            path,
            obs=self.obs[:max_idx].cpu().numpy(),
            actions=self.actions[:max_idx].cpu().numpy(),
            rewards=self.rewards[:max_idx, 0].cpu().numpy(),
            next_obs=self.next_obs[:max_idx].cpu().numpy(),
            dones=self.dones[:max_idx, 0].cpu().numpy(),
        )


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DoubleQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)


class SACActor(nn.Module):
    """Gaussian policy with tanh squashing, returns action and log-prob."""

    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden)
        self.act_limit = act_limit

    def _dist(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self._dist(obs)
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        logp = (-0.5 * (((pi - mu) / (std + 1e-6)) ** 2 + 2 * torch.log(std + 1e-6) + np.log(2 * np.pi))).sum(
            dim=-1, keepdim=True
        )
        pi_tanh = torch.tanh(pi)
        logp -= torch.log(1 - pi_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return self.act_limit * pi_tanh, logp

    def act_deterministic(self, obs):
        mu, _ = self._dist(obs)
        return self.act_limit * torch.tanh(mu)


class ObsNormalizer:
    """Track running mean/std for observation normalization."""

    def __init__(self, obs_dim, device):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = torch.tensor(1e-4, device=device)

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-6)


@dataclass
class TrainConfig:
    actor_names: List[str] = None
    isaac_conf_path: str = None
    goal_pose: List[float] = None
    num_envs: int = 32
    horizon: int = 120
    total_steps: int = 10000
    warmup_steps: int = 1000
    update_after: int = 1000
    update_every: int = 50
    gradient_steps: int = 50
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.01
    lr: float = 1e-4
    hidden: int = 128
    obs_dim: int = 26
    act_dim: int = 6
    act_limit: float = 0.2
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 200000
    log_every: int = 200
    export_ckpt: str = "outputs/offline_sac_prior.pt"
    export_dataset: str = "outputs/online_push_dataset.npz"
    log_dir: str = "outputs/tb/train_sac_push_online"
    seed: int = 1


def make_sim(cfg: TrainConfig) -> IsaacGymWrapper:
    actors = []
    for actor_name in cfg.actor_names:
        with open(os.path.join(cfg.isaac_conf_path, "actors", f"{actor_name}.yaml")) as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))

    isaac_cfg = IsaacGymConfig(dt=0.04, substeps=1)
    sim = IsaacGymWrapper(
        cfg=isaac_cfg,
        actors=actors,
        init_positions=[[0.0, 0.0, 0.0]],
        num_envs=cfg.num_envs,
        viewer=False,
    )
    return sim


def add_objects(sim: IsaacGymWrapper):
    table_size = [0.8, 1.0, 0.005]
    table_pos = [0.5, 0.0, table_size[-1] / 2]
    obj = [0.162, 0.086, 0.068, 0.300, 0.250, 0.400, 0.000]
    goal_pose = [0.7, 0.2, 0.5, 0.0, 0.0, 0.258819, 0.9659258]

    obj_size = [obj[0], obj[1], obj[2]]
    obj_init_pos = [obj[5], obj[6], table_size[-1] + obj[2] / 2]
    obj_init_ori = [0.0, 0.0, 0.0, 1.0]

    goal_size = [obj[0], obj[1], 0.005]
    goal_pos = [goal_pose[0], goal_pose[1], table_size[-1]]
    goal_ori = [goal_pose[3], goal_pose[4], goal_pose[5], goal_pose[6]]

    objects = [
        {
            "type": "box",
            "name": "obj_to_push",
            "size": obj_size,
            "init_pos": obj_init_pos,
            "init_ori": obj_init_ori,
            "mass": obj[4],
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 0.8],
            "friction": obj[3],
        },
        {
            "type": "box",
            "name": "table",
            "size": table_size,
            "init_pos": table_pos,
            "fixed": True,
            "handle": None,
            "color": [255 / 255, 120 / 255, 57 / 255],
            "friction": obj[3],
        },
        {
            "type": "box",
            "name": "goal",
            "size": goal_size,
            "init_pos": goal_pos,
            "init_ori": goal_ori,
            "fixed": True,
            "color": [119 / 255, 221 / 255, 119 / 255],
            "handle": None,
            "collision": False,
        },
    ]
    sim.add_to_envs(objects)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.lerp_(sp.data, tau)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(os.path.dirname(cfg.export_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.export_dataset), exist_ok=True)

    sim = make_sim(cfg)
    add_objects(sim)
    sim.save_root_state()

    goal_pose = torch.tensor(
        cfg.goal_pose or [0.7, 0.2, 0.5, 0.0, 0.0, 0.258819, 0.9659258],
        device="cuda:0",
    )

    actor = SACActor(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.act_limit).to(cfg.device)
    critic = DoubleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(cfg.device)
    critic_targ = DoubleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(cfg.device)
    critic_targ.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    log_alpha = torch.tensor(np.log(0.2), device=cfg.device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr)
    target_entropy = -float(cfg.act_dim)

    buf = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, cfg.act_dim, cfg.device)
    obs_norm = ObsNormalizer(cfg.obs_dim, cfg.device)
    writer = SummaryWriter(cfg.log_dir) if cfg.log_dir else None

    obs = default_push_observation(sim, goal_pose=goal_pose)
    obs_norm.update(obs)
    episode_step = torch.zeros(cfg.num_envs, device=cfg.device, dtype=torch.long)
    last_critic_loss = None
    last_actor_loss = None
    last_alpha_loss = None

    start = time.time()
    for global_step in range(1, cfg.total_steps + 1):
        with torch.no_grad():
            if global_step < cfg.warmup_steps:
                action = cfg.act_limit * (2 * torch.rand((cfg.num_envs, cfg.act_dim), device=cfg.device) - 1)
            else:
                normed_obs = obs_norm.normalize(obs)
                action, _ = actor.sample(normed_obs)

            sim.apply_robot_cmd_velocity(action)
            sim.step()

            next_obs = default_push_observation(sim, goal_pose=goal_pose)
            obs_norm.update(next_obs)
            cost = compute_push_cost(sim, goal_pose)
            reward = -cost.unsqueeze(-1)

            success = cost < 0.1
            episode_step += 1
            done = (episode_step >= cfg.horizon) | success
            done_tensor = done.float().unsqueeze(-1)

            buf.add(obs, action, reward, next_obs, done_tensor)

            # Reset finished envs
            if done.any():
                sim.reset_root_state()
                episode_step[done] = 0
                sim.gym.refresh_actor_root_state_tensor(sim.sim)
                sim.gym.refresh_dof_state_tensor(sim.sim)
                sim.gym.refresh_rigid_body_state_tensor(sim.sim)

            obs = next_obs

        if global_step >= cfg.update_after and global_step % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                b_obs, b_act, b_rew, b_nobs, b_done = buf.sample(cfg.batch_size)
                b_obs_n = obs_norm.normalize(b_obs)
                b_nobs_n = obs_norm.normalize(b_nobs)
                with torch.no_grad():
                    pi_next, logp_next = actor.sample(b_nobs_n)
                    q1_t, q2_t = critic_targ(b_nobs_n, pi_next)
                    q_targ = torch.min(q1_t, q2_t) - log_alpha.exp() * logp_next
                    backup = b_rew + cfg.gamma * (1 - b_done) * q_targ

                q1, q2 = critic(b_obs_n, b_act)
                critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                last_critic_loss = critic_loss

                pi, logp_pi = actor.sample(b_obs_n)
                q1_pi, q2_pi = critic(b_obs_n, pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (log_alpha.exp() * logp_pi - q_pi).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                last_actor_loss = actor_loss

                alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                last_alpha_loss = alpha_loss

                soft_update(critic_targ, critic, cfg.tau)

        if global_step % cfg.log_every == 0 or global_step == 1:
            fps = global_step / (time.time() - start)
            print(
                f"[step {global_step}] buffer={buf.ptr} fps={fps:.1f} "
                f"mean_cost={cost.mean().item():.3f}"
            )
            if writer:
                writer.add_scalar("train/fps", fps, global_step)
                writer.add_scalar("train/mean_cost", cost.mean().item(), global_step)
                if last_critic_loss is not None:
                    writer.add_scalar("loss/critic", last_critic_loss.item(), global_step)
                if last_actor_loss is not None:
                    writer.add_scalar("loss/actor", last_actor_loss.item(), global_step)
                if last_alpha_loss is not None:
                    writer.add_scalar("loss/alpha", last_alpha_loss.item(), global_step)
                writer.add_scalar("alpha/value", log_alpha.exp().item(), global_step)

    # Export checkpoint compatible with OfflineSACPrior
    max_idx = buf.capacity if buf.full else buf.ptr
    obs_mean = buf.obs[:max_idx].mean(dim=0).cpu()
    obs_std = buf.obs[:max_idx].std(dim=0).cpu()
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_mean": obs_norm.mean.cpu(),
            "obs_std": torch.sqrt(obs_norm.var).cpu(),
            "config": {
                "hidden": cfg.hidden,
                "act_limit": cfg.act_limit,
                "obs_dim": cfg.obs_dim,
                "act_dim": cfg.act_dim,
            },
        },
        cfg.export_ckpt,
    )
    print(f"Saved prior checkpoint to {cfg.export_ckpt}")
    buf.to_npz(cfg.export_dataset)
    print(f"Saved dataset to {cfg.export_dataset}")
    if writer:
        writer.flush()
        writer.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--total_steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    parser.add_argument("--export_ckpt", type=str, default=TrainConfig.export_ckpt)
    parser.add_argument("--export_dataset", type=str, default=TrainConfig.export_dataset)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log_dir", type=str, default=TrainConfig.log_dir)
    args = parser.parse_args()

    cfg = TrainConfig(
        actor_names=["xarm6_gripper"],
        isaac_conf_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "conf")),
    )
    cfg.num_envs = args.num_envs
    cfg.total_steps = args.total_steps
    cfg.horizon = args.horizon
    cfg.export_ckpt = args.export_ckpt
    cfg.export_dataset = args.export_dataset
    cfg.device = args.device
    cfg.log_dir = args.log_dir
    return cfg


if __name__ == "__main__":
    train(parse_args())
