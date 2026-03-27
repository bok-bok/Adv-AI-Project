import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.core import MLPActorCritic, combined_shape, discount_cumsum


class PPOBuffer:
    """
    Buffer for storing trajectories collected during PPO rollouts.
    Uses GAE-Lambda for advantage estimation.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call at the end of a trajectory (or when buffer is full).
        Computes GAE advantages and reward-to-go returns.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage estimation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Reward-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """Return all stored data as normalized tensors."""
        assert self.ptr == self.max_size

        # Normalize advantages
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOAgent:

    def __init__(self, obs_space, act_space, device, **config):
        self.device = device
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lam", 0.97)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.pi_lr = config.get("pi_lr", 3e-4)
        self.vf_lr = config.get("vf_lr", 1e-3)
        self.train_pi_iters = config.get("train_pi_iters", 80)
        self.train_v_iters = config.get("train_v_iters", 80)
        self.target_kl = config.get("target_kl", 0.01)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.steps_per_epoch = config.get("steps_per_epoch", 4000)
        hidden_sizes = config.get("hidden_sizes", (64, 64))
        activation = config.get("activation", nn.Tanh)

        self.ac = MLPActorCritic(
            obs_space, act_space, hidden_sizes=hidden_sizes, activation=activation
        ).to(device)

        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)

        self.obs_dim = obs_space.shape[0]
        self.discrete = hasattr(act_space, "n")
        # For discrete spaces, actions are scalars; for continuous, they are vectors.
        self.act_shape = None if self.discrete else act_space.shape[0]

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = (
            data["obs"].to(self.device),
            data["act"].to(self.device),
            data["adv"].to(self.device),
            data["logp"].to(self.device),
        )

        act_in = act.long() if self.discrete else act
        pi, logp = self.ac.pi(obs, act_in)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -torch.min(ratio * adv, clip_adv).mean()

        # Entropy bonus
        entropy = pi.entropy().mean()
        loss_pi = loss_pi - self.entropy_coeff * entropy

        approx_kl = (logp_old - logp).mean().item()
        info = dict(kl=approx_kl, entropy=entropy.item())
        return loss_pi, info

    def compute_loss_v(self, data):
        obs, ret = data["obs"].to(self.device), data["ret"].to(self.device)
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self, buf: PPOBuffer):
        data = buf.get()

        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, info = self.compute_loss_pi(data)
            if info["kl"] > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        return dict(loss_pi=loss_pi.item(), loss_v=loss_v.item(), kl=info["kl"])

    def step(self, obs_tensor):
        """Sample action, value, log-prob from current policy."""
        return self.ac.step(obs_tensor.to(self.device))

    def act_deterministic(self, obs_tensor):
        return self.ac.act_deterministic(obs_tensor.to(self.device))

    def save(self, path: str):
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "pi_optimizer": self.pi_optimizer.state_dict(),
                "vf_optimizer": self.vf_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint["ac"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer"])
