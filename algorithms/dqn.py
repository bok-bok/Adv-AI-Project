import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from networks.core import DQNNetwork

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, obs_dim, act_dim, device, **config):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.batch_size = config.get("batch_size", 128)
        self.gamma = config.get("gamma", 0.99)
        self.eps_start = config.get("eps_start", 0.9)
        self.eps_end = config.get("eps_end", 0.01)
        self.eps_decay = config.get("eps_decay", 2500)
        self.tau = config.get("tau", 0.005)
        self.lr = config.get("lr", 3e-4)
        self.memory_capacity = config.get("memory_capacity", 10_000)
        self.grad_clip = config.get("grad_clip", 100)

        self.policy_net = DQNNetwork(obs_dim, act_dim).to(device)
        self.target_net = DQNNetwork(obs_dim, act_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # NOTE: no .eval() on target_net — matches HW4 exactly

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.memory_capacity)
        self.steps_done = 0

    def select_action(self, state):
        """ε-greedy action selection with exponential epsilon decay."""
        sample = random.random()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.act_dim)]], device=self.device, dtype=torch.long
            )

    def select_action_greedy(self, state):
        """Pure greedy action (no exploration) for evaluation."""
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)

    def store(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize(self):
        """Sample a batch from replay memory and perform one gradient update."""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network
        target_sd = self.target_net.state_dict()
        policy_sd = self.policy_net.state_dict()
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * self.tau + target_sd[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_sd)

        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint.get("steps_done", 0)
