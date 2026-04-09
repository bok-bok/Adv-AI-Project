"""
train.py — CLI runner for DQN and PPO on LunarLander-v3 (or any Gymnasium env).

Usage:
    python train.py --algo dqn --episodes 600 --seed 42
    python train.py --algo ppo --epochs 150 --seed 42
"""

import argparse
import os
import random
from itertools import count

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt

from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent, PPOBuffer
import evaluate as evaluate_utils

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(name: str, seed: int) -> gym.Env:
    env = gym.make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def get_device(algo: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS skipped for both algos: networks are small (DQN: 128x128, PPO: 64x64)
    # and MPS dispatch overhead dominates actual compute at these sizes.
    # PPO rollout also calls step() with batch_size=1 per env step (4000 times),
    # which is worst-case for MPS. CPU wins for both.
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DQN training loop
# ---------------------------------------------------------------------------


def train_dqn(env, agent: DQNAgent, config: dict, save_dir: str, noise: str):
    num_episodes = config["episodes"]
    episode_rewards = []
    noise_suffix = f"_{noise}"

    print(f"[DQN] Training for {num_episodes} episodes on {config['env']} with noise='{noise}'...")

    pbar = tqdm(range(num_episodes), desc="DQN", unit="ep")
    for i_episode in pbar:
        state, _ = env.reset()
        if noise != "none":
            state = evaluate_utils.add_observation_noise(state, noise)
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        total_reward = 0.0

        for _ in count():
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            if noise != "none":
                obs = evaluate_utils.add_observation_noise(obs, noise)
            total_reward += reward
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)

            agent.store(
                state, action, next_state, torch.tensor([reward], dtype=torch.float32, device=agent.device)
            )
            state = next_state
            agent.optimize()

            if done:
                break

        episode_rewards.append(total_reward)
        recent = episode_rewards[-50:]
        pbar.set_postfix({"avg50": f"{np.mean(recent):.1f}", "last": f"{total_reward:.1f}"})

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model{noise_suffix}.pt")
    agent.save(model_path)
    print(f"[DQN] Model saved to {model_path}")

    # Save plot
    _save_plot(
        episode_rewards,
        f"DQN Training Rewards (Noise = {noise})",
        "Episode",
        "Total Reward",
        os.path.join(save_dir, f"training_curve{noise_suffix}.png"),
    )

    return episode_rewards


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------


def train_ppo(env, agent: PPOAgent, config: dict, save_dir: str, noise: str):
    num_epochs = config["epochs"]
    steps_per_epoch = config["steps_per_epoch"]
    obs_dim = env.observation_space.shape[0]
    # For discrete spaces, actions are scalars (act_shape=None); continuous: vector
    act_shape = agent.act_shape
    noise_suffix = f"_{noise}"

    epoch_returns = []
    print(f"[PPO] Training for {num_epochs} epochs ({steps_per_epoch} steps/epoch) on {config['env']} with noise='{noise}'...")

    obs, _ = env.reset()
    if noise != "none":
        obs = evaluate_utils.add_observation_noise(obs, noise)
    ep_ret = 0.0
    ep_len = 0
    episode_returns = []

    pbar = tqdm(range(num_epochs), desc="PPO", unit="epoch")
    for epoch in pbar:
        buf = PPOBuffer(
            obs_dim,
            act_shape,
            steps_per_epoch,
            gamma=agent.gamma,
            lam=agent.lam,
        )

        for t in range(steps_per_epoch):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            a, v, logp = agent.step(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(int(a) if agent.discrete else a)
            if noise != "none":
                next_obs = evaluate_utils.add_observation_noise(next_obs, noise)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated

            buf.store(obs, a, reward, v, logp)
            obs = next_obs

            timeout = ep_len == env.spec.max_episode_steps if env.spec else False
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    # Bootstrap value for truncated trajectory
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    _, v, _ = agent.step(obs_tensor)
                    last_val = v
                else:
                    last_val = 0.0
                    if ep_len > 0:
                        episode_returns.append(ep_ret)

                buf.finish_path(last_val)
                obs, _ = env.reset()
                if noise != "none":
                    obs = evaluate_utils.add_observation_noise(obs, noise)
                ep_ret = 0.0
                ep_len = 0

        update_info = agent.update(buf)

        mean_ret = np.mean(episode_returns) if episode_returns else float("nan")
        epoch_returns.append(mean_ret)
        episode_returns = []

        pbar.set_postfix(
            {
                "ret": f"{mean_ret:.1f}",
                "pi": f"{update_info['loss_pi']:.4f}",
                "vf": f"{update_info['loss_v']:.4f}",
                "kl": f"{update_info['kl']:.4f}",
            }
        )

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model{noise_suffix}.pt")
    agent.save(model_path)
    print(f"[PPO] Model saved to {model_path}")

    # Save plot
    _save_plot(
        epoch_returns,
        f"PPO Training Returns (Noise = {noise})",
        "Epoch",
        "Mean Return",
        os.path.join(save_dir, f"training_curve{noise_suffix}.png"),
    )

    return epoch_returns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plot(data, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data, alpha=0.6, label="Raw")
    if len(data) >= 10:
        window = min(50, len(data) // 5)
        smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(data)), smoothed, label=f"Smoothed (w={window})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"  Plot saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN or PPO on a Gymnasium environment.")
    parser.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("-e", "--episodes", type=int, default=10, help="DQN: number of episodes")
    parser.add_argument("--epochs", type=int, default=300, help="PPO: number of epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=None, help="Directory to save model and plot")
    parser.add_argument("--noise", default="none", choices=evaluate_utils.NOISE_CHOICES, help="Observation noise mode")
    # DQN hyperparams
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-decay", type=int, default=2500)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--dqn-lr", type=float, default=3e-4)
    # PPO hyperparams
    parser.add_argument("--steps-per-epoch", type=int, default=8000)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluate_utils.OBSERVATION_NOISE_RNG = np.random.default_rng(args.seed)

    device = get_device(args.algo)
    print(f"Using device: {device}")

    save_dir = args.save_dir or os.path.join("weights", args.algo)

    env = make_env(args.env, args.seed)

    if args.algo == "dqn":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        agent = DQNAgent(
            obs_dim,
            act_dim,
            device,
            batch_size=args.batch_size,
            gamma=args.gamma,
            eps_decay=args.eps_decay,
            tau=args.tau,
            lr=args.dqn_lr,
        )
        config = dict(env=args.env, episodes=args.episodes)
        train_dqn(env, agent, config, save_dir, args.noise)

    else:  # ppo
        agent = PPOAgent(
            env.observation_space,
            env.action_space,
            device,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            pi_lr=args.pi_lr,
            vf_lr=args.vf_lr,
            target_kl=args.target_kl,
            entropy_coeff=args.entropy_coeff,
            steps_per_epoch=args.steps_per_epoch,
        )
        config = dict(
            env=args.env,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
        train_ppo(env, agent, config, save_dir, args.noise)

    env.close()


if __name__ == "__main__":
    main()
