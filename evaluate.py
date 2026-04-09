"""
evaluate.py — Load a saved model and run N evaluation episodes.

Usage:
    python evaluate.py --algo dqn --checkpoint ./results/dqn/model.pt --episodes 100
    python evaluate.py --algo ppo --checkpoint ./results/ppo/model.pt --episodes 100
    python evaluate.py --algo dqn --checkpoint ./results/dqn/model.pt --episodes 5 --render
"""

import argparse

import gymnasium as gym
import numpy as np
import torch

from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent

ANGLE_MIN = -2 * np.pi
ANGLE_MAX = 2 * np.pi
OBSERVATION_NOISE_RNG = np.random.default_rng()
NOISE_CHOICES = ["none", "level_1_shift", "level_1", "level_2", "level_3", "level_4"]


def make_env(name: str, render: bool) -> gym.Env:
    render_mode = "human" if render else None
    return gym.make(name, render_mode=render_mode)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wrap_angle(value: float) -> float:
    span = ANGLE_MAX - ANGLE_MIN
    return ((value - ANGLE_MIN) % span) + ANGLE_MIN


def add_observation_noise(state, noise: str):
    noisy_state = np.array(state, dtype=np.float32, copy=True)

    if noise == "level_1_shift":
        mean = 0.1
        standard_deviation = 0.1
        boolean_flip_prob = 0.01

    elif noise == "level_1":
        mean = 0.0
        standard_deviation = 0.1
        boolean_flip_prob = 0.01
    
    elif noise == "level_2":
        mean = 0.0
        standard_deviation = 0.2
        boolean_flip_prob = 0.01

    elif noise == "level_3":
        mean = 0.0
        standard_deviation = 0.5
        boolean_flip_prob = 0.01
    
    elif noise == "level_4":
        mean = 0.0
        standard_deviation = 1.0
        boolean_flip_prob = 0.02
    
    else:
        raise ValueError(f"Unsupported noise mode: {noise}")
    

    noisy_state[:6] += OBSERVATION_NOISE_RNG.normal(loc=mean, scale=standard_deviation, size=6).astype(np.float32)
    noisy_state[4] = wrap_angle(float(noisy_state[4]))

    for idx in (6, 7):
        if OBSERVATION_NOISE_RNG.random() < boolean_flip_prob:
            noisy_state[idx] = 1.0 - noisy_state[idx]

    return noisy_state


def evaluate_dqn(env, agent: DQNAgent, num_episodes: int, noise: str = "none"):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if noise != "none":
            obs = add_observation_noise(obs, noise)
        state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action_greedy(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            if noise != "none":
                obs = add_observation_noise(obs, noise)
            total_reward += reward
            done = terminated or truncated
            if not done:
                state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        rewards.append(total_reward)
        print(f"  Episode {ep + 1:3d}: {total_reward:.1f}")
    return rewards


def evaluate_ppo(env, agent: PPOAgent, num_episodes: int, noise: str = "none"):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if noise != "none":
            obs = add_observation_noise(obs, noise)
        total_reward = 0.0
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = agent.act_deterministic(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            if noise != "none":
                obs = add_observation_noise(obs, noise)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"  Episode {ep + 1:3d}: {total_reward:.1f}")
    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN or PPO model.")
    parser.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("-c", "--checkpoint", default="results/dqn/model.pt", help="Path to saved model .pt file")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--noise", default="none", choices=NOISE_CHOICES, help="Observation noise mode")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    global OBSERVATION_NOISE_RNG

    args = parse_args()
    device = get_device()
    OBSERVATION_NOISE_RNG = np.random.default_rng(args.seed)
    env = make_env(args.env, args.render)
    env.reset(seed=args.seed)

    print(f"Evaluating {args.algo.upper()} checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env} | Episodes: {args.episodes}")
    print("-" * 50)

    if args.algo == "dqn":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        agent = DQNAgent(obs_dim, act_dim, device)
        agent.load(args.checkpoint)
        rewards = evaluate_dqn(env, agent, args.episodes, noise=args.noise)

    else:  # ppo
        agent = PPOAgent(env.observation_space, env.action_space, device)
        agent.load(args.checkpoint)
        rewards = evaluate_ppo(env, agent, args.episodes, noise=args.noise)

    env.close()

    print("-" * 50)
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean reward : {np.mean(rewards):.2f}")
    print(f"  Std  reward : {np.std(rewards):.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")


if __name__ == "__main__":
    main()
