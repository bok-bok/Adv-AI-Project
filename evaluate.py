"""
evaluate.py — Load a saved model and run N evaluation episodes.

Usage:
    python evaluate.py --algo dqn --episodes 100 --render none
    python evaluate.py --algo ppo --episodes 100 --render human
    python evaluate.py --algo dqn --episodes 5 --render mp4
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent

ANGLE_MIN = -2 * np.pi
ANGLE_MAX = 2 * np.pi
OBSERVATION_NOISE_RNG = np.random.default_rng()
NOISE_CHOICES = ["level_05", "none", "level_1_shift", "level_1", "level_2", "level_3", "level_4"]
RENDER_CHOICES = ["none", "human", "mp4"]
DEFAULT_VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = Path("videos")


def make_env(name: str, render: str) -> gym.Env:
    if render == "human":
        render_mode = "human"
    elif render == "mp4":
        render_mode = "rgb_array"
    else:
        render_mode = None
    return gym.make(name, render_mode=render_mode)


def video_output_path(algo: str, env_name: str, noise: str) -> Path:
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEO_OUTPUT_DIR / f"{algo}_{env_name}_{noise}.mp4"


class VideoRecorder:
    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = fps
        self.frames = []

    def add_frame(self, frame) -> None:
        if frame is not None:
            self.frames.append(np.asarray(frame))

    def save(self) -> None:
        if not self.frames:
            return
        with  imageio.get_writer(self.path, fps=self.fps, format="FFMPEG") as writer:
            for frame in self.frames:
                writer.append_data(frame)


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

    if noise == "level_05":
        mean = 0.0
        standard_deviation = 0.01
        boolean_flip_prob = 0.00

    elif noise == "level_1_shift":
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


def evaluate_dqn(env, agent: DQNAgent, num_episodes: int, noise: str = "none", recorder: VideoRecorder | None = None):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if recorder is not None:
            recorder.add_frame(env.render())
        if noise != "none":
            obs = add_observation_noise(obs, noise)
        state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action_greedy(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            if recorder is not None:
                recorder.add_frame(env.render())
            if noise != "none":
                obs = add_observation_noise(obs, noise)
            total_reward += reward
            done = terminated or truncated
            if not done:
                state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        rewards.append(total_reward)
        print(f"  Episode {ep + 1:3d}: {total_reward:.1f}")
    return rewards


def evaluate_ppo(env, agent: PPOAgent, num_episodes: int, noise: str = "none", recorder: VideoRecorder | None = None):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if recorder is not None:
            recorder.add_frame(env.render())
        if noise != "none":
            obs = add_observation_noise(obs, noise)
        total_reward = 0.0
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = agent.act_deterministic(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            if recorder is not None:
                recorder.add_frame(env.render())
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
    # parser.add_argument("-c", "--checkpoint", default="weights/dqn/model_none.pt", help="Path to saved model .pt file")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", default="none", choices=RENDER_CHOICES, help="Render mode")
    parser.add_argument("--noise", default="level_4", choices=NOISE_CHOICES, help="Observation noise mode")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    global OBSERVATION_NOISE_RNG

    args = parse_args()

    checkpoint = f"weights/{args.algo}/model_{args.noise}.pt"
    # checkpoint = f"weights/{args.algo}/model_level_1.pt"
    # checkpoint = f"weights/{args.algo}/model_none.pt"

    device = get_device()
    OBSERVATION_NOISE_RNG = np.random.default_rng(args.seed)
    env = make_env(args.env, args.render)
    env.reset(seed=args.seed)
    recorder = None
    video_path = None

    if args.render == "mp4":
        video_path = video_output_path(args.algo, args.env, args.noise)
        recorder = VideoRecorder(video_path, fps=DEFAULT_VIDEO_FPS)

    print(f"Evaluating {args.algo.upper()} checkpoint: {checkpoint}")
    print(f"Environment: {args.env} | Episodes: {args.episodes} | Render: {args.render}")
    print("-" * 50)

    if args.algo == "dqn":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        agent = DQNAgent(obs_dim, act_dim, device)
        agent.load(checkpoint)
        rewards = evaluate_dqn(env, agent, args.episodes, noise=args.noise, recorder=recorder)

    else:  # ppo
        agent = PPOAgent(env.observation_space, env.action_space, device)
        agent.load(checkpoint)
        rewards = evaluate_ppo(env, agent, args.episodes, noise=args.noise, recorder=recorder)

    env.close()
    if recorder is not None:
        recorder.save()

    print("-" * 50)
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean reward : {np.mean(rewards):.2f}")
    print(f"  Std  reward : {np.std(rewards):.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")
    if video_path is not None:
        print(f"  Video saved : {video_path}")


if __name__ == "__main__":
    main()
