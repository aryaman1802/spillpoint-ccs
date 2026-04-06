"""
DDQN for Optimal Stopping CO2 Injection — Region 2 only.

Features:
  1. Random-action baseline collected at start and plotted
     (cumulative reward trajectories + histogram).
  2. DDQN with 4 actions: [stop, inject+measure, inject, measure].
  3. Loss updates EVERY step (never freezes).
  4. One evaluation episode after every training episode.
  5. All on a single fixed-geology seed.

Usage:
    python ddqn_optimal_stopping.py --episodes 3000 --seed 42 --env_seed 42
"""

import os
import sys
import random
import argparse
from typing import List, Optional, Tuple
from collections import deque, namedtuple
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

print("Imports done!")

# ============================================================================
# Output directories
# ============================================================================
OUTPUT_DIR_PICS = os.environ.get("OUTPUT_DIR_PICS", "output_pics/ddqn_optimal_stopping")
OUTPUT_DIR_MODELS = os.environ.get("OUTPUT_DIR_MODELS", "output_models/ddqn_optimal_stopping")
OUTPUT_DIR_RESULTS = os.environ.get("OUTPUT_DIR_RESULTS", "output_results/ddqn_optimal_stopping")

# ============================================================================
# Default config
# ============================================================================
CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "buffer_size": 50_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 2000,
    "tau": 0.005,
    "hard_update_freq": int(1e12),
    "lr_scheduler": "cosine",
    "lr_decay_steps": 2500,
    "gamma": 0.99,
    "grad_clip": 1.0,
    "weight_decay": 1e-5,
    # Network architecture
    "gravity_channels": [32, 64],
    "fc_hidden": [128, 64],
    "dropout_rate": 0.0,
}

ENV_CONFIG = {
    "fixed_seed": 42,
    "nx_sensors": 30,
    "nz_sensors": 30,
    "injection_rate_m3": 999_999.99,
    "max_steps": 500,
}

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

N_RANDOM_BASELINE = 300


# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Network — Conv1d on gravity + concat scalars
# ============================================================================
class GravityFeatureExtractor(nn.Module):
    """Conv1d encoder for the 30-element gravity signal."""

    def __init__(self, input_length: int = 30,
                 hidden_channels: List[int] = [32, 64]):
        super().__init__()
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(hidden_channels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3,
                                    stride=2 if i > 0 else 1, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_size = in_ch

    def forward(self, x):
        # x: (B, gravity_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, L)
        x = self.conv(x)
        x = self.pool(x)                # (B, C, 1)
        return x.squeeze(-1)            # (B, C)


class DDQNNetwork(nn.Module):
    """
    DDQN Q-network.

    Splits the 32-dim observation into:
      - gravity (first 30 dims) → Conv1d feature extractor
      - scalars (last 2 dims)   → concatenated after extraction

    Then FC head → Q-values for 4 actions.
    """

    def __init__(self, gravity_size: int = 30, scalar_size: int = 2,
                 n_actions: int = 4,
                 gravity_channels: List[int] = [32, 64],
                 fc_hidden: List[int] = [128, 64]):
        super().__init__()
        self.gravity_size = gravity_size
        self.scalar_size = scalar_size

        self.gravity_encoder = GravityFeatureExtractor(
            gravity_size, gravity_channels
        )
        combined_dim = self.gravity_encoder.output_size + scalar_size

        fc_layers = []
        in_dim = combined_dim
        for h in fc_hidden:
            fc_layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            in_dim = h
        self.fc = nn.Sequential(*fc_layers)

        self.q_head = nn.Linear(fc_hidden[-1], n_actions)
        nn.init.uniform_(self.q_head.weight, -0.003, 0.003)
        nn.init.uniform_(self.q_head.bias, -0.003, 0.003)

    def forward(self, obs):
        gravity = obs[:, : self.gravity_size]
        scalars = obs[:, self.gravity_size :]
        grav_feat = self.gravity_encoder(gravity)
        combined = torch.cat([grav_feat, scalars], dim=1)
        return self.q_head(self.fc(combined))


# ============================================================================
# Random Baseline — cumulative reward trajectories
# ============================================================================
def run_random_baseline(env, num_episodes: int = 300) -> dict:
    """
    Run purely random episodes.  Record step-by-step cumulative reward
    so we can plot trajectories (cum reward vs step).
    """
    print(f"\nRunning {num_episodes} random-action baseline episodes ...")
    all_cum_rewards: List[List[float]] = []      # per-episode cum-reward curves
    final_rewards, final_trapped, final_lengths = [], [], []
    action_counts = np.zeros(4, dtype=int)

    for ep in range(num_episodes):
        state, _ = env.reset()
        cum_reward = 0.0
        trajectory = [0.0]
        done = False
        while not done:
            action = env.action_space.sample()
            action_counts[action] += 1
            state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            cum_reward += reward
            trajectory.append(cum_reward)

        all_cum_rewards.append(trajectory)
        final_rewards.append(cum_reward)
        final_trapped.append(info.get("cumulative_trapped_r2_m3", 0.0) / 1e6)
        final_lengths.append(len(trajectory) - 1)

    results = {
        "trajectories": all_cum_rewards,
        "final_rewards": final_rewards,
        "final_trapped": final_trapped,
        "final_lengths": final_lengths,
        "action_distribution": action_counts.tolist(),
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "mean_trapped": float(np.mean(final_trapped)),
        "mean_length": float(np.mean(final_lengths)),
    }
    print(f"  Random baseline: reward = {results['mean_reward']:.4f} "
          f"± {results['std_reward']:.4f} M$,  "
          f"trapped = {results['mean_trapped']:.4f} Mm³,  "
          f"mean length = {results['mean_length']:.1f}")
    print(f"  Action distribution: "
          f"stop={action_counts[0]}, inj+meas={action_counts[1]}, "
          f"inj={action_counts[2]}, meas={action_counts[3]}")
    return results


def plot_random_baseline(baseline: dict, save_path: str):
    """
    Two-panel figure:
      Left  – Cumulative reward vs step for every random episode
      Right – Histogram of final cumulative rewards
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random-Action Baseline", fontsize=14, fontweight="bold")

    # ---- Left: trajectories ----
    for traj in baseline["trajectories"]:
        ax1.plot(traj, alpha=0.15, color="grey", linewidth=0.5)

    # Mean trajectory (pad shorter trajectories with their final value)
    max_len = max(len(t) for t in baseline["trajectories"])
    padded = np.full((len(baseline["trajectories"]), max_len), np.nan)
    for i, traj in enumerate(baseline["trajectories"]):
        padded[i, : len(traj)] = traj
        padded[i, len(traj) :] = traj[-1]          # hold final value

    mean_traj = np.nanmean(padded, axis=0)
    p25 = np.nanpercentile(padded, 25, axis=0)
    p75 = np.nanpercentile(padded, 75, axis=0)
    steps = np.arange(max_len)
    ax1.plot(steps, mean_traj, color="red", linewidth=2, label="Mean")
    ax1.fill_between(steps, p25, p75, color="red", alpha=0.15, label="25-75 %ile")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Reward (M$)")
    ax1.set_title("Cumulative Reward vs Random Action Sequences")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Right: histogram ----
    ax2.hist(baseline["final_rewards"], bins=40, color="steelblue",
             edgecolor="black", alpha=0.8)
    ax2.axvline(baseline["mean_reward"], color="red", linewidth=2,
                label=f"Mean = {baseline['mean_reward']:.2f} M$")
    ax2.set_xlabel("Final Cumulative Reward (M$)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Final Rewards ({len(baseline['final_rewards'])} episodes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Random baseline plot saved to: {save_path}")


# ============================================================================
# DDQN Agent
# ============================================================================
class DDQNAgent:
    def __init__(self, env, config: dict):
        self.env = env
        self.config = config
        self.n_actions = env.action_space.n  # 4

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]
        self.hard_update_freq = config["hard_update_freq"]
        self.learning_rate = config["learning_rate"]
        self.buffer_size = config["buffer_size"]
        self.grad_clip = config["grad_clip"]

        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_episodes = config["epsilon_decay_episodes"]
        self.epsilon = self.epsilon_start

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Observation structure
        obs, _ = env.reset()
        obs_size = obs.shape[0]       # 32
        gravity_size = env.nx_sensors  # 30
        scalar_size = obs_size - gravity_size  # 2

        self.policy_net = DDQNNetwork(
            gravity_size=gravity_size,
            scalar_size=scalar_size,
            n_actions=self.n_actions,
            gravity_channels=config["gravity_channels"],
            fc_hidden=config["fc_hidden"],
        ).to(self.device)

        self.target_net = DDQNNetwork(
            gravity_size=gravity_size,
            scalar_size=scalar_size,
            n_actions=self.n_actions,
            gravity_channels=config["gravity_channels"],
            fc_hidden=config["fc_hidden"],
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get("weight_decay", 1e-5),
        )

        lr_sched = config.get("lr_scheduler", "cosine")
        if lr_sched == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.get("lr_decay_steps", 2500)
            )
        elif lr_sched == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get("lr_decay_steps", 1000),
                gamma=config.get("lr_decay_gamma", 0.5),
            )
        else:
            self.scheduler = None

        self.buffer = ReplayBuffer(self.buffer_size)

        # ---- Histories ----
        self.train_history = {
            "rewards": [], "trapped_co2": [], "episode_lengths": [],
            "losses": [], "epsilons": [], "num_measurements": [],
            "has_spilled": [],
        }
        self.eval_history = {
            "rewards": [], "trapped_co2": [], "episode_lengths": [],
            "num_measurements": [], "has_spilled": [],
        }
        self.episode_actions: List[List[int]] = []
        self.total_steps = 0
        self.episode_count = 0

    # ------------------------------------------------------------------
    def get_epsilon(self, episode: int) -> float:
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            episode / self.epsilon_decay_episodes
        )

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            self.policy_net.eval()
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            self.policy_net.train()
            return q.argmax(dim=1).item()

    # ------------------------------------------------------------------
    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = Transition(*zip(*self.buffer.sample(self.batch_size)))
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        with torch.no_grad():
            self.policy_net.eval()
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            self.policy_net.train()
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft target update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        return loss.item()

    # ------------------------------------------------------------------
    def _run_eval_episode(self) -> dict:
        state, _ = self.env.reset()
        ep_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = self.select_action(state, training=False)
            state, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1
        return {
            "reward": ep_reward,
            "trapped": info.get("cumulative_trapped_r2_m3", 0.0) / 1e6,
            "length": steps,
            "num_measurements": info.get("num_measurements", 0),
            "has_spilled": info.get("has_spilled", False),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, num_episodes: int = 3000, print_freq: int = 100) -> dict:
        print(f"\n{'='*60}")
        print("DDQN Optimal Stopping — Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Actions: 4 (stop / inject+meas / inject / meas)")
        print(f"Epsilon: {self.epsilon_start} → {self.epsilon_end} over "
              f"{self.epsilon_decay_episodes} eps")
        print(f"Total episodes: {num_episodes}")
        print(f"{'='*60}\n")

        start_time = time.time()
        best_avg_reward = -float("inf")
        best_episode = 0
        reward_window = deque(maxlen=100)

        for episode in tqdm(range(num_episodes), desc="Training"):
            self.epsilon = self.get_epsilon(episode)

            # ============ TRAINING EPISODE ============
            state, _ = self.env.reset()
            ep_reward = 0.0
            ep_losses = []
            ep_actions = []
            done = False
            step_count = 0

            while not done:
                action = self.select_action(state, training=True)
                ep_actions.append(action)
                next_state, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                self.buffer.push(state, action, reward, next_state, float(done))

                loss = self.update()
                if loss is not None:
                    ep_losses.append(loss)

                state = next_state
                ep_reward += reward
                self.total_steps += 1
                step_count += 1

            if self.scheduler is not None:
                self.scheduler.step()

            trapped_mm3 = info.get("cumulative_trapped_r2_m3", 0.0) / 1e6
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0

            self.train_history["rewards"].append(ep_reward)
            self.train_history["trapped_co2"].append(trapped_mm3)
            self.train_history["episode_lengths"].append(step_count)
            self.train_history["losses"].append(avg_loss)
            self.train_history["epsilons"].append(self.epsilon)
            self.train_history["num_measurements"].append(
                info.get("num_measurements", 0))
            self.train_history["has_spilled"].append(
                info.get("has_spilled", False))
            self.episode_actions.append(ep_actions)

            reward_window.append(ep_reward)
            avg_reward = np.mean(reward_window)
            self.episode_count += 1

            if avg_reward > best_avg_reward and episode > 100:
                best_avg_reward = avg_reward
                best_episode = episode

            # ============ EVALUATION EPISODE ============
            eval_res = self._run_eval_episode()
            self.eval_history["rewards"].append(eval_res["reward"])
            self.eval_history["trapped_co2"].append(eval_res["trapped"])
            self.eval_history["episode_lengths"].append(eval_res["length"])
            self.eval_history["num_measurements"].append(
                eval_res["num_measurements"])
            self.eval_history["has_spilled"].append(eval_res["has_spilled"])

            # ============ LOGGING ============
            if (episode + 1) % print_freq == 0:
                avg_r = np.mean(self.train_history["rewards"][-100:])
                avg_t = np.mean(self.train_history["trapped_co2"][-100:])
                avg_l = np.mean(self.train_history["episode_lengths"][-100:])
                avg_er = np.mean(self.eval_history["rewards"][-100:])
                avg_et = np.mean(self.eval_history["trapped_co2"][-100:])
                elapsed = time.time() - start_time
                remaining = (num_episodes - episode - 1) / (
                    (episode + 1) / elapsed)
                phase = "EXPLOIT" if self.epsilon <= self.epsilon_end + 1e-9 \
                    else "EXPLORE"
                print(
                    f"\nEp {episode+1}/{num_episodes} [{phase}]: "
                    f"Train R={avg_r:.2f} M$, Trapped={avg_t:.4f} Mm³, "
                    f"Len={avg_l:.0f}, Loss={avg_loss:.6f}, ε={self.epsilon:.4f}"
                )
                print(
                    f"  Eval R={avg_er:.2f} M$, Eval Trapped={avg_et:.4f} Mm³"
                )
                print(
                    f"  Time: {elapsed/60:.1f} min elapsed, "
                    f"~{remaining/60:.1f} min remaining"
                )

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!  Best avg reward: {best_avg_reward:.2f} M$ "
              f"(ep {best_episode})")
        print(f"Total steps: {self.total_steps},  "
              f"Time: {total_time/60:.1f} min")
        print(f"{'='*60}\n")

        return {
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "episode_actions": self.episode_actions,
            "best_avg_reward": best_avg_reward,
            "best_episode": best_episode,
            "total_steps": self.total_steps,
            "total_time_seconds": total_time,
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------
    def evaluate(self, num_episodes: int = 20) -> dict:
        rewards, trapped, lengths, meas_counts, spill_flags = [], [], [], [], []
        for _ in range(num_episodes):
            r = self._run_eval_episode()
            rewards.append(r["reward"])
            trapped.append(r["trapped"])
            lengths.append(r["length"])
            meas_counts.append(r["num_measurements"])
            spill_flags.append(r["has_spilled"])

        results = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_trapped": float(np.mean(trapped)),
            "std_trapped": float(np.std(trapped)),
            "mean_length": float(np.mean(lengths)),
            "spill_rate": float(np.mean(spill_flags)),
            "mean_measurements": float(np.mean(meas_counts)),
        }
        print(f"Final evaluation ({num_episodes} eps):")
        print(f"  Reward:  {results['mean_reward']:.2f} ± "
              f"{results['std_reward']:.2f} M$")
        print(f"  Trapped: {results['mean_trapped']:.4f} ± "
              f"{results['std_trapped']:.4f} Mm³")
        print(f"  Spill rate: {results['spill_rate']:.1%},  "
              f"Avg measurements: {results['mean_measurements']:.1f}")
        return results

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "total_steps": self.total_steps,
        }, path)
        print(f"Model saved to: {path}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_training(self, save_path: str,
                      random_baseline: Optional[dict] = None):
        """
        9-panel figure:
          Row 0: Train Reward | Train Trapped | Episode Length
          Row 1: Loss         | Epsilon       | Action Distribution
          Row 2: Eval Reward  | Eval Trapped  | Measurements & Spills
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(
            "DDQN Optimal Stopping — Region 2 Injection",
            fontsize=14, fontweight="bold",
        )

        eps_arr = np.arange(len(self.train_history["rewards"]))
        w = 100
        decay_ep = self.epsilon_decay_episodes

        def sm(data, window=w):
            if len(data) < window:
                return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode="valid")

        random_mean_r = random_baseline["mean_reward"] if random_baseline else None
        random_mean_t = random_baseline["mean_trapped"] if random_baseline else None

        # ---- Row 0 Col 0: Train Reward ----
        ax = axes[0, 0]
        r = np.array(self.train_history["rewards"])
        ax.plot(eps_arr, r, alpha=0.25, color="tab:blue", lw=0.5)
        if len(r) >= w:
            ax.plot(np.arange(w - 1, len(r)), sm(r),
                    color="tab:orange", lw=2, label="100-ep avg")
        if random_mean_r is not None:
            ax.axhline(random_mean_r, color="grey", ls=":", lw=1.5,
                       label=f"Random mean")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward (M$)")
        ax.set_title("Training Reward"); ax.legend(); ax.grid(True, alpha=0.3)

        # ---- Row 0 Col 1: Train Trapped ----
        ax = axes[0, 1]
        t = np.array(self.train_history["trapped_co2"])
        ax.plot(eps_arr, t, alpha=0.25, color="tab:green", lw=0.5)
        if len(t) >= w:
            ax.plot(np.arange(w - 1, len(t)), sm(t),
                    color="tab:red", lw=2, label="100-ep avg")
        if random_mean_t is not None:
            ax.axhline(random_mean_t, color="grey", ls=":", lw=1.5,
                       label="Random mean")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Trapped R2 (Mm³)")
        ax.set_title("Training Trapped CO2"); ax.legend(); ax.grid(True, alpha=0.3)

        # ---- Row 0 Col 2: Episode Length ----
        ax = axes[0, 2]
        l_arr = np.array(self.train_history["episode_lengths"])
        ax.plot(eps_arr, l_arr, alpha=0.25, color="tab:purple", lw=0.5)
        if len(l_arr) >= w:
            ax.plot(np.arange(w - 1, len(l_arr)), sm(l_arr),
                    color="tab:orange", lw=2, label="100-ep avg")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
        ax.set_title("Episode Length"); ax.legend(); ax.grid(True, alpha=0.3)

        # ---- Row 1 Col 0: Loss ----
        ax = axes[1, 0]
        loss = np.array(self.train_history["losses"])
        ax.plot(eps_arr, loss, alpha=0.25, color="tab:red", lw=0.5)
        if len(loss) >= w:
            ax.plot(np.arange(w - 1, len(loss)), sm(loss),
                    color="tab:blue", lw=2, label="100-ep avg")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

        # ---- Row 1 Col 1: Epsilon ----
        ax = axes[1, 1]
        ax.plot(eps_arr, self.train_history["epsilons"],
                color="tab:brown", lw=2)
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6,
                   label=f"ε→{self.epsilon_end} @ ep {decay_ep}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon Schedule"); ax.legend(); ax.grid(True, alpha=0.3)

        # ---- Row 1 Col 2: Action Distribution ----
        ax = axes[1, 2]
        action_counts = np.zeros((len(self.episode_actions), 4))
        for i, ep_acts in enumerate(self.episode_actions):
            for a in ep_acts:
                action_counts[i, a] += 1
        row_sums = action_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        action_freq = action_counts / row_sums
        a_w = 50
        labels = ["Stop", "Inj+Meas", "Inject", "Measure"]
        colors = ["red", "blue", "green", "orange"]
        for a_idx in range(4):
            if len(action_freq) >= a_w:
                s = np.convolve(action_freq[:, a_idx],
                                np.ones(a_w) / a_w, mode="valid")
                ax.plot(np.arange(a_w - 1, len(action_freq)), s,
                        lw=1.5, label=labels[a_idx], color=colors[a_idx])
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Frequency")
        ax.set_title("Action Distribution"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ---- Row 2 Col 0: Eval Reward ----
        ax = axes[2, 0]
        er = np.array(self.eval_history["rewards"])
        ax.plot(er, alpha=0.3, color="tab:cyan", lw=0.5)
        if len(er) >= w:
            ax.plot(np.arange(w - 1, len(er)), sm(er),
                    color="darkblue", lw=2, label="100-ep avg")
        if random_mean_r is not None:
            ax.axhline(random_mean_r, color="grey", ls=":", lw=1.5,
                       label="Random mean")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward (M$)")
        ax.set_title("Eval Reward (greedy)"); ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Row 2 Col 1: Eval Trapped ----
        ax = axes[2, 1]
        et = np.array(self.eval_history["trapped_co2"])
        ax.plot(et, alpha=0.3, color="tab:olive", lw=0.5)
        if len(et) >= w:
            ax.plot(np.arange(w - 1, len(et)), sm(et),
                    color="darkgreen", lw=2, label="100-ep avg")
        if random_mean_t is not None:
            ax.axhline(random_mean_t, color="grey", ls=":", lw=1.5,
                       label="Random mean")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Trapped R2 (Mm³)")
        ax.set_title("Eval Trapped (greedy)"); ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Row 2 Col 2: Measurements & Spills ----
        ax = axes[2, 2]
        meas = np.array(self.eval_history["num_measurements"], dtype=float)
        spill = np.array(self.eval_history["has_spilled"], dtype=float)
        if len(meas) >= w:
            ax.plot(np.arange(w - 1, len(meas)), sm(meas),
                    color="blue", lw=2, label="Measurements (100-avg)")
        if len(spill) >= w:
            ax2 = ax.twinx()
            ax2.plot(np.arange(w - 1, len(spill)), sm(spill) * 100,
                     color="red", lw=2, label="Spill rate % (100-avg)")
            ax2.set_ylabel("Spill rate (%)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper left")
        ax.axvline(decay_ep, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Episode"); ax.set_ylabel("# Measurements", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.set_title("Eval: Measurements & Spillover")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training plot saved to: {save_path}")


# ============================================================================
# Utility
# ============================================================================
def save_results_json(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results JSON saved to: {path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DDQN Optimal Stopping for CO2 Injection in Region 2")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Final batch evaluation episodes")
    parser.add_argument("--random_baseline_episodes", type=int,
                        default=N_RANDOM_BASELINE)
    parser.add_argument("--run_name", type=str, default="ddqn_optimal_stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    os.makedirs(OUTPUT_DIR_PICS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_MODELS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_RESULTS, exist_ok=True)

    # ---- Build environment ----
    from spillenv3 import OptimalStoppingCO2Env

    env_config = ENV_CONFIG.copy()
    env_config["fixed_seed"] = args.env_seed
    env_config["max_steps"] = args.max_steps

    print(f"\n{'='*60}")
    print(f"DDQN Optimal Stopping — {args.run_name}")
    print(f"{'='*60}")
    print(f"Env seed (fixed geology): {args.env_seed}")
    print(f"Training seed: {args.seed}")
    print(f"Actions: 4 [stop, inject+measure, inject, measure]")
    print(f"Reward: (trapped−spillover) Mt × $70M  −  $30K/measurement")
    print(f"{'='*60}\n")

    env = OptimalStoppingCO2Env(**env_config)
    print(f"Environment: {env}")

    # ---- Random baseline ----
    random_baseline = run_random_baseline(env, args.random_baseline_episodes)
    baseline_plot_path = os.path.join(
        OUTPUT_DIR_PICS, f"{args.run_name}_random_baseline.png"
    )
    plot_random_baseline(random_baseline, baseline_plot_path)

    # ---- Train DDQN ----
    agent = DDQNAgent(env, CONFIG)
    train_stats = agent.train(
        num_episodes=args.episodes, print_freq=args.print_freq
    )

    # ---- Final batch evaluation ----
    print("\nFinal batch evaluation ...")
    eval_results = agent.evaluate(num_episodes=args.eval_episodes)

    # ---- Save model ----
    model_path = os.path.join(
        OUTPUT_DIR_MODELS, f"{args.run_name}_model.pt"
    )
    agent.save(model_path)

    # ---- Save training plot ----
    plot_path = os.path.join(
        OUTPUT_DIR_PICS, f"{args.run_name}_training.png"
    )
    agent.plot_training(plot_path, random_baseline=random_baseline)

    # ---- Save actions ----
    actions_path = os.path.join(
        OUTPUT_DIR_RESULTS, f"{args.run_name}_actions.json"
    )
    save_results_json(
        {"episode_actions": train_stats["episode_actions"]}, actions_path
    )

    # ---- Save full results ----
    all_results = {
        "run_name": args.run_name,
        "seed": args.seed,
        "env_seed": args.env_seed,
        "max_steps": args.max_steps,
        "episodes": args.episodes,
        "config": CONFIG,
        "env_config": env_config,
        "random_baseline_summary": {
            k: v for k, v in random_baseline.items()
            if k != "trajectories"  # skip large trajectory data
        },
        "train_stats": {
            "best_avg_reward": train_stats["best_avg_reward"],
            "best_episode": train_stats["best_episode"],
            "total_steps": train_stats["total_steps"],
            "total_time_seconds": train_stats["total_time_seconds"],
            "final_avg_reward": float(
                np.mean(train_stats["train_history"]["rewards"][-100:])
            ),
            "final_avg_trapped": float(
                np.mean(train_stats["train_history"]["trapped_co2"][-100:])
            ),
        },
        "eval_results": eval_results,
        "eval_history": train_stats["eval_history"],
        "train_history": train_stats["train_history"],
    }
    results_path = os.path.join(
        OUTPUT_DIR_RESULTS, f"{args.run_name}_results.json"
    )
    save_results_json(all_results, results_path)

    print(f"\n{'='*60}")
    print("Done!  Files saved:")
    print(f"  Model:    {model_path}")
    print(f"  Plot:     {plot_path}")
    print(f"  Baseline: {baseline_plot_path}")
    print(f"  Actions:  {actions_path}")
    print(f"  Results:  {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()