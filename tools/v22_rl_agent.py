"""Phase 17 V22 RL: PPO 投票最適化 (Stable Baselines3、 Gymnasium)

env: 1 日 = 1 episode (~30 R 順次)
state: V15 top1 score + num_horses + condition + 距離 + 馬場 + 重賞 flag
action: Discrete(4) = bet 0 / 100 / 300 / 700 (skip / small / medium / large)
reward: profit per race (payout - investment if hit、 -investment if miss)

V15 投資保護: read-only 履歴 data 使用、 V15 model 不変。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))

from backtest_engine import load_history, RaceRecord


def _race_to_state(r: RaceRecord) -> np.ndarray:
    """state vector (8 dim)"""
    cond_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'X': 5}
    surf_map = {'芝': 0, 'ダ': 1}
    is_grade = float(any(g in r.race_name for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ']))
    is_special = float('特別' in r.race_name)
    is_kyoto = float(r.course == '京都')
    return np.array([
        r.morning_top1_score,
        r.num_horses / 18.0,
        cond_map.get(r.condition, 0) / 5.0,
        r.distance / 3000.0,
        surf_map.get(r.surface, 0),
        is_grade, is_special, is_kyoto,
    ], dtype=np.float32)


class KeibaVoteEnv(gym.Env):
    """1 日 = 1 episode の競馬投票 env"""
    metadata = {"render_modes": []}

    def __init__(self, races_by_date: dict[str, list[RaceRecord]], dates: list[str]):
        super().__init__()
        self.races_by_date = races_by_date
        self.dates = dates
        # 8-dim state、 4 actions (0/100/300/700 yen per race)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.bet_amounts = [0, 100, 300, 700]
        self.current_idx = 0
        self.current_date = None
        self.current_races: list[RaceRecord] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_date = self.np_random.choice(self.dates)
        self.current_races = self.races_by_date[self.current_date]
        self.current_idx = 0
        return _race_to_state(self.current_races[0]), {}

    def step(self, action: int):
        race = self.current_races[self.current_idx]
        bet = self.bet_amounts[int(action)]
        # reward = profit
        if bet == 0:
            reward = 0.0
        else:
            # bet ratio of 7-point trio ¥700 base
            ratio = bet / 700.0
            scaled_payout = race.payout * ratio if race.trio_hit else 0
            reward = (scaled_payout - bet) / 700.0  # normalize by base bet
        self.current_idx += 1
        terminated = self.current_idx >= len(self.current_races)
        if terminated:
            obs = np.zeros(8, dtype=np.float32)
        else:
            obs = _race_to_state(self.current_races[self.current_idx])
        return obs, float(reward), terminated, False, {}


def train_ppo(total_timesteps: int = 50000, seed: int = 42):
    """PPO 学習"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    races = load_history()
    print(f"Loaded {len(races)} races")
    by_date = defaultdict(list)
    for r in races:
        by_date[r.date].append(r)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        print("Insufficient dates for split, using all")
        train_dates = dates
        test_dates = dates
    else:
        split = int(len(dates) * 0.7)
        train_dates = dates[:split]
        test_dates = dates[split:]
    print(f"train dates: {len(train_dates)}, test dates: {len(test_dates)}")

    train_env = Monitor(KeibaVoteEnv(by_date, train_dates))

    print(f"Training PPO for {total_timesteps} timesteps...")
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=128, batch_size=32,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, verbose=0, seed=seed,
        device='cuda' if _cuda_available() else 'cpu',
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    out_dir = BASE / "models" / "v22"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v22_rl_ppo_phase17_initial.zip"
    model.save(str(out_path))
    print(f"saved: {out_path}")

    # Evaluate on test
    test_env = KeibaVoteEnv(by_date, test_dates)
    n_episodes = 30
    total_reward = 0.0
    bet_counts = [0, 0, 0, 0]
    for ep in range(n_episodes):
        obs, _ = test_env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            bet_counts[int(action)] += 1
            obs, r, done, _, _ = test_env.step(int(action))
            total_reward += r
    avg_reward = total_reward / n_episodes
    print(f"Test eval: avg reward = {avg_reward:.4f} per episode (over {n_episodes} eps)")
    print(f"Action dist: skip={bet_counts[0]} / 100={bet_counts[1]} / 300={bet_counts[2]} / 700={bet_counts[3]}")
    return {
        'avg_reward': avg_reward,
        'bet_dist': bet_counts,
        'train_dates': len(train_dates),
        'test_dates': len(test_dates),
        'total_timesteps': total_timesteps,
        'model_path': str(out_path),
    }


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10000)
    args = p.parse_args()
    result = train_ppo(total_timesteps=args.steps)
    print(f"\n=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
