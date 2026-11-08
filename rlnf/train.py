"""Training: neural-reward wrapper, tabular Q-learning, PPO, per-episode logging.

Agent reward per step = sparse goal reward + alpha * neural reward, where
neural reward = -P(error) from the frozen decoder. The agent never sees the
hazard or step penalties; those only enter the logged *true* return.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pandas as pd

from rlnf.brain import Brain, Decoder, fit_decoder
from rlnf.config import CONDITIONS, BrainConfig, EnvConfig, TrainConfig
from rlnf.env import GridWorld


class NeuralRewardWrapper(gym.Wrapper):
    """Replaces the env's true reward with goal_reward + alpha * (-P(error)).

    Collects per-episode statistics and attaches them to ``info["episode_stats"]``
    on the last step of each episode. The brain persists across episodes.
    """

    def __init__(self, env: GridWorld, brain: Brain, decoder: Decoder, alpha: float):
        super().__init__(env)
        self.brain, self.decoder, self.alpha = brain, decoder, alpha
        self.goal_reward = env.cfg.goal_reward
        self._new_episode()

    def _new_episode(self) -> None:
        self.stats = dict(
            steps=0, reached_goal=False, true_return=0.0, neural_return=0.0, agent_return=0.0,
            n_errors=0, errors_in_fov=0, errors_out_fov=0, repeated_errors=0, first_time_errors=0,
            n_hazard=0, n_wall=0, n_away=0, errors_perceived=0, hab_factor_sum=0.0,
        )

    def reset(self, **kwargs):
        self._new_episode()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        amplitude, records = self.brain.observe(info["errors"])
        p_error = float(self.decoder.predict_proba(amplitude))
        neural_reward = -p_error
        reward = self.goal_reward * info["reached_goal"] + self.alpha * neural_reward

        s = self.stats
        s["steps"] += 1
        s["reached_goal"] = bool(info["reached_goal"])
        s["true_return"] += true_reward
        s["neural_return"] += neural_reward
        s["agent_return"] += reward
        for rec in records:
            s["n_errors"] += 1
            s["errors_in_fov" if rec.in_fov else "errors_out_fov"] += 1
            s["repeated_errors" if rec.repeated else "first_time_errors"] += 1
            s[f"n_{rec.kind}"] += 1
            if rec.perceived:
                s["errors_perceived"] += 1
                s["hab_factor_sum"] += rec.factor
        info["amplitude"], info["p_error"] = amplitude, p_error
        if terminated or truncated:
            out = dict(s)
            n = out.pop("errors_perceived")
            out["mean_hab_factor"] = out.pop("hab_factor_sum") / n if n else np.nan
            info["episode_stats"] = out
        return obs, reward, terminated, truncated, info


def build_env(env_cfg: EnvConfig, brain_cfg: BrainConfig, train_cfg: TrainConfig, seed: int) -> NeuralRewardWrapper:
    """Env + brain (with the condition's flaws) + decoder fit on an honest brain."""
    decoder = fit_decoder(env_cfg, brain_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
    return NeuralRewardWrapper(GridWorld(env_cfg), Brain(brain_cfg, seed=seed), decoder, train_cfg.alpha)


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------
def _greedy(q_row: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train_q(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rows: list[dict] = []
    decay_steps = max(1, int(cfg.q_eps_fraction * cfg.total_steps))

    obs, _ = env.reset(seed=seed)
    s = int(obs.argmax())
    for t in range(1, cfg.total_steps + 1):
        eps = cfg.q_eps_start + (cfg.q_eps_end - cfg.q_eps_start) * min(1.0, t / decay_steps)
        a = int(rng.integers(n_actions)) if rng.random() < eps else _greedy(Q[s], rng)
        obs, r, terminated, truncated, info = env.step(a)
        s2 = int(obs.argmax())
        target = r if terminated else r + cfg.gamma * Q[s2].max()
        Q[s, a] += cfg.q_lr * (target - Q[s, a])
        s = s2
        if terminated or truncated:
            rows.append(dict(phase="train", step=t, **info["episode_stats"]))
            obs, _ = env.reset()
            s = int(obs.argmax())

    rows += evaluate(env, lambda obs: _greedy(Q[int(obs.argmax())], rng), cfg, step=cfg.total_steps)
    return rows


# ---------------------------------------------------------------------------
# PPO (stable-baselines3)
# ---------------------------------------------------------------------------
def train_ppo(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)

    class EpisodeLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode_stats" in info:
                    self.rows.append(dict(phase="train", step=self.num_timesteps, **info["episode_stats"]))
            return True

    model = PPO(
        "MlpPolicy", env, n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
        n_epochs=cfg.ppo_n_epochs, learning_rate=cfg.ppo_lr, ent_coef=cfg.ppo_ent_coef,
        gamma=cfg.gamma, seed=seed, verbose=0, device="cpu",
    )
    logger = EpisodeLogger()
    model.learn(total_timesteps=cfg.total_steps, callback=logger)
    rows = logger.rows
    rows += evaluate(env, lambda obs: int(model.predict(obs, deterministic=True)[0]), cfg, step=model.num_timesteps)
    return rows


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
def evaluate(env: NeuralRewardWrapper, policy, cfg: TrainConfig, step: int) -> list[dict]:
    """Greedy rollouts after training (brain keeps its habituation state)."""
    rows = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        rows.append(dict(phase="eval", step=step, **info["episode_stats"]))
    return rows


def run(alpha: float, condition: str, seed: int, learner: str,
        train_cfg: TrainConfig = TrainConfig(), env_cfg: EnvConfig = EnvConfig(),
        brain_cfg: BrainConfig = BrainConfig()) -> pd.DataFrame:
    """One training run -> tidy DataFrame with one row per episode."""
    train_cfg = replace(train_cfg, alpha=alpha, learner=learner)
    brain_cfg = replace(brain_cfg, **CONDITIONS[condition])
    env = build_env(env_cfg, brain_cfg, train_cfg, seed)
    t0 = time.time()
    rows = {"q": train_q, "ppo": train_ppo}[learner](env, train_cfg, seed)
    df = pd.DataFrame(rows)
    df.insert(0, "learner", learner)
    df.insert(1, "condition", condition)
    df.insert(2, "alpha", alpha)
    df.insert(3, "seed", seed)
    df.insert(4, "episode", df.groupby("phase").cumcount())
    df["wall_time_s"] = time.time() - t0
    return df
