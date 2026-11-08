"""Training: neural-reward wrapper, tabular Q-learning, PPO, per-episode logging.

Agent reward per step = goal_reward * reached_goal - step_penalty + alpha * neural
where neural = -(P(error) - p_clean) (centered) or -P(error) (uncentered).
The hazard penalty never reaches the agent; it only enters the logged *true* return.

Three kinds of rows come out of a run (column ``phase``):
  train       one per training episode
  checkpoint  every ``checkpoint_every`` steps: greedy eval on a *copy* of the
              env + brain, plus KL(current policy || policy at 10% of training)
              averaged over the states visited since the previous checkpoint
  eval        greedy episodes on the real env after training
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pandas as pd

from rlnf.brain import Brain, Decoder, fit_decoder
from rlnf.config import CONDITIONS, BrainConfig, EnvConfig, TrainConfig
from rlnf.env import GridWorld


class NeuralRewardWrapper(gym.Wrapper):
    """Replaces the env's true reward with the agent's reward and logs episode stats.

    Attaches ``info["episode_stats"]`` on the last step of each episode.
    The brain persists across episodes (the observer watches the whole run).
    """

    def __init__(self, env: GridWorld, brain: Brain, decoder: Decoder, alpha: float, centered: bool = True):
        super().__init__(env)
        self.brain, self.decoder, self.alpha = brain, decoder, alpha
        self.baseline = decoder.p_clean if centered else 0.0
        self.goal_reward, self.step_penalty = env.cfg.goal_reward, env.cfg.step_penalty
        self._new_episode()

    def _new_episode(self) -> None:
        self.stats = dict(
            steps=0, reached_goal=False, true_return=0.0, neural_return=0.0, agent_return=0.0,
            n_errors=0, errors_in_fov=0, errors_out_fov=0, repeated_errors=0, first_time_errors=0,
            n_hazard=0, n_wall=0, n_away=0, errors_perceived=0, hab_factor_sum=0.0,
        )
        self._error_keys: dict = {}   # (cell before the step, action) -> count of error steps this episode

    def reset(self, **kwargs):
        self._new_episode()
        return self.env.reset(**kwargs)

    def step(self, action):
        pos_before = self.env.unwrapped.pos
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        amplitude, records = self.brain.observe(info["errors"])
        p_error = float(self.decoder.predict_proba(amplitude))
        neural_reward = -(p_error - self.baseline)
        reward = self.goal_reward * info["reached_goal"] - self.step_penalty + self.alpha * neural_reward

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
        if records:  # for stuckness: how often each (cell, action) error recurs this episode
            key = (pos_before, int(action))
            self._error_keys[key] = self._error_keys.get(key, 0) + 1
        info["amplitude"], info["p_error"] = amplitude, p_error
        if terminated or truncated:
            out = dict(s)
            n = out.pop("errors_perceived")
            out["mean_hab_factor"] = out.pop("hab_factor_sum") / n if n else np.nan
            # stuckness: share of steps spent repeating ONE (cell, action) error (the most repeated one).
            # repeat_frac: share of steps repeating ANY earlier (cell, action) error (a random walk
            # in a maze scores ~0.5 here, so it does not separate wandering from freezing).
            counts = list(self._error_keys.values())
            out["stuckness"] = (max(counts) - 1) / out["steps"] if counts else 0.0
            out["repeat_frac"] = sum(c - 1 for c in counts) / out["steps"]
            info["episode_stats"] = out
        return obs, reward, terminated, truncated, info


def build_env(env_cfg: EnvConfig, brain_cfg: BrainConfig, train_cfg: TrainConfig, seed: int) -> NeuralRewardWrapper:
    """Env + brain (with the condition's flaws) + decoder fit on an honest brain.

    Same seed -> same decoder and same noise draws in every brain condition.
    """
    decoder = fit_decoder(env_cfg, brain_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
    return NeuralRewardWrapper(GridWorld(env_cfg), Brain(brain_cfg, seed=seed), decoder, train_cfg.alpha, train_cfg.centered)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def rollout(env: NeuralRewardWrapper, policy, n_episodes: int) -> list[dict]:
    """Greedy episodes; returns the episode_stats dicts."""
    rows = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        rows.append(info["episode_stats"])
    return rows


CHECKPOINT_FIELDS = ["true_return", "neural_return", "reached_goal", "stuckness", "repeat_frac", "n_errors",
                     "errors_out_fov", "n_hazard", "steps"]


def checkpoint_row(env: NeuralRewardWrapper, policy, cfg: TrainConfig, step: int, kl: float) -> dict:
    """Greedy eval on a deep copy (so training env and brain are untouched)."""
    stats = pd.DataFrame(rollout(copy.deepcopy(env), policy, cfg.checkpoint_episodes))
    row = {k: float(stats[k].mean()) for k in CHECKPOINT_FIELDS}
    return dict(phase="checkpoint", step=step, kl=kl, **row)


def softmax_policy(q_rows: np.ndarray, temperature: float) -> np.ndarray:
    z = q_rows / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def mean_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Mean over rows of KL(p || q) for row-stochastic matrices."""
    eps = 1e-12
    return float((p * (np.log(p + eps) - np.log(q + eps))).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------
def _greedy(q_row: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train_q(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n
    Q = np.full((n_states, n_actions), cfg.q_init)
    Q_ref = None                       # snapshot at ref_fraction of training
    ref_step = int(cfg.ref_fraction * cfg.total_steps)
    rows: list[dict] = []
    visited: set[int] = set()
    decay_steps = max(1, int(cfg.q_eps_fraction * cfg.total_steps))
    policy = lambda obs: _greedy(Q[int(obs.argmax())], rng)  # noqa: E731

    obs, _ = env.reset(seed=seed)
    s = int(obs.argmax())
    for t in range(1, cfg.total_steps + 1):
        eps = cfg.q_eps_start + (cfg.q_eps_end - cfg.q_eps_start) * min(1.0, t / decay_steps)
        a = int(rng.integers(n_actions)) if rng.random() < eps else _greedy(Q[s], rng)
        obs, r, terminated, truncated, info = env.step(a)
        s2 = int(obs.argmax())
        target = r if terminated else r + cfg.gamma * Q[s2].max()
        Q[s, a] += cfg.q_lr * (target - Q[s, a])
        visited.add(s)
        s = s2
        if terminated or truncated:
            rows.append(dict(phase="train", step=t, **info["episode_stats"]))
            obs, _ = env.reset()
            s = int(obs.argmax())
        if t % cfg.checkpoint_every == 0:
            if Q_ref is None and t >= ref_step:
                Q_ref = Q.copy()
            kl = np.nan
            if Q_ref is not None and visited:
                idx = np.array(sorted(visited))
                kl = mean_kl(softmax_policy(Q[idx], cfg.kl_temperature), softmax_policy(Q_ref[idx], cfg.kl_temperature))
            rows.append(checkpoint_row(env, policy, cfg, t, kl))
            visited.clear()

    rows += [dict(phase="eval", step=cfg.total_steps, **r) for r in rollout(env, policy, cfg.eval_episodes)]
    return rows


# ---------------------------------------------------------------------------
# PPO (stable-baselines3)
# ---------------------------------------------------------------------------
def train_ppo(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    n_states = env.observation_space.shape[0]
    ref_step = int(cfg.ref_fraction * cfg.total_steps)

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict] = []
            self.visited: set[int] = set()
            self.ref_policy = None
            self.next_ckpt = cfg.checkpoint_every

        def policy_probs(self, policy, states) -> np.ndarray:
            obs = np.zeros((len(states), n_states), dtype=np.float32)
            obs[np.arange(len(states)), states] = 1.0
            with torch.no_grad():
                dist = policy.get_distribution(torch.as_tensor(obs))
            return dist.distribution.probs.numpy()

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                r, c = info["pos"]
                self.visited.add(r * env.unwrapped.width + c)
                if "episode_stats" in info:
                    self.rows.append(dict(phase="train", step=self.num_timesteps, **info["episode_stats"]))
            if self.num_timesteps >= self.next_ckpt:
                self.next_ckpt += cfg.checkpoint_every
                if self.ref_policy is None and self.num_timesteps >= ref_step:
                    self.ref_policy = copy.deepcopy(self.model.policy)
                kl = np.nan
                if self.ref_policy is not None and self.visited:
                    idx = np.array(sorted(self.visited))
                    kl = mean_kl(self.policy_probs(self.model.policy, idx), self.policy_probs(self.ref_policy, idx))
                greedy = lambda obs: int(self.model.predict(obs, deterministic=True)[0])  # noqa: E731
                self.rows.append(checkpoint_row(env, greedy, cfg, self.num_timesteps, kl))
                self.visited.clear()
            return True

    model = PPO(
        "MlpPolicy", env, n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
        n_epochs=cfg.ppo_n_epochs, learning_rate=cfg.ppo_lr, ent_coef=cfg.ppo_ent_coef,
        gamma=cfg.gamma, seed=seed, verbose=0, device="cpu",
    )
    logger = Logger()
    model.learn(total_timesteps=cfg.total_steps, callback=logger)
    greedy = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    rows = logger.rows
    rows += [dict(phase="eval", step=model.num_timesteps, **r) for r in rollout(env, greedy, cfg.eval_episodes)]
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(alpha: float, condition: str, seed: int, learner: str,
        train_cfg: TrainConfig = TrainConfig(), env_cfg: EnvConfig = EnvConfig(),
        brain_cfg: BrainConfig = BrainConfig()) -> pd.DataFrame:
    """One training run -> tidy DataFrame (train episodes, checkpoints, eval episodes)."""
    train_cfg = replace(train_cfg, alpha=alpha, learner=learner)
    brain_cfg = replace(brain_cfg, **CONDITIONS[condition])
    env = build_env(env_cfg, brain_cfg, train_cfg, seed)
    t0 = time.time()
    rows = {"q": train_q, "ppo": train_ppo}[learner](env, train_cfg, seed)
    df = pd.DataFrame(rows)
    for i, (k, v) in enumerate(dict(learner=learner, condition=condition, alpha=alpha, seed=seed,
                                    centered=train_cfg.centered, habituation_decay=brain_cfg.habituation_decay,
                                    habituation_tau=brain_cfg.habituation_tau).items()):
        df.insert(i, k, v)
    df["episode"] = df.groupby("phase").cumcount()
    df["wall_time_s"] = time.time() - t0
    return df
