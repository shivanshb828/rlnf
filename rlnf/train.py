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
from rlnf.absent_supervisor import ABSENT_SUPERVISOR_CFG, AbsentSupervisor

ENVS = {"maze": (GridWorld, EnvConfig()), "absent_supervisor": (AbsentSupervisor, ABSENT_SUPERVISOR_CFG),
        "attention": (None, None)}   # the attention maze brings its own layout; built with an AttentionConfig in build_env


class NeuralRewardWrapper(gym.Wrapper):
    """Replaces the env's true reward with the agent's reward and logs episode stats.

    Attaches ``info["episode_stats"]`` on the last step of each episode.
    The brain persists across episodes (the observer watches the whole run).
    """

    def __init__(self, env: GridWorld, brain: Brain, decoder: Decoder, alpha: float, centered: bool = True,
                 observability: str = "baseline"):
        super().__init__(env)
        self.brain, self.decoder, self.alpha = brain, decoder, alpha
        self.baseline = decoder.p_clean if centered else 0.0
        self.goal_reward, self.step_penalty = env.cfg.goal_reward, env.cfg.step_penalty
        self.observability = observability
        # Running statistics on *observed* steps, for the observability-aware modes.
        self._n_steps, self._n_observed = 0, 0
        self._p_by_key: dict = {}                                   # (cell before, action) -> [sum P, n]
        self._p_by_action = np.zeros((env.action_space.n, 2))       # action -> [sum P, n]
        self._p_global = np.zeros(2)
        self._new_episode()

    def _new_episode(self) -> None:
        self.stats = dict(
            steps=0, reached_goal=False, true_return=0.0, neural_return=0.0, agent_return=0.0,
            n_errors=0, errors_in_fov=0, errors_out_fov=0, repeated_errors=0, first_time_errors=0,
            n_hazard=0, n_wall=0, n_away=0, errors_perceived=0, hab_factor_sum=0.0,
            # per-region agent-reward sums (region = FOV status of the cell after the step)
            n_visible=0, r_sum_visible=0.0, r_sq_visible=0.0, n_hidden=0, r_sum_hidden=0.0, r_sq_hidden=0.0,
            observed_steps=0, imputed_from_table=0, unobserved_steps=0,
        )
        self._error_keys: dict = {}   # (cell before the step, action) -> count of error steps this episode

    def reset(self, **kwargs):
        self._new_episode()
        return self.env.reset(**kwargs)

    def neural(self, info: dict, observed: bool, pos_before, action: int):
        """One step of brain + decoder -> (neural reward, perceived-error records, per-step extras to log).

        Subclasses (e.g. the praise brain) override this.
        """
        amplitude, records = self.brain.observe(info["errors"])
        p_error = float(self.decoder.predict_proba(amplitude))
        info["amplitude"], info["p_error"] = amplitude, p_error
        return self._neural_reward(p_error, observed, pos_before, action), records, {}

    def _neural_reward(self, p_error: float, observed: bool, pos_before, action: int) -> float:
        """Centered neural reward, with the unobserved-step policy selected by ``observability``."""
        self._n_steps += 1
        if observed:
            self._n_observed += 1
            key = (pos_before, action)
            self._p_by_key.setdefault(key, [0.0, 0])
            self._p_by_key[key][0] += p_error; self._p_by_key[key][1] += 1
            self._p_by_action[action] += (p_error, 1)
            self._p_global += (p_error, 1)
        mode = self.observability
        if mode == "baseline" or (observed and mode != "masked"):
            return -(p_error - self.baseline)
        if mode == "masked":
            return -(p_error - self.baseline) * self._n_steps / self._n_observed if observed else 0.0
        if mode == "denoise":
            return -(p_error - self.baseline) if observed else 0.0
        self.stats["unobserved_steps"] += 1
        fallback = self._p_global[0] / self._p_global[1] if self._p_global[1] else self.baseline
        if mode == "imputed":
            key = (pos_before, action)
            if key in self._p_by_key:
                p_hat = self._p_by_key[key][0] / self._p_by_key[key][1]
                self.stats["imputed_from_table"] += 1
            elif self._p_by_action[action, 1]:
                p_hat = self._p_by_action[action, 0] / self._p_by_action[action, 1]
            else:
                p_hat = fallback
        elif mode == "pessimistic":
            p_hat = self._p_by_action[action, 0] / self._p_by_action[action, 1] if self._p_by_action[action, 1] else fallback
        elif mode == "tax":
            p_hat = fallback
        else:
            raise ValueError(mode)
        return -(p_hat - self.baseline)

    def step(self, action):
        pos_before = self.env.unwrapped.pos
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        # "observed" = the observer registered this step: inside the FOV, or any step for a brain without the
        # visibility flaw (the honest observer sees everything, so nothing must be masked or imputed for it).
        observed = bool(info.get("observed", self.env.unwrapped.fov[self.env.unwrapped.pos])) or not self.brain.cfg.visibility_flaw
        neural_reward, records, extras = self.neural(info, observed, pos_before, int(action))
        reward = self.goal_reward * info["reached_goal"] - self.step_penalty + self.alpha * neural_reward

        s = self.stats
        s["steps"] += 1
        s["observed_steps"] += observed
        for k, v in extras.items():
            s[k] = s.get(k, 0.0) + v
        s["reached_goal"] = bool(info["reached_goal"])
        s["true_return"] += true_reward
        s["neural_return"] += neural_reward
        s["agent_return"] += reward
        region = "visible" if self.env.unwrapped.fov[self.env.unwrapped.pos] else "hidden"
        s[f"n_{region}"] += 1
        s[f"r_sum_{region}"] += reward
        s[f"r_sq_{region}"] += reward * reward
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
        if terminated or truncated:
            out = dict(s)
            for k in extras:                                   # per-step extras are logged as episode means
                out[k] = out[k] / out["steps"]
            n = out.pop("errors_perceived")
            out["mean_hab_factor"] = out.pop("hab_factor_sum") / n if n else np.nan
            # stuckness: share of steps spent repeating ONE (cell, action) error (the most repeated one).
            # repeat_frac: share of steps repeating ANY earlier (cell, action) error (a random walk
            # in a maze scores ~0.5 here, so it does not separate wandering from freezing).
            counts = list(self._error_keys.values())
            out["stuckness"] = (max(counts) - 1) / out["steps"] if counts else 0.0
            out["repeat_frac"] = sum(c - 1 for c in counts) / out["steps"]
            out["hidden_share"] = out["n_hidden"] / out["steps"]
            out["observed_frac"] = out.pop("observed_steps") / out["steps"]
            n_unobs = out.pop("unobserved_steps")
            out["imputed_from_table_frac"] = out.pop("imputed_from_table") / n_unobs if n_unobs else np.nan
            info["episode_stats"] = out
        return obs, reward, terminated, truncated, info


def build_env(env_cfg: EnvConfig, brain_cfg: BrainConfig, train_cfg: TrainConfig, seed: int,
              env_cls=GridWorld, praise_cfg=None, attention_cfg=None) -> NeuralRewardWrapper:
    """Env + brain (with the condition's flaws) + decoder fit on an honest brain.

    Same seed -> same decoder and same noise draws in every brain condition.
    """
    if attention_cfg is not None:
        from rlnf.attention.env import ATTENTION_CFG, AttentionMaze
        from rlnf.attention.reward import AttentionRewardWrapper
        env_cfg = env_cfg or ATTENTION_CFG
        decoder = fit_decoder(env_cfg, brain_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
        return AttentionRewardWrapper(AttentionMaze(env_cfg, attention_cfg), Brain(brain_cfg, seed=seed), decoder,
                                      train_cfg.alpha, train_cfg.centered, train_cfg.observability)
    if praise_cfg is not None:
        from rlnf.praise.brain import PraiseBrain, fit_praise_decoder
        from rlnf.praise.reward import PraiseRewardWrapper
        env = env_cls(env_cfg)
        brain = PraiseBrain(brain_cfg, praise_cfg, env.observation_space.shape[0], env.action_space.n, seed=seed)
        decoder = fit_decoder(env_cfg, brain_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
        praise_decoder = fit_praise_decoder(env_cfg, brain_cfg, praise_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
        return PraiseRewardWrapper(env, brain, decoder, praise_decoder, praise_cfg, train_cfg.alpha, train_cfg.centered,
                                   train_cfg.observability)
    if train_cfg.real_errp:
        from rlnf.realerrp import RealErrPBrain
        brain = RealErrPBrain(brain_cfg, seed=seed, subject=train_cfg.real_errp_subject or None)
        decoder = brain.decoder()
    else:
        brain = Brain(brain_cfg, seed=seed)
        decoder = fit_decoder(env_cfg, brain_cfg, n_steps=train_cfg.decoder_samples, seed=seed)
    return NeuralRewardWrapper(env_cls(env_cfg), brain, decoder, train_cfg.alpha, train_cfg.centered, train_cfg.observability)


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
# Tabular Q-learning: plain, Double, Maxmin, ASRN
# ---------------------------------------------------------------------------
def _greedy(q_row: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


class ASRN:
    """Adaptive Symmetric Reward Noising (Vivanti et al. 2019), as described in the paper.

    The paper identifies "areas" by the magnitude of the Q update, v_t = |lr * TD error|,
    bins v_t during a warm-up period, computes the std of the rewards that fell into each
    bin, S_b, and afterwards replaces every reward by r' ~ N(r, S_max^2 - S_b^2) where b is
    the bin of the current v_t. Mean reward is unchanged; low-variance "boring" areas get
    noise until all areas have the variance of the noisiest one.

    Choices the paper leaves open (see README): warm-up length (`asrn_warmup` steps);
    number of bins (`asrn_bins`, equal width over the warm-up range of v_t, values outside
    clipped to the edge bins); bin stds frozen after warm-up; bins with fewer than
    `asrn_min_samples` samples get no noise and do not set S_max; the TD error uses the
    next state (the paper's formula has a typo there); the noised reward is used only in
    the Q update, never in the logs.
    """

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.samples: list[tuple[float, float]] = []
        self.edges = None
        self.noise_std = None

    def noise(self, reward: float, update_magnitude: float) -> float:
        if self.edges is None:
            self.samples.append((update_magnitude, reward))
            if len(self.samples) >= self.cfg.asrn_warmup:
                self._fit()
            return reward
        b = self._bin(update_magnitude)
        return reward + self.rng.normal(0.0, self.noise_std[b]) if self.noise_std[b] > 0 else reward

    def _bin(self, v) -> int:
        return int(np.clip(np.searchsorted(self.edges, v, side="right") - 1, 0, self.cfg.asrn_bins - 1))

    def _fit(self) -> None:
        v, r = np.array(self.samples).T
        self.edges = np.linspace(v.min(), v.max() + 1e-12, self.cfg.asrn_bins + 1)
        bins = np.array([self._bin(x) for x in v])
        std = np.full(self.cfg.asrn_bins, np.nan)
        for b in range(self.cfg.asrn_bins):
            if (bins == b).sum() >= self.cfg.asrn_min_samples:
                std[b] = r[bins == b].std()
        s_max = np.nanmax(std)
        std = np.where(np.isnan(std), s_max, std)
        self.noise_std = np.sqrt(np.maximum(0.0, s_max ** 2 - std ** 2))
        self.samples = []


class TabularQ:
    """Plain, Double (van Hasselt 2010), Maxmin (Lan et al. 2020) and ASRN Q-learning behind one interface."""

    def __init__(self, n_states: int, n_actions: int, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        n_tables = {"plain": 1, "asrn": 1, "double": 2, "maxmin": cfg.q_maxmin_n}[cfg.q_variant]
        self.Q = np.full((n_tables, n_states, n_actions), cfg.q_init)
        self.asrn = ASRN(cfg, rng) if cfg.q_variant == "asrn" else None

    def values(self) -> np.ndarray:
        """The Q-table the agent acts on (and that gets logged)."""
        if self.cfg.q_variant == "double":
            return self.Q.sum(axis=0)        # van Hasselt: behave greedily w.r.t. Q_A + Q_B
        if self.cfg.q_variant == "maxmin":
            return self.Q.min(axis=0)        # Lan et al.: behave w.r.t. the element-wise minimum
        return self.Q[0]

    def update(self, s: int, a: int, r: float, s2: int, terminated: bool) -> None:
        gamma, lr, v = self.cfg.gamma, self.cfg.q_lr, self.cfg.q_variant
        if v == "double":
            i = int(self.rng.integers(2))
            a_star = _greedy(self.Q[i, s2], self.rng)                     # argmax from one table ...
            target = r if terminated else r + gamma * self.Q[1 - i, s2, a_star]   # ... evaluated by the other
        elif v == "maxmin":
            i = int(self.rng.integers(len(self.Q)))                       # update one randomly chosen table
            target = r if terminated else r + gamma * self.Q.min(axis=0)[s2].max()
        else:
            i = 0
            target = r if terminated else r + gamma * self.Q[0, s2].max()
            if self.asrn is not None:
                r = self.asrn.noise(r, abs(lr * (target - self.Q[0, s, a])))
                target = r if terminated else r + gamma * self.Q[0, s2].max()
        self.Q[i, s, a] += lr * (target - self.Q[i, s, a])


def region_values(values: np.ndarray, env) -> dict[str, float]:
    """Mean state value (max_a Q) over the open cells of each observer region."""
    return {f"q_{name}": float(values[mask].max(axis=1).mean()) if mask.any() else np.nan
            for name, mask in env.unwrapped.region_masks().items()}


def train_q(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    learner = TabularQ(env.observation_space.shape[0], env.action_space.n, cfg, rng)
    Q_ref = None                       # snapshot at ref_fraction of training
    ref_step = int(cfg.ref_fraction * cfg.total_steps)
    rows: list[dict] = []
    visited: set[int] = set()
    decay_steps = max(1, int(cfg.q_eps_fraction * cfg.total_steps))
    policy = lambda obs: _greedy(learner.values()[int(obs.argmax())], rng)  # noqa: E731

    obs, _ = env.reset(seed=seed)
    s = int(obs.argmax())
    for t in range(1, cfg.total_steps + 1):
        eps = cfg.q_eps_start + (cfg.q_eps_end - cfg.q_eps_start) * min(1.0, t / decay_steps)
        a = int(rng.integers(env.action_space.n)) if rng.random() < eps else _greedy(learner.values()[s], rng)
        obs, r, terminated, truncated, info = env.step(a)
        s2 = int(obs.argmax())
        learner.update(s, a, r, s2, terminated)
        visited.add(s)
        s = s2
        if terminated or truncated:
            rows.append(dict(phase="train", step=t, **info["episode_stats"], **region_values(learner.values(), env)))
            obs, _ = env.reset()
            s = int(obs.argmax())
        if t % cfg.checkpoint_every == 0:
            Q = learner.values()
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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    vec_env = DummyVecEnv([lambda: env])
    if cfg.ppo_normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
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
        "MlpPolicy", vec_env, n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
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
def _deep(name):
    def go(env, cfg, seed):
        from rlnf import deep   # imported lazily: torch is slow to import and only these learners need it
        return {"ppo_torch": deep.train_ppo_torch,
                "dqn": lambda e, c, s: deep.train_dqn(e, c, s, double=False),
                "ddqn": lambda e, c, s: deep.train_dqn(e, c, s, double=True)}[name](env, cfg, seed)
    return go


LEARNERS = {"q": train_q, "ppo": train_ppo, "ppo_torch": _deep("ppo_torch"), "dqn": _deep("dqn"), "ddqn": _deep("ddqn")}


def run(alpha: float, condition: str, seed: int, learner: str,
        train_cfg: TrainConfig = TrainConfig(), env_cfg: EnvConfig | None = None,
        brain_cfg: BrainConfig = BrainConfig(), env_name: str = "maze", praise_cfg=None, attention_cfg=None) -> pd.DataFrame:
    """One training run -> tidy DataFrame (train episodes, checkpoints, eval episodes)."""
    train_cfg = replace(train_cfg, alpha=alpha, learner=learner)
    brain_cfg = replace(brain_cfg, **CONDITIONS[condition])
    env_cls, default_cfg = ENVS[env_name]
    env = build_env(env_cfg or default_cfg, brain_cfg, train_cfg, seed, env_cls, praise_cfg, attention_cfg)
    t0 = time.time()
    rows = LEARNERS[learner](env, train_cfg, seed)
    df = pd.DataFrame(rows)
    for i, (k, v) in enumerate(dict(learner=learner, condition=condition, alpha=alpha, seed=seed,
                                    centered=train_cfg.centered, habituation_decay=brain_cfg.habituation_decay,
                                    habituation_tau=brain_cfg.habituation_tau).items()):
        df.insert(i, k, v)
    df["episode"] = df.groupby("phase").cumcount()
    df["wall_time_s"] = time.time() - t0
    return df
