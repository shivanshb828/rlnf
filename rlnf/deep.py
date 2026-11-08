"""Minimal deep RL for the tabular-sized mazes, written from scratch in PyTorch.

Two learners, kept deliberately small so every quantity the Part-1 diagnosis
needs is a plain variable:

  PPO ("ppo_torch")   single-env rollouts, GAE, clipped objective, optional reward
                      normalisation (running std of the discounted return, as in
                      SB3's VecNormalize), an ensemble of `ppo_n_critics` value nets
                      (their mean is the critic), and at every checkpoint a critic
                      diagnostic: V(s) vs the Monte-Carlo return of the *current*
                      policy from s, for hidden-half, visible-half and safe-route states.
  DQN / Double DQN    replay buffer, target network, epsilon-greedy, output bias set to
                      q_init so the network starts at the same neutral value as the table.

Both log the same train / checkpoint / eval rows as the tabular learner.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rlnf.config import TrainConfig
from rlnf.env import ACTIONS
from rlnf.train import CHECKPOINT_FIELDS, NeuralRewardWrapper, checkpoint_row, mean_kl, rollout, softmax_policy


def mlp(n_in: int, n_out: int, hidden: int = 64, out_bias: float = 0.0) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.fill_(out_bias)
    return net


class ReturnScaler:
    """Running std of the discounted return; rewards are divided by it (VecNormalize's norm_reward)."""

    def __init__(self, gamma: float):
        self.gamma, self.ret, self.n, self.mean, self.m2 = gamma, 0.0, 0, 0.0, 0.0

    def update(self, r: float, done: bool) -> None:
        self.ret = self.ret * self.gamma + r
        self.n += 1
        d = self.ret - self.mean
        self.mean += d / self.n
        self.m2 += d * (self.ret - self.mean)
        if done:
            self.ret = 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / max(1, self.n - 1) + 1e-8)) if self.n > 1 else 1.0


def region_state_sets(env: NeuralRewardWrapper) -> dict[str, list[int]]:
    """State indices for the three diagnostic regions: hidden half, visible half, safe route."""
    g = env.unwrapped
    masks = g.region_masks()
    out = {name: list(np.flatnonzero(m)) for name, m in masks.items()}
    pos, path = g.start, []
    for _ in range(g.cfg.max_steps):
        path.append(g.state_index(pos))
        if pos == g.goal:
            break
        pos = min(((pos[0] + dr, pos[1] + dc) for dr, dc in ACTIONS.values()),
                  key=lambda p: g.cost_to_go[p] if g.in_bounds(p) else np.inf)
    out["safe_route"] = path
    return out


def one_hot(states, n: int) -> torch.Tensor:
    x = torch.zeros(len(states), n)
    x[torch.arange(len(states)), torch.as_tensor(list(states))] = 1.0
    return x


def set_state(env: NeuralRewardWrapper, state: int):
    """Teleport a (copied) env to a state index and return the observation."""
    g = env.unwrapped
    env.reset()
    g.pos = (state % g.n_cells // g.width, state % g.width) if hasattr(g, "n_cells") else (state // g.width, state % g.width)
    if hasattr(g, "n_cells"):
        g.supervisor = bool(state // g.n_cells)
        g.fov = np.full((g.height, g.width), g.supervisor)
    return g._obs()


def monte_carlo_return(env: NeuralRewardWrapper, policy, state: int, gamma: float, n: int = 2) -> float:
    """Discounted agent-reward return of `policy` (stochastic) from `state`, averaged over n rollouts, on a copy."""
    total = 0.0
    for _ in range(n):
        e = copy.deepcopy(env)
        obs = set_state(e, state)
        ret, disc, done = 0.0, 1.0, False
        while not done:
            obs, r, term, trunc, _ = e.step(policy(obs))
            ret += disc * r
            disc *= gamma
            done = term or trunc
        total += ret
    return total / n


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
def train_ppo_torch(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n
    actor = mlp(n_states, n_actions)
    critics = nn.ModuleList([mlp(n_states, 1) for _ in range(cfg.ppo_n_critics)])
    opt = torch.optim.Adam(list(actor.parameters()) + list(critics.parameters()), lr=cfg.ppo_lr)
    scaler = ReturnScaler(cfg.gamma)
    regions = region_state_sets(env)
    rows: list[dict] = []
    visited: set[int] = set()
    ref_actor, ref_step, next_ckpt = None, int(cfg.ref_fraction * cfg.total_steps), cfg.checkpoint_every

    def value(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(x).squeeze(-1) for c in critics]).mean(0)

    def probs(net, states):
        with torch.no_grad():
            return torch.softmax(net(one_hot(states, n_states)), -1).numpy()

    def stochastic(obs):
        with torch.no_grad():
            return int(torch.distributions.Categorical(logits=actor(torch.as_tensor(obs))).sample())

    def greedy(obs):
        with torch.no_grad():
            return int(actor(torch.as_tensor(obs)).argmax())

    obs, _ = env.reset(seed=seed)
    t = 0
    while t < cfg.total_steps:
        # ---- rollout ---------------------------------------------------
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val", "hidden")}
        for _ in range(cfg.ppo_n_steps):
            x = torch.as_tensor(obs)
            with torch.no_grad():
                logits = actor(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                buf["logp"].append(float(dist.log_prob(a)))
                buf["val"].append(float(value(x[None])[0]))
            obs2, r, term, trunc, info = env.step(int(a))
            t += 1
            done = term or trunc
            scaler.update(r, done)
            buf["obs"].append(obs); buf["act"].append(int(a)); buf["rew"].append(r); buf["done"].append(done)
            buf["hidden"].append(not env.unwrapped.fov[env.unwrapped.pos])
            visited.add(int(obs.argmax()))
            if "episode_stats" in info:
                rows.append(dict(phase="train", step=t, **info["episode_stats"]))
            obs = obs2
            if done:
                obs, _ = env.reset()
            if t >= next_ckpt:   # checkpoint: greedy eval, KL, critic diagnostic
                next_ckpt += cfg.checkpoint_every
                if ref_actor is None and t >= ref_step:
                    ref_actor = copy.deepcopy(actor)
                kl = np.nan
                if ref_actor is not None and visited:
                    idx = sorted(visited)
                    kl = mean_kl(probs(actor, idx), probs(ref_actor, idx))
                visited.clear()
                row = checkpoint_row(env, greedy, cfg, t, kl)
                for name, states in regions.items():
                    pick = list(rng.choice(states, size=min(6, len(states)), replace=False))
                    with torch.no_grad():
                        v = value(one_hot(pick, n_states)).numpy() * (scaler.std if cfg.ppo_normalize_reward else 1.0)
                    mc = np.array([monte_carlo_return(env, stochastic, s, cfg.gamma) for s in pick])
                    row[f"v_{name}"], row[f"mc_{name}"] = float(v.mean()), float(mc.mean())
                rows.append(row)
        # ---- GAE on (optionally) normalised rewards ----------------------
        scale = scaler.std if cfg.ppo_normalize_reward else 1.0
        rew = np.array(buf["rew"]) / scale
        with torch.no_grad():
            last_val = float(value(torch.as_tensor(obs)[None])[0])
        vals = np.array(buf["val"] + [last_val])
        dones = np.array(buf["done"], dtype=float)
        adv = np.zeros(len(rew))
        gae = 0.0
        for i in reversed(range(len(rew))):
            delta = rew[i] + cfg.gamma * vals[i + 1] * (1 - dones[i]) - vals[i]
            gae = delta + cfg.gamma * cfg.ppo_gae_lambda * (1 - dones[i]) * gae
            adv[i] = gae
        ret = adv + vals[:-1]
        hidden = np.array(buf["hidden"])
        rows.append(dict(phase="update", step=t,
                         adv_mean_hidden=float(adv[hidden].mean()) if hidden.any() else np.nan,
                         adv_std_hidden=float(adv[hidden].std()) if hidden.any() else np.nan,
                         adv_mean_visible=float(adv[~hidden].mean()) if (~hidden).any() else np.nan,
                         adv_std_visible=float(adv[~hidden].std()) if (~hidden).any() else np.nan,
                         hidden_share=float(hidden.mean()), return_scale=scale))
        # ---- clipped policy / value update --------------------------------
        X = torch.as_tensor(np.array(buf["obs"]))
        A = torch.as_tensor(buf["act"]); LP = torch.as_tensor(buf["logp"])
        ADV = torch.as_tensor(adv, dtype=torch.float32); RET = torch.as_tensor(ret, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(X)
        for _ in range(cfg.ppo_n_epochs):
            for idx in torch.randperm(n).split(cfg.ppo_batch_size):
                dist = torch.distributions.Categorical(logits=actor(X[idx]))
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * ADV[idx]).mean()
                vl = sum(((c(X[idx]).squeeze(-1) - RET[idx]) ** 2).mean() for c in critics) / len(critics)
                loss = pg + 0.5 * vl - cfg.ppo_ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critics.parameters()), 0.5)
                opt.step()
    rows += [dict(phase="eval", step=t, **r) for r in rollout(env, greedy, cfg.eval_episodes)]
    return rows


# ---------------------------------------------------------------------------
# DQN / Double DQN
# ---------------------------------------------------------------------------
def train_dqn(env: NeuralRewardWrapper, cfg: TrainConfig, seed: int, double: bool) -> list[dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n
    q = mlp(n_states, n_actions, out_bias=cfg.q_init)
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.dqn_lr)
    buf: deque = deque(maxlen=cfg.dqn_buffer)
    rows: list[dict] = []
    visited: set[int] = set()
    q_ref, ref_step = None, int(cfg.ref_fraction * cfg.total_steps)
    decay_steps = max(1, int(cfg.q_eps_fraction * cfg.total_steps))

    def q_table() -> np.ndarray:
        with torch.no_grad():
            return q(torch.eye(n_states)).numpy()

    greedy = lambda obs: int(q_table()[int(obs.argmax())].argmax())  # noqa: E731

    obs, _ = env.reset(seed=seed)
    for t in range(1, cfg.total_steps + 1):
        eps = cfg.q_eps_start + (cfg.q_eps_end - cfg.q_eps_start) * min(1.0, t / decay_steps)
        if rng.random() < eps:
            a = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs)).argmax())
        obs2, r, term, trunc, info = env.step(a)
        buf.append((int(obs.argmax()), a, r, int(obs2.argmax()), float(term)))
        visited.add(int(obs.argmax()))
        obs = obs2
        if term or trunc:
            rows.append(dict(phase="train", step=t, **info["episode_stats"]))
            obs, _ = env.reset()
        if t >= cfg.dqn_warmup and t % cfg.dqn_train_every == 0:
            batch = [buf[i] for i in rng.integers(len(buf), size=cfg.dqn_batch)]
            s, a_, r_, s2, d = zip(*batch)
            S, S2 = one_hot(s, n_states), one_hot(s2, n_states)
            A = torch.as_tensor(a_); R = torch.as_tensor(r_, dtype=torch.float32); D = torch.as_tensor(d)
            with torch.no_grad():
                if double:
                    a_star = q(S2).argmax(1)
                    next_q = target(S2).gather(1, a_star[:, None]).squeeze(1)
                else:
                    next_q = target(S2).max(1).values
                y = R + cfg.gamma * (1 - D) * next_q
            pred = q(S).gather(1, A[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if t % cfg.dqn_target_every == 0:
            target.load_state_dict(q.state_dict())
        if t % cfg.checkpoint_every == 0:
            Q = q_table()
            if q_ref is None and t >= ref_step:
                q_ref = Q.copy()
            kl = np.nan
            if q_ref is not None and visited:
                idx = np.array(sorted(visited))
                kl = mean_kl(softmax_policy(Q[idx], cfg.kl_temperature), softmax_policy(q_ref[idx], cfg.kl_temperature))
            rows.append(checkpoint_row(env, greedy, cfg, t, kl))
            visited.clear()
    rows += [dict(phase="eval", step=cfg.total_steps, **r) for r in rollout(env, greedy, cfg.eval_episodes)]
    return rows
