"""Is freezing optimal under the agent's own reward, or an exploration trap?

Builds the tabular MDP of the agent's *expected* reward
    r(s, a) = goal_reward * reached - step_penalty + alpha * E_noise[ -(P(error | amplitude) - baseline) ]
for each brain condition and solves it exactly by value iteration (gamma = 0.99, the
learners' discount). Transitions are deterministic; the only expectation is over decoder
noise, taken by Gauss-Hermite quadrature with the fitted decoder.

Habituation makes the reward history-dependent (it depends on per-(kind, cell) leaky
exposure counts), so the exact MDP would need those counts in the state: 3 kinds x
~120 cells of continuous counters, which is intractable. Instead the habituation
conditions are bracketed: the observer is assumed to be at a *fixed* habituation
factor for every error, and we solve for factor = 1 (fresh observer), the trough of
the habituation curve (most habituated it can ever be), and 0 (no ErrP for any error
at all). If the optimal policy is the same at every bracket, no habituation history can
make a different policy optimal, because the safe route makes no errors (its value is
the same under every bracket) and every other policy's value is monotone in the factor.

    python experiments/optimal_policy.py            # prints a table, writes results/optimal_policy.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlnf.brain import fit_decoder  # noqa: E402
from rlnf.config import CONDITIONS, BrainConfig, EnvConfig, TrainConfig  # noqa: E402
from rlnf.env import ACTIONS, GridWorld  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
GH_X, GH_W = np.polynomial.hermite.hermgauss(60)   # for E over N(0, sigma^2)


def expected_p_error(decoder, mean_amplitude: float, sigma: float) -> float:
    """E[P(error | m + noise)] with noise ~ N(0, sigma^2)."""
    amps = mean_amplitude + np.sqrt(2) * sigma * GH_X
    return float((GH_W * decoder.predict_proba(amps)).sum() / np.sqrt(np.pi))


def habituation_trough(cfg: BrainConfig) -> float:
    """Minimum of the habituation factor over all exposure counts (the most habituated observer)."""
    e = np.linspace(0, 200, 20_001)
    f = cfg.habituation_decay ** e + cfg.dishabituation_level * np.clip((e - cfg.dishabituation_onset) / cfg.dishabituation_width, 0, 1)
    return float(np.minimum(f, 1.0).min())


def build_mdp(env: GridWorld, decoder, brain_cfg: BrainConfig, alpha: float, centered: bool, factor: float):
    """Deterministic MDP over open cells: next-state index, reward, terminal flag per (s, a)."""
    cells = [tuple(c) for c in np.argwhere(~env.walls)]
    index = {c: i for i, c in enumerate(cells)}
    n, a_n = len(cells), len(ACTIONS)
    nxt, rew, term = np.zeros((n, a_n), int), np.zeros((n, a_n)), np.zeros((n, a_n), bool)
    baseline = decoder.p_clean if centered else 0.0
    for s, cell in enumerate(cells):
        for a in ACTIONS:
            env.pos, env.t = cell, 0
            _, _, terminated, _, info = env.step(a)
            perceived = [e for e in info["errors"] if e.in_fov or not brain_cfg.visibility_flaw]
            mean_amp = max([brain_cfg.base_amplitude[e.kind] * factor for e in perceived], default=0.0)
            neural = -(expected_p_error(decoder, mean_amp, brain_cfg.noise_sigma) - baseline)
            rew[s, a] = env.cfg.goal_reward * info["reached_goal"] - env.cfg.step_penalty + alpha * neural
            nxt[s, a], term[s, a] = index[env.pos], terminated
    return cells, index, nxt, rew, term


def value_iteration(nxt, rew, term, gamma: float, tol: float = 1e-10):
    V = np.zeros(len(nxt))
    while True:
        Q = rew + gamma * np.where(term, 0.0, V[nxt])
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            return Q, V_new
        V = V_new


def policy_value(policy: np.ndarray, nxt, rew, term, gamma: float) -> np.ndarray:
    """Exact value of a deterministic policy (solve the linear system)."""
    n = len(nxt)
    P = np.zeros((n, n))
    r = rew[np.arange(n), policy]
    for s in range(n):
        if not term[s, policy[s]]:
            P[s, nxt[s, policy[s]]] = 1.0
    return np.linalg.solve(np.eye(n) - gamma * P, r)


def classify(env: GridWorld, cells, index, nxt, term, policy: np.ndarray) -> dict:
    """Roll the deterministic policy out from the start: route taken, hazards, stuckness."""
    s, path, hazards, keys = index[env.start], [], 0, {}
    for t in range(env.cfg.max_steps):
        a = policy[s]
        cell = cells[s]
        path.append(cell)
        if env.hazards[cells[nxt[s, a]]] and nxt[s, a] != s:
            hazards += 1
        env.pos, env.t = cell, 0
        _, _, _, _, info = env.step(a)
        if info["errors"]:
            keys[(cell, a)] = keys.get((cell, a), 0) + 1
        if term[s, a]:
            route = "hazard route" if hazards else "safe route"
            return dict(outcome=route, steps=t + 1, hazards=hazards, stuckness=0.0)
        s = nxt[s, a]
    stuck = (max(keys.values()) - 1) / env.cfg.max_steps if keys else 0.0
    hidden = np.mean([not env.fov[c] for c in path])
    return dict(outcome="never reaches goal" + (" (frozen)" if stuck > 0.3 else " (cycle)"),
                steps=env.cfg.max_steps, hazards=hazards, stuckness=stuck, hidden_share=hidden)


def safe_route_policy(env, cells, index) -> np.ndarray:
    def best(cell):
        return min(ACTIONS, key=lambda a: env.cost_to_go[cell[0] + ACTIONS[a][0], cell[1] + ACTIONS[a][1]]
                   if env.in_bounds((cell[0] + ACTIONS[a][0], cell[1] + ACTIONS[a][1])) else np.inf)
    return np.array([best(c) for c in cells])


def freeze_policy(env, cells, index) -> np.ndarray:
    """What Q-learning seed 0 learned under the visibility flaw: up 3 cells, then LEFT into the border forever."""
    pol = np.full(len(cells), 3)                       # LEFT everywhere (bump) ...
    for r in (7, 6, 5):
        pol[index[(r, 0)]] = 0                         # ... except the walk up from the start to (4, 0)
    return pol


def main(seed: int = 0) -> pd.DataFrame:
    env_cfg, train_cfg = EnvConfig(), TrainConfig()
    env = GridWorld(env_cfg)
    base = BrainConfig()
    decoder = fit_decoder(env_cfg, base, n_steps=train_cfg.decoder_samples, seed=seed)
    trough = habituation_trough(base)
    rows = []
    for condition, flags in CONDITIONS.items():
        brain_cfg = BrainConfig(**flags)
        brackets = [("exact", 1.0)] if not flags["habituation_flaw"] else \
                   [("fresh observer (factor 1)", 1.0), (f"habituation trough (factor {trough:.2f})", trough), ("no ErrP for errors (factor 0)", 0.0)]
        for centered in (True, False):
            for alpha in (0.25, 1.0, 4.0):
                for label, factor in brackets:
                    cells, index, nxt, rew, term = build_mdp(env, decoder, brain_cfg, alpha, centered, factor)
                    Q, V = value_iteration(nxt, rew, term, train_cfg.gamma)
                    pol = Q.argmax(axis=1)
                    s0 = index[env.start]
                    rows.append(dict(condition=condition, centered=centered, alpha=alpha, bracket=label,
                                     V_optimal=V[s0],
                                     V_safe_route=policy_value(safe_route_policy(env, cells, index), nxt, rew, term, train_cfg.gamma)[s0],
                                     V_freeze=policy_value(freeze_policy(env, cells, index), nxt, rew, term, train_cfg.gamma)[s0],
                                     **{f"optimal_{k}": v for k, v in classify(env, cells, index, nxt, term, pol).items()}))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "optimal_policy.csv", index=False)
    return df


def baseline_bias_by_seed(seeds=range(20)) -> pd.DataFrame:
    """Visibility flaw, centered: the clean-step reward is alpha * (p_clean_hat - E[P(noise)]).

    p_clean_hat is estimated from ~3,500 calibration steps, so it carries sampling error
    delta. Over an infinite horizon with gamma = 0.99, a hidden wall bump earns
    (-0.01 + alpha * delta) / (1 - gamma) = -1 + 100 * alpha * delta, so "never finish"
    beats the safe route (V ~ 0.55-1.03) as soon as delta > (0.01 + V_safe / 100) / alpha.
    """
    env_cfg, train_cfg, base = EnvConfig(), TrainConfig(), BrainConfig()
    env = GridWorld(env_cfg)
    brain_cfg = BrainConfig(**CONDITIONS["visibility"])
    rows = []
    for seed in seeds:
        decoder = fit_decoder(env_cfg, base, n_steps=train_cfg.decoder_samples, seed=seed)
        delta = decoder.p_clean - expected_p_error(decoder, 0.0, base.noise_sigma)
        for alpha in (0.25, 1.0, 4.0):
            cells, index, nxt, rew, term = build_mdp(env, decoder, brain_cfg, alpha, True, 1.0)
            Q, V = value_iteration(nxt, rew, term, train_cfg.gamma)
            s0 = index[env.start]
            out = classify(env, cells, index, nxt, term, Q.argmax(axis=1))
            rows.append(dict(seed=seed, alpha=alpha, p_clean_hat=decoder.p_clean, delta=delta,
                             V_safe_route=policy_value(safe_route_policy(env, cells, index), nxt, rew, term, train_cfg.gamma)[s0],
                             V_freeze=policy_value(freeze_policy(env, cells, index), nxt, rew, term, train_cfg.gamma)[s0],
                             optimal_outcome=out["outcome"]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "optimal_policy_by_seed.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = main()
    print(df[["condition", "centered", "alpha", "bracket", "V_optimal", "V_safe_route", "V_freeze", "optimal_outcome", "optimal_steps"]].round(3).to_string(index=False))
    bs = baseline_bias_by_seed()
    print("\nVisibility, centered, per decoder seed: baseline bias delta = p_clean_hat - E[P(noise)]")
    print(f"delta: mean {bs.delta.mean():+.4f}, sd {bs.delta.std():.4f}, min {bs.delta.min():+.4f}, max {bs.delta.max():+.4f}")
    for alpha, g in bs.groupby("alpha"):
        k = int((g.V_freeze > g.V_safe_route).sum())
        print(f"alpha={alpha}: freezing beats the safe route in {k}/20 decoder seeds; optimal outcomes: {g.optimal_outcome.value_counts().to_dict()}")
