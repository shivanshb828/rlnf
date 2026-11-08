"""Sweeps, run in parallel. Each experiment writes three tidy CSVs into results/.

    python experiments/sweep.py --experiment main --learner q      # v2 main sweep
    python experiments/sweep.py --experiment noise                  # v2.1 decoder-noise dose-response
    python experiments/sweep.py --smoke --experiment noise          # a handful of short runs
    python experiments/sweep.py --estimate --experiment ppo_norm    # time one job and print the ETA

Experiments (seed counts in SEEDS)
  main      alpha x brain condition x {centered, uncentered}; PPO on alpha {0,1,4}, centered, no reward norm
  ablation  habituation brain; habituation_decay x habituation_tau x alpha {1, 4}
  noise     Q, centered: noise_sigma x {honest, visibility} x alpha {0.25, 1, 4}
  explore   Q, visibility, alpha 1, sigma 0.3: learning rate x exploration schedule
  ppo_norm  PPO with VecNormalize reward scaling: {honest, visibility} x alpha {1, 4}

Outputs (prefix = experiment name)
  <prefix>_runs.csv         one row per run: greedy-eval means + late-training aggregates + the run's tags
  <prefix>_checkpoints.csv  one row per checkpoint: greedy-eval means + KL from the 10% policy
  <prefix>_curves.csv       one row per run per 5k-step bin: training-episode means
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlnf.config import CONDITIONS, BrainConfig, TrainConfig  # noqa: E402
from rlnf.train import run  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
DECAYS = [0.9, 0.8, 0.6]           # habituation strength: weak -> strong
TAUS = [100.0, 500.0, 2500.0]      # recovery time constant in env steps: fast -> slow
SIGMAS = [0.03, 0.1, 0.2, 0.3, 0.5, 0.8]
LEARNING_RATES = [0.02, 0.1, 0.3]
EXPLORATION = {"low": dict(q_eps_end=0.01, q_eps_fraction=0.25),      # epsilon 1 -> 0.01 over the first quarter
               "default": dict(q_eps_end=0.05, q_eps_fraction=0.5),
               "high": dict(q_eps_end=0.3, q_eps_fraction=0.9)}       # epsilon 1 -> 0.3 over 90% of training
SEEDS = {("main", "q"): 20, ("main", "ppo"): 10, ("ablation", "q"): 20, ("ablation", "ppo"): 5,
         ("noise", "q"): 20, ("explore", "q"): 20, ("ppo_norm", "ppo"): 10}
BIN = 5_000
TRAIN_METRICS = ["true_return", "neural_return", "agent_return", "reached_goal", "steps", "n_errors",
                 "errors_in_fov", "errors_out_fov", "repeated_errors", "first_time_errors",
                 "n_hazard", "n_wall", "n_away", "mean_hab_factor", "stuckness", "repeat_frac"]


def job(learner, condition, alpha, seed, train=None, brain=None, **tags) -> dict:
    """A job = run() arguments + config overrides + tag columns copied into the output rows."""
    return dict(learner=learner, condition=condition, alpha=alpha, seed=seed,
                train=train or {}, brain=brain or {}, tags=tags)


def jobs_for(experiment: str, learner: str) -> list[dict]:
    n = SEEDS.get((experiment, learner), 0)
    seeds = range(n)
    if experiment == "main":
        alphas, centered = (ALPHAS, [True, False]) if learner == "q" else ([0.0, 1.0, 4.0], [True])
        return [job(learner, c, a, s, train=dict(centered=ce, ppo_normalize_reward=False))
                for a, c, ce, s in itertools.product(alphas, CONDITIONS, centered, seeds)]
    if experiment == "ablation":
        return [job(learner, "habituation", a, s, train=dict(ppo_normalize_reward=False),
                    brain=dict(habituation_decay=d, habituation_tau=t))
                for d, t, a, s in itertools.product(DECAYS, TAUS, [1.0, 4.0], seeds)]
    if experiment == "noise":
        return [job(learner, c, a, s, brain=dict(noise_sigma=sig), noise_sigma=sig)
                for sig, c, a, s in itertools.product(SIGMAS, ["honest", "visibility"], [0.25, 1.0, 4.0], seeds)]
    if experiment == "explore":
        return [job(learner, "visibility", 1.0, s, train=dict(q_lr=lr, **EXPLORATION[name]),
                    q_lr=lr, exploration=name, **EXPLORATION[name])
                for lr, name, s in itertools.product(LEARNING_RATES, EXPLORATION, seeds)]
    if experiment == "ppo_norm":
        return [job(learner, c, a, s, train=dict(ppo_normalize_reward=True), ppo_normalize_reward=True)
                for c, a, s in itertools.product(["honest", "visibility"], [1.0, 4.0], seeds)]
    raise ValueError(experiment)


def smoke_subset(jobs: list[dict]) -> list[dict]:
    """Seed 0 only, extreme alphas only, at most 8 jobs."""
    picked = [j for j in jobs if j["seed"] == 0]
    alphas = sorted({j["alpha"] for j in picked})
    picked = [j for j in picked if j["alpha"] in (alphas[0], alphas[-1])]
    return picked[:: max(1, len(picked) // 8)][:8]


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _job(args):
    j, total_steps = args
    train_cfg = replace(TrainConfig(total_steps=total_steps), **j["train"])
    brain_cfg = replace(BrainConfig(), **j["brain"])
    df = run(j["alpha"], j["condition"], j["seed"], j["learner"], train_cfg=train_cfg, brain_cfg=brain_cfg)
    return summarize(df, j["tags"])


def summarize(df: pd.DataFrame, tags: dict) -> dict[str, pd.DataFrame]:
    """Collapse one run's episode log into the three output tables."""
    ids = {**df.iloc[0][["learner", "condition", "alpha", "seed", "centered", "habituation_decay", "habituation_tau"]].to_dict(), **tags}
    tr, ck, ev = (df[df.phase == p] for p in ("train", "checkpoint", "eval"))
    late = tr[tr.step > 0.8 * tr.step.max()]
    runs = {**ids, **{k: ev[k].mean() for k in TRAIN_METRICS},
            "frac_out_fov_late": late.errors_out_fov.sum() / max(1, late.n_errors.sum()),
            "n_errors_late": late.n_errors.mean(), "stuckness_late": late.stuckness.mean(),
            "goal_rate_late": late.reached_goal.mean(), "wall_time_s": df.wall_time_s.iloc[0]}
    checkpoints = ck[["step", "kl", "true_return", "neural_return", "reached_goal", "stuckness", "repeat_frac",
                      "n_errors", "errors_out_fov", "n_hazard", "steps"]].assign(**ids)
    curves = (tr.assign(bin=(tr.step // BIN) * BIN + BIN).groupby("bin")[TRAIN_METRICS].mean()
              .reset_index().assign(**ids))
    return dict(runs=pd.DataFrame([runs]), checkpoints=checkpoints, curves=curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["main", "ablation", "noise", "explore", "ppo_norm"], nargs="+",
                    default=["main", "ablation"])
    ap.add_argument("--learner", choices=["q", "ppo", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    args = ap.parse_args()
    learners = ["q", "ppo"] if args.learner == "both" else [args.learner]
    RESULTS.mkdir(exist_ok=True)

    for exp in args.experiment:
        jobs = [j for l in learners for j in jobs_for(exp, l)]
        if not jobs:
            print(f"[{exp}] no jobs for learner(s) {learners}")
            continue
        if args.estimate:
            t0 = time.time()
            _job((jobs[0], args.total_steps))
            per = time.time() - t0
            print(f"[{exp}] {len(jobs)} jobs x {per:.1f}s / {args.workers} workers ~ {len(jobs) * per / args.workers / 60:.1f} min uncontended")
            continue
        steps = min(args.total_steps, 20_000) if args.smoke else args.total_steps
        if args.smoke:
            jobs = smoke_subset(jobs)
        prefix = f"smoke_{exp}" if args.smoke else exp
        print(f"[{exp}] {len(jobs)} jobs, {steps} steps each, {args.workers} workers", flush=True)
        t0 = time.time()
        parts: dict[str, list] = {"runs": [], "checkpoints": [], "curves": []}
        with Pool(args.workers, initializer=_init_worker) as pool:
            for i, out in enumerate(pool.imap_unordered(_job, [(j, steps) for j in jobs]), 1):
                for k in parts:
                    parts[k].append(out[k])
                r = out["runs"].iloc[0]
                print(f"[{exp} {i:4d}/{len(jobs)}] {r.learner:3s} {r.condition:11s} a={r.alpha:<4} seed={r.seed:<2} "
                      f"true={r.true_return:+.2f} goal={r.reached_goal:.2f} stuck={r.stuckness:.2f} ({time.time() - t0:.0f}s)", flush=True)
        for k, frames in parts.items():
            out = RESULTS / f"{prefix}_{k}.csv"
            if out.exists() and not args.smoke and args.learner != "both":
                old = pd.read_csv(out)              # keep the other learner's rows
                frames = [old[~old.learner.isin(learners)]] + frames
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[{exp}] wrote {prefix}_{{runs,checkpoints,curves}}.csv in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
