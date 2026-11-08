"""v2 sweeps, run in parallel. Writes three tidy CSVs per experiment into results/.

    python experiments/sweep.py --smoke                       # a few short runs
    python experiments/sweep.py --estimate                    # time one full job per learner
    python experiments/sweep.py --experiment main --learner q # etc.

Experiments
  main      alpha x brain condition x {centered, uncentered}; Q: 20 seeds, PPO: alpha {0,1,4}, 10 seeds, centered only
  ablation  habituation brain only; habituation_decay x habituation_tau x alpha {1, 4}; Q: 20 seeds, PPO: 5 seeds

Outputs (prefix = experiment name)
  <prefix>_runs.csv         one row per run: greedy-eval means + late-training error-location fractions
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
PPO_ALPHAS = [0.0, 1.0, 4.0]
DECAYS = [0.9, 0.8, 0.6]           # habituation strength: weak -> strong
TAUS = [100.0, 500.0, 2500.0]      # recovery time constant in env steps: fast -> slow
SEEDS = {"q": 20, "ppo": 10}
ABLATION_SEEDS = {"q": 20, "ppo": 5}
BIN = 5_000
KEYS = ["learner", "condition", "alpha", "seed", "centered", "habituation_decay", "habituation_tau"]
TRAIN_METRICS = ["true_return", "neural_return", "agent_return", "reached_goal", "steps", "n_errors",
                 "errors_in_fov", "errors_out_fov", "repeated_errors", "first_time_errors",
                 "n_hazard", "n_wall", "n_away", "mean_hab_factor", "stuckness", "repeat_frac"]


def jobs_for(experiment: str, learner: str, smoke: bool) -> list[dict]:
    if experiment == "main":
        alphas = ALPHAS if learner == "q" else PPO_ALPHAS
        centered = [True, False] if learner == "q" else [True]
        seeds = range(SEEDS[learner])
        grid = [dict(alpha=a, condition=c, centered=ce, seed=s)
                for a, c, ce, s in itertools.product(alphas, CONDITIONS, centered, seeds)]
    else:
        grid = [dict(alpha=a, condition="habituation", centered=True, seed=s, habituation_decay=d, habituation_tau=t)
                for d, t, a, s in itertools.product(DECAYS, TAUS, [1.0, 4.0], range(ABLATION_SEEDS[learner]))]
    if smoke:
        grid = [g for g in grid if g["seed"] == 0 and g["alpha"] in (0.0, 4.0)
                and g["condition"] in ("honest", "habituation")
                and g.get("habituation_decay", 0.6) == 0.6 and g.get("habituation_tau", 500.0) == 500.0]
    return [dict(learner=learner, **g) for g in grid]


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _job(args):
    job, total_steps = args
    train_cfg = TrainConfig(total_steps=total_steps, centered=job["centered"])
    brain_cfg = BrainConfig()
    if "habituation_decay" in job:
        brain_cfg = replace(brain_cfg, habituation_decay=job["habituation_decay"], habituation_tau=job["habituation_tau"])
    df = run(job["alpha"], job["condition"], job["seed"], job["learner"], train_cfg=train_cfg, brain_cfg=brain_cfg)
    return summarize(df)


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collapse one run's episode log into the three output tables."""
    ids = df.iloc[0][KEYS].to_dict()
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
    ap.add_argument("--experiment", choices=["main", "ablation", "both"], default="both")
    ap.add_argument("--learner", choices=["q", "ppo", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    args = ap.parse_args()
    learners = ["q", "ppo"] if args.learner == "both" else [args.learner]
    experiments = ["main", "ablation"] if args.experiment == "both" else [args.experiment]

    if args.estimate:
        for learner in learners:
            t0 = time.time()
            _job((dict(learner=learner, alpha=1.0, condition="honest", centered=True, seed=0), args.total_steps))
            per = time.time() - t0
            for exp in experiments:
                n = len(jobs_for(exp, learner, False))
                print(f"{learner:3s} {exp:8s}: {n:4d} jobs x {per:.1f}s / {args.workers} workers ~ {n * per / args.workers / 60:.1f} min (uncontended)")
        return

    RESULTS.mkdir(exist_ok=True)
    for exp in experiments:
        jobs = [j for l in learners for j in jobs_for(exp, l, args.smoke)]
        steps = min(args.total_steps, 20_000) if args.smoke else args.total_steps
        prefix = f"smoke_{exp}" if args.smoke else exp
        print(f"[{exp}] {len(jobs)} jobs, {steps} steps each, {args.workers} workers", flush=True)
        t0 = time.time()
        parts: dict[str, list] = {"runs": [], "checkpoints": [], "curves": []}
        with Pool(args.workers, initializer=_init_worker) as pool:
            for i, out in enumerate(pool.imap_unordered(_job, [(j, steps) for j in jobs]), 1):
                for k in parts:
                    parts[k].append(out[k])
                r = out["runs"].iloc[0]
                print(f"[{exp} {i:4d}/{len(jobs)}] {r.learner:3s} {r.condition:11s} a={r.alpha:<4} c={int(r.centered)} "
                      f"d={r.habituation_decay} tau={r.habituation_tau:<6} seed={r.seed:<2} "
                      f"true={r.true_return:+.2f} goal={r.reached_goal:.2f} stuck={r.stuckness:.2f} ({time.time() - t0:.0f}s)", flush=True)
        for k, frames in parts.items():
            out = RESULTS / f"{prefix}_{k}.csv"
            if (args.learner != "both" or args.smoke) and out.exists() and not args.smoke:
                old = pd.read_csv(out)
                old = old[~old.learner.isin(learners)]
                frames = [old] + frames
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[{exp}] wrote {prefix}_{{runs,checkpoints,curves}}.csv in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
