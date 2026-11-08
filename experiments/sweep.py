"""Run the alpha x brain-condition x seed x learner sweep in parallel.

    python experiments/sweep.py --smoke          # 4 quick jobs, ~1 min
    python experiments/sweep.py --estimate       # time one job per learner, print the ETA
    python experiments/sweep.py                  # full sweep -> results/sweep.csv

Output: one tidy CSV with one row per episode (train and eval phases).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlnf.config import CONDITIONS, TrainConfig  # noqa: E402
from rlnf.train import run  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = [0, 1, 2, 3, 4]
LEARNERS = ["q", "ppo"]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _init_worker():
    # One thread per worker: the MLP is tiny, so threading only adds contention.
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _job(args):
    alpha, condition, seed, learner, total_steps = args
    cfg = TrainConfig(total_steps=total_steps)
    return run(alpha, condition, seed, learner, train_cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 alphas x 2 conditions x 1 seed, short runs")
    ap.add_argument("--estimate", action="store_true", help="time one full-length job per learner and exit")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    alphas, conditions, seeds, steps = ALPHAS, list(CONDITIONS), SEEDS, args.total_steps
    out = args.out or RESULTS / "sweep.csv"
    if args.smoke:
        alphas, conditions, seeds, steps = [0.0, 1.0], ["honest", "both"], [0], min(steps, 20_000)
        out = args.out or RESULTS / "smoke.csv"
    if args.estimate:
        for learner in LEARNERS:
            t0 = time.time()
            _job((1.0, "honest", 0, learner, steps))
            print(f"{learner}: {time.time() - t0:.1f}s for one job of {steps} steps")
        n = len(ALPHAS) * len(CONDITIONS) * len(SEEDS)
        print(f"full sweep = {n} jobs per learner on {args.workers} workers")
        return

    jobs = [(a, c, s, l, steps) for a, c, s, l in itertools.product(alphas, conditions, seeds, LEARNERS)]
    print(f"{len(jobs)} jobs, {steps} steps each, {args.workers} workers -> {out}")
    t0 = time.time()
    frames = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, df in enumerate(pool.imap_unordered(_job, jobs), 1):
            frames.append(df)
            r = df.iloc[0]
            ev = df[df.phase == "eval"]
            print(f"[{i:3d}/{len(jobs)}] {r.learner:3s} {r.condition:11s} alpha={r.alpha:<4} seed={r.seed} "
                  f"eval true={ev.true_return.mean():+.2f} goal={ev.reached_goal.mean():.2f} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    out.parent.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
