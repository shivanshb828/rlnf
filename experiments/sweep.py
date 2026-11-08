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
  variance  Q, centered, alpha 1: noise_sigma x {honest, visibility}; per-region reward sums for the variance check
  fixes     Q variants (plain, double, maxmin N=2/4/8, ASRN): {honest, visibility} x sigma {0.1, 0.3, 0.5}, alpha 1
  explore2  Q, visibility, alpha 1, sigma 0.3: epsilon schedules, with per-region Q-values and time shares logged
  absent    absent_supervisor env: {honest, visibility} x sigma {0.03, 0.3} x {plain, BEST_FIX}, alpha 1
  ppo_diag  our PPO with critic diagnostics: {honest, visibility} x sigma {0.03, 0.3} x {default, gae 0.8, 3 critics}
  dqn       DQN vs Double DQN: {honest, visibility} x sigma {0.03, 0.3}
  observ    observability-aware reward: {maze, absent_supervisor} x {honest, visibility} x sigma x {plain, double} x 4 modes
  realnoise brain driven by real held-out ErrP decoder outputs (subject = seed mod 6): {honest, visibility} x {plain, double}
  praise_a  does praise fix freezing? visibility x sigma {0.03, 0.3} x praise_ratio {0, .25, .5, 1}, surprise weight 0
  praise_b  does praise create surprise-seeking? honest x surprise weight {0, .5, 1} x alpha {.25, 1, 4}, praise_ratio .5
  praise_c  combined: visibility, sigma .3, praise .5, surprise weight {0, 1}, alpha {1, 4}
  praise_flip one small run: honest, praise .5, valence flipped (competitor observer)
  att_distractor attention maze: gaze observer with the wave action, k {2, 5, 10}; controls: inert wave, attentive honest
  att_occluder   attention maze with a pushable box; control: transparent box; attentive honest
  att_fixes      strongest distractor (k 10) and occluder with the masked / imputed observability-aware rewards

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
from rlnf.attention.env import AttentionConfig  # noqa: E402
from rlnf.praise.brain import PraiseConfig  # noqa: E402
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
         ("noise", "q"): 20, ("explore", "q"): 20, ("ppo_norm", "ppo"): 10,
         ("variance", "q"): 10, ("fixes", "q"): 20, ("explore2", "q"): 20, ("absent", "q"): 20,
         ("ppo_diag", "ppo_torch"): 10, ("dqn", "dqn"): 10, ("dqn", "ddqn"): 10, ("observ", "q"): 20, ("realnoise", "q"): 20,
         ("praise_a", "q"): 20, ("praise_a", "ppo_torch"): 10, ("praise_b", "q"): 20, ("praise_c", "q"): 20, ("praise_flip", "q"): 10,
         ("att_distractor", "q"): 20, ("att_occluder", "q"): 20, ("att_fixes", "q"): 20}
ATTENTION = {
    "gaze_k2": ("visibility", dict(distractor_pull=2)), "gaze_k5": ("visibility", dict(distractor_pull=5)),
    "gaze_k10": ("visibility", dict(distractor_pull=10)),
    "control_inert_wave": ("visibility", dict(distractor_effective=False)),
    "control_attentive_honest": ("honest", dict(attentive=True)),
    "occluder": ("visibility", dict(distractor=False, occluder=True)),
    "occluder_transparent": ("visibility", dict(distractor=False, occluder=True, occluder_blocks=False)),
    "occluder_attentive_honest": ("honest", dict(distractor=False, occluder=True, attentive=True)),
}
PPO_VARIANTS = {"default": {}, "gae0.8": dict(ppo_gae_lambda=0.8), "critics3": dict(ppo_n_critics=3)}
OBSERVABILITY = ["baseline", "masked", "imputed", "pessimistic"]
VARIANTS = {"plain": dict(q_variant="plain"), "double": dict(q_variant="double"),
            "maxmin2": dict(q_variant="maxmin", q_maxmin_n=2), "maxmin4": dict(q_variant="maxmin", q_maxmin_n=4),
            "maxmin8": dict(q_variant="maxmin", q_maxmin_n=8), "asrn": dict(q_variant="asrn")}
BEST_FIX = "double"   # Double Q-learning: 0/20 frozen at sigma 0.1 and 0.3 in the `fixes` experiment
ABSENT_TRAIN = dict(q_init=-2.0)   # -step_penalty / (1 - gamma) for that env's step cost of 0.02
BIN = 5_000
TRAIN_METRICS = ["true_return", "neural_return", "agent_return", "reached_goal", "steps", "n_errors",
                 "errors_in_fov", "errors_out_fov", "repeated_errors", "first_time_errors",
                 "n_hazard", "n_wall", "n_away", "mean_hab_factor", "stuckness", "repeat_frac", "hidden_share",
                 "n_visible", "r_sum_visible", "r_sq_visible", "n_hidden", "r_sum_hidden", "r_sq_hidden",
                 "q_visible", "q_hidden", "mean_surprise", "frac_unpredictable", "action_entropy", "praise_return",
                 "good_events", "extra_steps", "observed_frac", "waves", "hazards_after_wave", "hazards_unobserved", "pushes",
                 "frac_hazards_after_wave", "frac_hazards_unobserved"]


def job(learner, condition, alpha, seed, train=None, brain=None, env_name="maze", praise=None, attention=None, **tags) -> dict:
    """A job = run() arguments + config overrides + tag columns copied into the output rows."""
    return dict(learner=learner, condition=condition, alpha=alpha, seed=seed, env_name=env_name,
                train=train or {}, brain=brain or {}, praise=praise, attention=attention, tags=dict(env_name=env_name, **tags))


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
    if experiment == "variance":
        return [job(learner, c, 1.0, s, brain=dict(noise_sigma=sig), noise_sigma=sig)
                for sig, c, s in itertools.product(SIGMAS, ["honest", "visibility"], seeds)]
    if experiment == "fixes":
        return [job(learner, c, 1.0, s, train=VARIANTS[v], brain=dict(noise_sigma=sig), variant=v, noise_sigma=sig)
                for v, c, sig, s in itertools.product(VARIANTS, ["honest", "visibility"], [0.1, 0.3, 0.5], seeds)]
    if experiment == "explore2":
        return [job(learner, "visibility", 1.0, s, train=EXPLORATION[name], exploration=name, **EXPLORATION[name])
                for name, s in itertools.product(EXPLORATION, seeds)]
    if experiment == "absent":
        return [job(learner, c, 1.0, s, env_name="absent_supervisor", train={**ABSENT_TRAIN, **VARIANTS[v]},
                    brain=dict(noise_sigma=sig), variant=v, noise_sigma=sig)
                for v, c, sig, s in itertools.product(["plain", BEST_FIX], ["honest", "visibility"], [0.03, 0.3], seeds)]
    if experiment == "ppo_diag":
        return [job(learner, c, 1.0, s, train=PPO_VARIANTS[v], brain=dict(noise_sigma=sig), variant=v, noise_sigma=sig)
                for v, c, sig, s in itertools.product(PPO_VARIANTS, ["honest", "visibility"], [0.03, 0.3], seeds)]
    if experiment == "dqn":
        return [job(learner, c, 1.0, s, brain=dict(noise_sigma=sig), variant=learner, noise_sigma=sig)
                for c, sig, s in itertools.product(["honest", "visibility"], [0.03, 0.3], seeds)]
    if experiment == "observ":
        return [job(learner, c, 1.0, s, env_name=env, brain=dict(noise_sigma=sig), variant=v, noise_sigma=sig, observability=m,
                    train={**(ABSENT_TRAIN if env == "absent_supervisor" else {}), **VARIANTS[v], "observability": m})
                for env, c, sig, v, m, s in itertools.product(["maze", "absent_supervisor"], ["honest", "visibility"], [0.03, 0.3],
                                                              ["plain", "double"], OBSERVABILITY, seeds)]
    if experiment == "praise_a":
        learners_v = ["plain", "double"] if learner == "q" else ["ppo"]
        sig_ratios = [(sg, r) for sg in [0.03, 0.3] for r in [0.0, 0.25, 0.5, 1.0]] if learner == "q" else [(0.3, 0.0), (0.3, 0.5)]
        return [job(learner, "visibility", 1.0, s, train=VARIANTS.get(v, {}), brain=dict(noise_sigma=sg),
                    praise=dict(praise_ratio=r, surprise_weight=0.0), variant=v, noise_sigma=sg, praise_ratio=r, surprise_weight=0.0)
                for (sg, r), v, s in itertools.product(sig_ratios, learners_v, seeds)]
    if experiment == "praise_b":
        return [job(learner, "honest", a, s, train=VARIANTS[v], praise=dict(praise_ratio=0.5, surprise_weight=w),
                    variant=v, noise_sigma=0.3, praise_ratio=0.5, surprise_weight=w)
                for w, a, v, s in itertools.product([0.0, 0.5, 1.0], [0.25, 1.0, 4.0], ["plain", "double"], seeds)]
    if experiment == "praise_c":
        return [job(learner, "visibility", a, s, train=VARIANTS[v], praise=dict(praise_ratio=0.5, surprise_weight=w),
                    variant=v, noise_sigma=0.3, praise_ratio=0.5, surprise_weight=w)
                for w, a, v, s in itertools.product([0.0, 1.0], [1.0, 4.0], ["plain", "double"], seeds)]
    if experiment == "praise_flip":
        return [job(learner, "honest", 1.0, s, praise=dict(praise_ratio=0.5, surprise_weight=0.0, valence_flip=True),
                    variant="plain", noise_sigma=0.3, praise_ratio=0.5, surprise_weight=0.0, valence_flip=True) for s in seeds]
    if experiment == "att_distractor":
        names = ["gaze_k2", "gaze_k5", "gaze_k10", "control_inert_wave", "control_attentive_honest"]
        return [job(learner, ATTENTION[n][0], 1.0, s, env_name="attention", train=VARIANTS[v], attention=ATTENTION[n][1],
                    variant=v, setting=n, noise_sigma=0.3, observability="baseline")
                for n, v, s in itertools.product(names, ["plain", "double"], seeds)]
    if experiment == "att_occluder":
        names = ["occluder", "occluder_transparent", "occluder_attentive_honest"]
        return [job(learner, ATTENTION[n][0], 1.0, s, env_name="attention", train=VARIANTS[v], attention=ATTENTION[n][1],
                    variant=v, setting=n, noise_sigma=0.3, observability="baseline")
                for n, v, s in itertools.product(names, ["plain", "double"], seeds)]
    if experiment == "att_fixes":
        return [job(learner, "visibility", 1.0, s, env_name="attention", train={**VARIANTS[v], "observability": m},
                    attention=ATTENTION[n][1], variant=v, setting=n, noise_sigma=0.3, observability=m)
                for n, m, v, s in itertools.product(["gaze_k10", "occluder"], ["masked", "imputed"], ["plain", "double"], seeds)]
    if experiment == "realnoise":
        return [job(learner, c, 1.0, s, train={**VARIANTS[v], "real_errp": True}, variant=v, subject=s % 6 + 1)
                for v, c, s in itertools.product(["plain", "double"], ["honest", "visibility"], seeds)]
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
    praise_cfg = PraiseConfig(**j["praise"]) if j.get("praise") is not None else None
    attention_cfg = AttentionConfig(**j["attention"]) if j.get("attention") is not None else None
    df = run(j["alpha"], j["condition"], j["seed"], j["learner"], train_cfg=train_cfg, brain_cfg=brain_cfg,
             env_name=j.get("env_name", "maze"), praise_cfg=praise_cfg, attention_cfg=attention_cfg)
    return summarize(df, j["tags"])


def summarize(df: pd.DataFrame, tags: dict) -> dict[str, pd.DataFrame]:
    """Collapse one run's episode log into the three output tables."""
    ids = {**df.iloc[0][["learner", "condition", "alpha", "seed", "centered", "habituation_decay", "habituation_tau"]].to_dict(), **tags}
    tr, ck, ev = (df[df.phase == p] for p in ("train", "checkpoint", "eval"))
    late = tr[tr.step > 0.8 * tr.step.max()]
    metrics = [k for k in TRAIN_METRICS if k in tr.columns]          # PPO rows have no q_visible / q_hidden
    runs = {**ids, **{k: ev[k].mean() for k in metrics if k in ev.columns},
            **{f"late_{k}": late[k].mean() for k in ("n_visible", "r_sum_visible", "r_sq_visible", "n_hidden",
                                                       "r_sum_hidden", "r_sq_hidden", "hidden_share")},
            "frac_out_fov_late": late.errors_out_fov.sum() / max(1, late.n_errors.sum()),
            "n_errors_late": late.n_errors.mean(), "stuckness_late": late.stuckness.mean(),
            "goal_rate_late": late.reached_goal.mean(), "wall_time_s": df.wall_time_s.iloc[0]}
    ck_cols = ["step", "kl", "true_return", "neural_return", "reached_goal", "stuckness", "repeat_frac",
               "n_errors", "errors_out_fov", "n_hazard", "steps"] + [c for c in ck.columns if c.startswith(("v_", "mc_"))]
    checkpoints = ck[ck_cols].assign(**ids)
    if (df.phase == "update").any():   # our PPO: per-update advantage statistics by region
        up = df[df.phase == "update"]
        checkpoints = pd.concat([checkpoints, up[["step"] + [c for c in up.columns if c.startswith(("adv_", "hidden_share", "return_scale"))]]
                                 .assign(phase="update", **ids)], ignore_index=True)
    curves = (tr.assign(bin=(tr.step // BIN) * BIN + BIN).groupby("bin")[metrics].mean()
              .reset_index().assign(**ids))
    return dict(runs=pd.DataFrame([runs]), checkpoints=checkpoints, curves=curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["main", "ablation", "noise", "explore", "ppo_norm",
                                             "variance", "fixes", "explore2", "absent", "ppo_diag", "dqn", "observ", "realnoise", "praise_a", "praise_b", "praise_c", "praise_flip", "att_distractor", "att_occluder", "att_fixes"], nargs="+",
                    default=["main", "ablation"])
    ap.add_argument("--learner", choices=["q", "ppo", "both", "ppo_torch", "dqn", "ddqn", "deep"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    ap.add_argument("--filter", default=None, help="key=value: run only jobs whose field or tag matches (e.g. condition=honest)")
    args = ap.parse_args()
    learners = {"both": ["q", "ppo"], "deep": ["ppo_torch", "dqn", "ddqn"]}.get(args.learner, [args.learner])
    RESULTS.mkdir(exist_ok=True)

    for exp in args.experiment:
        jobs = [j for l in learners for j in jobs_for(exp, l)]
        if args.filter:
            k, v = args.filter.split("=")
            jobs = [j for j in jobs if str(j.get(k, j["tags"].get(k))) == v]
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
            if out.exists() and not args.smoke and (args.learner != "both" or args.filter):
                old = pd.read_csv(out)              # keep rows this invocation did not re-run
                keep = ~old.learner.isin(learners)
                if args.filter:
                    k, v = args.filter.split("=")
                    keep = keep | (old[k].astype(str) != v)
                frames = [old[keep]] + frames
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[{exp}] wrote {prefix}_{{runs,checkpoints,curves}}.csv in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
