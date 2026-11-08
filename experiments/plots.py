"""Plots for the sweep. Reads results/sweep.csv, writes PNGs next to it.

    python experiments/plots.py                     # results/sweep.csv -> results/*.png
    python experiments/plots.py results/smoke.csv   # any other sweep file

Figures (one per learner, learners side by side where sensible):
  curves_<learner>.png  neural and true return vs training steps, one line per alpha,
                        one row per brain condition, mean +/- 95% CI across seeds
  headline.png          final (greedy eval) true return vs alpha, one line per condition
  hiding.png            fraction of eval errors outside the FOV vs alpha, per condition
  summary.csv           the numbers behind headline.png and hiding.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = ["honest", "visibility", "habituation", "both"]
LEARNER_NAMES = {"q": "Tabular Q-learning", "ppo": "PPO"}
# Categorical palette (validated, fixed order) + marker as a second cue.
COND_COLOR = dict(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
COND_MARKER = dict(zip(CONDITIONS, ["o", "s", "^", "D"]))
# Ordinal ramp for alpha: gray for alpha = 0 (no neural reward), then one blue hue light -> dark.
ALPHA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BIN = 5_000  # training steps per bin for the curves

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def alpha_colors(alphas) -> dict:
    alphas = sorted(alphas)
    nonzero = [a for a in alphas if a > 0]
    ramp = [ALPHA_RAMP[int(round(i * (len(ALPHA_RAMP) - 1) / max(1, len(nonzero) - 1)))] for i in range(len(nonzero))]
    return {0.0: MUTED, **dict(zip(nonzero, ramp))}


def mean_ci(values: pd.Series) -> tuple[float, float]:
    """Mean and 95% CI half-width across seeds (t-free: 1.96 * sem)."""
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


KEYS = ["learner", "condition", "alpha", "seed"]


def per_run_eval(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: greedy-eval means, plus error-location fractions over late training.

    The FOV fractions use the last 20% of *training* episodes rather than eval:
    a converged honest-brain policy makes no errors at all in eval, so its
    fraction would be undefined, while exploration noise during training
    reveals where each agent's mistakes happen.
    """
    ev = df[df.phase == "eval"]
    out = ev.groupby(KEYS)[["true_return", "neural_return", "reached_goal", "steps", "n_errors", "n_hazard"]].mean()
    tr = df[df.phase == "train"]
    late = tr[tr.step > 0.8 * tr.groupby(KEYS).step.transform("max")]
    tot = late.groupby(KEYS)[["errors_out_fov", "n_errors", "repeated_errors"]].sum()
    out["frac_out_fov"] = tot.errors_out_fov / tot.n_errors.replace(0, np.nan)
    out["frac_repeated"] = tot.repeated_errors / tot.n_errors.replace(0, np.nan)
    return out.reset_index()


def plot_curves(df: pd.DataFrame, learner: str, out: Path) -> None:
    d = df[(df.learner == learner) & (df.phase == "train")].copy()
    d["bin"] = (d.step // BIN) * BIN + BIN
    colors = alpha_colors(d.alpha.unique())
    conds = [c for c in CONDITIONS if c in set(d.condition)]
    fig, axes = plt.subplots(len(conds), 2, figsize=(9, 2.4 * len(conds)), sharex=True, squeeze=False)
    for i, cond in enumerate(conds):
        for j, metric in enumerate(["neural_return", "true_return"]):
            ax = axes[i, j]
            for alpha, da in d[d.condition == cond].groupby("alpha"):
                per_seed = da.groupby(["bin", "seed"])[metric].mean().unstack("seed")
                m = per_seed.mean(axis=1)
                n = per_seed.notna().sum(axis=1)
                ci = 1.96 * per_seed.std(axis=1, ddof=1) / np.sqrt(n)
                ax.plot(m.index, m, color=colors[alpha], lw=1.6, label=f"α = {alpha:g}")
                if (n > 1).any():
                    ax.fill_between(m.index, m - ci, m + ci, color=colors[alpha], alpha=0.15, lw=0)
            ax.set_title(f"{cond} brain — {metric.replace('_', ' ')}", loc="left", fontsize=9)
            if i == len(conds) - 1:
                ax.set_xlabel("training steps")
    axes[0, 0].legend(ncol=2, fontsize=8, title="neural reward weight", title_fontsize=8)
    fig.suptitle(f"{LEARNER_NAMES[learner]}: training curves (mean ± 95% CI over seeds, {BIN}-step bins)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_alpha(runs: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, ylim=None) -> None:
    learners = [l for l in ["q", "ppo"] if l in set(runs.learner)]
    fig, axes = plt.subplots(1, len(learners), figsize=(4.4 * len(learners), 3.4), sharey=True, squeeze=False)
    for ax, learner in zip(axes[0], learners):
        r = runs[runs.learner == learner]
        for cond in CONDITIONS:
            rc = r[r.condition == cond]
            if rc.empty:
                continue
            stats = rc.groupby("alpha")[metric].apply(mean_ci)
            x = np.array(stats.index, dtype=float)
            m = np.array([s[0] for s in stats]); ci = np.array([s[1] for s in stats])
            ax.errorbar(x, m, yerr=ci, color=COND_COLOR[cond], marker=COND_MARKER[cond], ms=5, lw=1.6,
                        capsize=2, label=cond, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(sorted(runs.alpha.unique()))
        ax.set_xticklabels([f"{a:g}" for a in sorted(runs.alpha.unique())])
        ax.set_xlabel("neural reward weight α")
        ax.set_title(LEARNER_NAMES[learner], loc="left", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="brain condition", fontsize=8, title_fontsize=8,
               loc="lower center", ncol=len(labels) + 1, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(path: Path) -> None:
    df = pd.read_csv(path)
    out_dir = path.parent
    runs = per_run_eval(df)
    summary = runs.groupby(["learner", "condition", "alpha"]).agg(
        true_return=("true_return", "mean"), true_return_sd=("true_return", "std"),
        neural_return=("neural_return", "mean"), goal_rate=("reached_goal", "mean"),
        steps=("steps", "mean"), n_errors=("n_errors", "mean"), n_hazard=("n_hazard", "mean"),
        frac_out_fov=("frac_out_fov", "mean"), frac_repeated=("frac_repeated", "mean"), n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    for learner in sorted(df.learner.unique()):
        plot_curves(df, learner, out_dir / f"curves_{learner}.png")
    n_seeds = runs.seed.nunique()
    plot_by_alpha(runs, "true_return", "final true return (greedy eval)",
                  f"Headline: true task return after training vs neural reward weight (mean ± 95% CI, {n_seeds} seeds)",
                  out_dir / "headline.png", ylim=(-1.1, 1.0))
    plot_by_alpha(runs, "frac_out_fov", "fraction of errors outside the FOV",
                  "Hiding mistakes: share of errors outside the FOV in the last 20% of training (mean ± 95% CI)",
                  out_dir / "hiding.png", ylim=(-0.05, 1.05))
    print(f"wrote plots to {out_dir}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "sweep.csv")
