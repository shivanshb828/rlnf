"""v2 plots. Reads results/{main,ablation}_*.csv, writes PNGs + summary tables to results/.

    python experiments/plots.py

Figures
  overopt_<learner>.png   headline: true return (top) and neural return (bottom) vs KL from the
                          10% policy, one panel per brain condition, one trajectory per alpha
  final_true_return.png   final true return vs alpha; Q centered | Q uncentered | PPO centered
  stuckness_alpha.png     eval stuckness vs alpha per brain condition (same three panels)
  stuckness_ablation.png  stuckness and goal rate vs habituation strength, lines per tau
  hiding.png              fraction of late-training errors outside the FOV vs alpha
Tables
  summary_main.csv, summary_ablation.csv   means and "k of n seeds" counts behind every claim
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONDITIONS = ["honest", "visibility", "habituation", "both"]
LEARNER_NAMES = {"q": "Tabular Q-learning", "ppo": "PPO"}
COND_COLOR = dict(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))   # validated categorical order
COND_MARKER = dict(zip(CONDITIONS, ["o", "s", "^", "D"]))
ALPHA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]               # one hue, light -> dark
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
FROZEN = 0.3      # eval stuckness above this counts a seed as "frozen"
HAZARD_ROUTE = 0.5  # mean eval hazard entries above this counts a seed as "took the hazard route"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def alpha_colors(alphas) -> dict:
    nonzero = sorted(a for a in set(alphas) if a > 0)
    idx = np.linspace(0, len(ALPHA_RAMP) - 1, len(nonzero)).round().astype(int) if nonzero else []
    return {0.0: MUTED, **{a: ALPHA_RAMP[i] for a, i in zip(nonzero, idx)}}


def ci95(v: pd.Series) -> float:
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def panels(n: int, width: float = 4.2, height: float = 3.3):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), sharey=True, squeeze=False)
    return fig, axes[0]


def finish(fig, axes, title: str, out: Path, legend_from=None) -> None:
    handles, labels = (legend_from or axes[0]).get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alpha_axis(ax, alphas) -> None:
    alphas = sorted(set(alphas))
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel("neural reward weight α")


def by_alpha(ax, runs: pd.DataFrame, metric: str) -> None:
    for cond in CONDITIONS:
        rc = runs[runs.condition == cond]
        if rc.empty:
            continue
        g = rc.groupby("alpha")[metric]
        # Lines are drawn progressively thinner and on top (honest widest, underneath), so a line that
        # exactly coincides with a later one (common here) still peeks out.
        rank = CONDITIONS.index(cond)
        ax.errorbar(g.mean().index, g.mean(), yerr=g.apply(ci95), color=COND_COLOR[cond], marker=COND_MARKER[cond],
                    ms=7 - rank, lw=3.4 - 0.7 * rank, capsize=2, label=cond, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, zorder=1 + rank, alpha=0.9)
    alpha_axis(ax, runs.alpha)


def three_panels(runs: pd.DataFrame):
    """(label, subset) for Q centered, Q uncentered, PPO centered: whichever exist."""
    out = []
    for learner, centered in (("q", True), ("q", False), ("ppo", True)):
        sub = runs[(runs.learner == learner) & (runs.centered == centered)]
        if not sub.empty:
            out.append((f"{LEARNER_NAMES[learner]}, {'centered' if centered else 'uncentered'}", sub))
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_overopt(ck: pd.DataFrame, learner: str, out: Path) -> None:
    d = ck[(ck.learner == learner) & (ck.centered) & ck.kl.notna()]
    colors = alpha_colors(d.alpha.unique())
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(3.6 * len(CONDITIONS), 5.6), sharex="col", sharey="row", squeeze=False)
    for j, cond in enumerate(CONDITIONS):
        dc = d[d.condition == cond]
        for alpha, da in dc.groupby("alpha"):
            traj = da.groupby("step")[["kl", "true_return", "neural_return"]].mean().sort_index()
            for i, metric in enumerate(["true_return", "neural_return"]):
                ax = axes[i, j]
                ax.plot(traj.kl, traj[metric], color=colors[alpha], lw=1.4, marker="o", ms=2.5, label=f"α = {alpha:g}")
                ax.plot(traj.kl.iloc[-1], traj[metric].iloc[-1], marker="o", ms=6, color=colors[alpha],
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
        axes[0, j].set_title(f"{cond} brain", loc="left", fontsize=9)
        axes[1, j].set_xlabel("KL(policy ‖ policy at 10% of training), visited states")
    axes[0, 0].set_ylabel("true return (greedy, 5 episodes)")
    axes[1, 0].set_ylabel("neural return (greedy, 5 episodes)")
    axes[0, 0].set_ylim(-1.45, 1.0)
    axes[1, 0].set_ylim(-20, 1.5)   # alpha = 0 honest sits far below (-40 to -55); it is not optimizing this quantity
    axes[0, 0].legend(fontsize=7, title="α", title_fontsize=8, ncol=2)
    fig.suptitle(f"{LEARNER_NAMES[learner]}: return vs divergence from the early policy "
                 "(each point = one checkpoint, mean over seeds; large dot = end of training)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_metric_vs_alpha(runs: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, ylim=None) -> None:
    groups = three_panels(runs)
    fig, axes = panels(len(groups))
    for ax, (label, sub) in zip(axes, groups):
        by_alpha(ax, sub, metric)
        ax.set_title(label, loc="left", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    finish(fig, axes, title, out)


def plot_ablation(ab: pd.DataFrame, out: Path) -> None:
    ab = ab.assign(strength=1 - ab.habituation_decay)
    taus = sorted(ab.habituation_tau.unique())
    tau_colors = dict(zip(taus, ["#86b6ef", "#2a78d6", "#0d366b"]))
    combos = [(l, a) for l in ("q", "ppo") for a in sorted(ab.alpha.unique()) if not ab[(ab.learner == l) & (ab.alpha == a)].empty]
    fig, axes = plt.subplots(2, len(combos), figsize=(3.4 * len(combos), 5.4), sharex=True, sharey="row", squeeze=False)
    for j, (learner, alpha) in enumerate(combos):
        sub = ab[(ab.learner == learner) & (ab.alpha == alpha)]
        for i, metric in enumerate(["stuckness", "reached_goal"]):
            ax = axes[i, j]
            for tau in taus:
                g = sub[sub.habituation_tau == tau].groupby("strength")[metric]
                ax.errorbar(g.mean().index, g.mean(), yerr=g.apply(ci95), color=tau_colors[tau], marker="o", ms=4,
                            lw=1.5, capsize=2, label=f"τ = {tau:g} steps", markeredgecolor=SURFACE)
            ax.set_xticks(sorted(ab.strength.unique()))
        n = sub.seed.nunique()
        axes[0, j].set_title(f"{LEARNER_NAMES[learner]}, α = {alpha:g} ({n} seeds)", loc="left", fontsize=9)
        axes[1, j].set_xlabel("habituation strength (1 − decay)")
    axes[0, 0].set_ylabel("eval stuckness")
    axes[1, 0].set_ylabel("eval goal rate")
    axes[0, 0].legend(fontsize=7, title="recovery time constant", title_fontsize=7)
    fig.suptitle("Freezing ablation (habituation brain only): does stuckness scale with habituation strength?",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def rise_then_fall(ck: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Per run: true return at the end is >= 0.3 below its checkpoint peak while the neural return
    (3-checkpoint rolling mean) at the end is >= its value at that peak. Returns one bool per run."""
    def one(g):
        g = g.sort_values("step")
        g = g[g.kl.notna()]
        if len(g) < 4:
            return False
        neural = g.neural_return.rolling(3, min_periods=1).mean().to_numpy()
        true = g.true_return.to_numpy()
        peak = int(true.argmax())
        return bool(true[-1] <= true[peak] - 0.3 and neural[-1] >= neural[peak])
    return ck.groupby(keys).apply(one)


def summarize(runs: pd.DataFrame, ck: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rtf = rise_then_fall(ck, keys + ["seed"]).rename("rise_fall").reset_index()
    r = runs.merge(rtf, on=keys + ["seed"], how="left")
    r["hazard_route"] = (r.n_hazard > HAZARD_ROUTE) & (r.reached_goal >= 0.5)   # reached the goal via hazards
    agg = r.groupby(keys).agg(
        n_seeds=("seed", "count"), true_return=("true_return", "mean"), true_return_sd=("true_return", "std"),
        neural_return=("neural_return", "mean"), goal_rate=("reached_goal", "mean"),
        k_goal=("reached_goal", lambda s: int((s >= 0.5).sum())),
        k_hazard_route=("hazard_route", "sum"),
        k_frozen=("stuckness", lambda s: int((s > FROZEN).sum())),
        k_rise_fall=("rise_fall", lambda s: int(s.fillna(False).sum())),
        stuckness=("stuckness", "mean"), stuckness_late=("stuckness_late", "mean"), goal_rate_late=("goal_rate_late", "mean"),
        frac_out_fov_late=("frac_out_fov_late", "mean"), n_errors_late=("n_errors_late", "mean"),
    ).reset_index()
    return agg


def main(results: Path = RESULTS) -> None:
    runs = pd.read_csv(results / "main_runs.csv")
    ck = pd.read_csv(results / "main_checkpoints.csv")
    keys = ["learner", "condition", "alpha", "centered"]
    summary = summarize(runs, ck, keys)
    summary.to_csv(results / "summary_main.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(2).to_string(index=False))

    for learner in runs.learner.unique():
        plot_overopt(ck, learner, results / f"overopt_{learner}.png")
    n = runs.groupby("learner").seed.nunique().to_dict()
    seeds = ", ".join(f"{LEARNER_NAMES[k]} {v} seeds" for k, v in n.items())
    plot_metric_vs_alpha(runs, "true_return", "final true return (greedy eval)",
                         f"Final true return vs neural reward weight (mean ± 95% CI; {seeds})",
                         results / "final_true_return.png", ylim=(-1.6, 1.0))
    plot_metric_vs_alpha(runs, "stuckness", "eval stuckness",
                         f"Stuckness = share of steps repeating one (cell, action) error, greedy eval (mean ± 95% CI; {seeds})",
                         results / "stuckness_alpha.png", ylim=(-0.05, 1.05))
    plot_metric_vs_alpha(runs, "frac_out_fov_late", "fraction of errors outside the FOV",
                         f"Hiding mistakes: errors outside the FOV in the last 20% of training, equal-length routes (mean ± 95% CI; {seeds})",
                         results / "hiding.png", ylim=(-0.05, 1.05))

    ab_path = results / "ablation_runs.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        ab_ck = pd.read_csv(results / "ablation_checkpoints.csv")
        ab_keys = ["learner", "alpha", "habituation_decay", "habituation_tau"]
        ab_summary = summarize(ab, ab_ck, ab_keys)
        ab_summary.to_csv(results / "summary_ablation.csv", index=False)
        print(ab_summary.round(2).to_string(index=False))
        plot_ablation(ab, results / "stuckness_ablation.png")
    print(f"wrote figures and summaries to {results}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS)
