"""Part 2 (praise hacking) analyses. Reads results/praise_{a,b,c,flip}_*.csv.

    python experiments/part2.py

Writes figures/part2_praise_vs_freeze.png, figures/part2_surprise_pressure.png and results/summary_praise_*.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
FROZEN = 0.3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
W_COLOR = {0.0: "#86b6ef", 0.5: "#2a78d6", 1.0: "#0d366b"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def ci95(v):
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def agg(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    r = runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5,
                    hazard_route=(runs.n_hazard > 0.5) & (runs.reached_goal >= 0.5))
    cols = dict(n_seeds=("seed", "count"), k_frozen=("frozen", "sum"), k_goal=("goal", "sum"), k_hazard_route=("hazard_route", "sum"),
                true_return=("true_return", "mean"), true_return_ci=("true_return", ci95), neural_return=("neural_return", "mean"),
                stuckness_late=("stuckness_late", "mean"), goal_rate_late=("goal_rate_late", "mean"))
    for k in ("mean_surprise", "frac_unpredictable", "action_entropy", "praise_return", "extra_steps"):
        if k in runs.columns:
            cols[k] = (k, "mean")
    return r.groupby(keys).agg(**cols).reset_index()


def rise_then_fall(ck: pd.DataFrame, keys: list[str]) -> pd.Series:
    def one(g):
        g = g.sort_values("step"); g = g[g.kl.notna()]
        if len(g) < 4:
            return False
        true = g.true_return.to_numpy(); neural = g.neural_return.rolling(3, min_periods=1).mean().to_numpy()
        pk = int(true.argmax())
        return bool(true[-1] <= true[pk] - 0.3 and neural[-1] >= neural[pk])
    return ck.groupby(keys).apply(one)


def praise_a() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "praise_a_runs.csv")
    s = agg(runs, ["learner", "variant", "noise_sigma", "praise_ratio"])
    s.to_csv(RESULTS / "summary_praise_a.csv", index=False)
    d = s[s.learner == "q"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True, squeeze=False)
    for ax, sig in zip(axes[0], sorted(d.noise_sigma.unique())):
        for v, col in (("plain", "#eb6834"), ("double", "#2a78d6")):
            g = d[(d.noise_sigma == sig) & (d.variant == v)].sort_values("praise_ratio")
            ax.plot(g.praise_ratio, g.k_frozen, color=col, marker="o", ms=5, lw=1.6, label=f"{v} Q", markeredgecolor=SURFACE)
        ax.set_title(f"visibility brain, σ = {sig:g}", loc="left", fontsize=9)
        ax.set_xlabel("praise ratio (weight of the praise term)"); ax.set_xticks([0, 0.25, 0.5, 1.0]); ax.set_ylim(-0.5, 21)
    axes[0, 0].set_ylabel("seeds frozen (of 20)"); axes[0, 0].legend(fontsize=8)
    fig.suptitle("Does praise fix freezing? (α = 1, centered, surprise weight 0, 20 seeds per point)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "part2_praise_vs_freeze.png", dpi=150); plt.close(fig)
    return s


def praise_b() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "praise_b_runs.csv")
    ck = pd.read_csv(RESULTS / "praise_b_checkpoints.csv")
    keys = ["variant", "surprise_weight", "alpha"]
    s = agg(runs, keys)
    rtf = rise_then_fall(ck, keys + ["seed"]).rename("rise_fall").reset_index()
    s = s.merge(rtf.groupby(keys).rise_fall.sum().rename("k_rise_fall").reset_index(), on=keys)
    s.to_csv(RESULTS / "summary_praise_b.csv", index=False)
    # figure: true return vs alpha, vs training steps (checkpoints), vs KL; one line per surprise weight; plain Q
    d = ck[(ck.variant == "plain") & ck.kl.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), squeeze=False)
    for w, col in W_COLOR.items():
        g = s[(s.variant == "plain") & (s.surprise_weight == w)].sort_values("alpha")
        axes[0, 0].errorbar(g.alpha, g.true_return, yerr=g.true_return_ci, color=col, marker="o", ms=5, lw=1.6, capsize=2, label=f"w = {w:g}", markeredgecolor=SURFACE)
        for alpha, ls in ((1.0, "-"), (4.0, "--")):
            t = d[(d.surprise_weight == w) & (d.alpha == alpha)].groupby("step")[["true_return", "kl", "neural_return"]].mean().sort_index()
            axes[0, 1].plot(t.index, t.true_return, color=col, ls=ls, lw=1.5, label=f"w = {w:g}, α = {alpha:g}")
            axes[0, 2].plot(t.kl, t.true_return, color=col, ls=ls, lw=1.5, marker="o", ms=2.5, label=f"w = {w:g}, α = {alpha:g}")
    axes[0, 0].set_xscale("symlog", linthresh=0.25); axes[0, 0].set_xticks([0.25, 1, 4]); axes[0, 0].set_xticklabels(["0.25", "1", "4"])
    axes[0, 0].set_xlabel("α"); axes[0, 0].set_ylabel("final true return (greedy eval)"); axes[0, 0].set_title("vs α", loc="left", fontsize=9)
    axes[0, 1].set_xlabel("training steps"); axes[0, 1].set_title("vs training steps (checkpoint greedy eval)", loc="left", fontsize=9)
    axes[0, 2].set_xlabel("KL from the 10% policy"); axes[0, 2].set_title("vs divergence from the early policy", loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=7, title="surprise weight", title_fontsize=7); axes[0, 2].legend(fontsize=6.5)
    fig.suptitle("Surprise confound under the honest brain: true return vs optimisation pressure (plain Q, praise ratio 0.5, 20 seeds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "part2_surprise_pressure.png", dpi=150); plt.close(fig)
    return s


def praise_c() -> pd.DataFrame:
    s = agg(pd.read_csv(RESULTS / "praise_c_runs.csv"), ["variant", "surprise_weight", "alpha"])
    s.to_csv(RESULTS / "summary_praise_c.csv", index=False)
    return s


def praise_flip() -> pd.DataFrame:
    s = agg(pd.read_csv(RESULTS / "praise_flip_runs.csv"), ["variant"])
    s.to_csv(RESULTS / "summary_praise_flip.csv", index=False)
    return s


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    for name, fn in (("praise_a", praise_a), ("praise_b", praise_b), ("praise_c", praise_c), ("praise_flip", praise_flip)):
        if (RESULTS / f"{name}_runs.csv").exists():
            print(f"== {name}")
            print(fn().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
