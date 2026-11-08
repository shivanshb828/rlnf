"""v3 analysis: is the visibility-flaw freeze the Boring Areas Trap? Reads results/{variance,fixes,explore2,absent}_*.csv.

    python experiments/trap.py

Writes
  results/variance_by_region.png     agent-reward mean and variance per region (visible / hidden) vs decoder noise
  results/fixes.png                  freeze rate per learner variant vs noise, visibility and honest brains
  results/exploration_regions.png    time share in the hidden half and region-level Q-values over training, per epsilon schedule
  results/summary_{variance,fixes,explore2,absent}.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FROZEN = 0.3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
REGION_COLOR = {"visible": "#2a78d6", "hidden": "#eb6834"}
VARIANT_ORDER = ["plain", "double", "maxmin2", "maxmin4", "maxmin8", "asrn"]
VARIANT_COLOR = dict(zip(VARIANT_ORDER, ["#898781", "#2a78d6", "#86b6ef", "#1c5cab", "#0d366b", "#eb6834"]))
VARIANT_MARKER = dict(zip(VARIANT_ORDER, ["o", "s", "^", "^", "^", "D"]))
SCHEDULE_COLOR = {"low": "#86b6ef", "default": "#2a78d6", "high": "#0d366b"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def ci95(v: pd.Series) -> float:
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def region_stats(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Pooled per-region reward mean and variance from the summed columns."""
    out = df.copy()
    for reg in ("visible", "hidden"):
        n, s1, s2 = (out[f"{prefix}n_{reg}"], out[f"{prefix}r_sum_{reg}"], out[f"{prefix}r_sq_{reg}"])
        out[f"mean_{reg}"] = s1 / n.replace(0, np.nan)
        out[f"var_{reg}"] = s2 / n.replace(0, np.nan) - out[f"mean_{reg}"] ** 2
    return out


# ---------------------------------------------------------------------------
# 1. variance check
# ---------------------------------------------------------------------------
def variance_check(acc: pd.DataFrame) -> pd.DataFrame:
    """Reward statistics per region over the FIRST 20% of training (before any policy has settled)."""
    cv = pd.read_csv(RESULTS / "variance_curves.csv")
    early = cv[cv.bin <= 0.2 * cv.bin.max()]
    sums = early.groupby(["condition", "noise_sigma", "seed"])[[c for c in cv.columns if c.startswith(("n_vis", "n_hid", "r_sum", "r_sq"))]].sum().reset_index()
    st = region_stats(sums)
    agg = st.groupby(["condition", "noise_sigma"]).agg(
        n_seeds=("seed", "count"), mean_visible=("mean_visible", "mean"), mean_hidden=("mean_hidden", "mean"),
        var_visible=("var_visible", "mean"), var_hidden=("var_hidden", "mean"),
        var_visible_ci=("var_visible", ci95), var_hidden_ci=("var_hidden", ci95)).reset_index()
    agg["var_gap"] = agg.var_visible - agg.var_hidden
    agg["var_ratio"] = agg.var_visible / agg.var_hidden
    runs = pd.read_csv(RESULTS / "variance_runs.csv")
    frz = runs.assign(frozen=runs.stuckness > FROZEN).groupby(["condition", "noise_sigma"]).frozen.sum().rename("k_frozen").reset_index()
    agg = agg.merge(frz, on=["condition", "noise_sigma"]).merge(acc[["noise_sigma", "balanced_accuracy"]], on="noise_sigma")
    agg.to_csv(RESULTS / "summary_variance.csv", index=False)
    return agg


def plot_variance(agg: pd.DataFrame, out: Path) -> None:
    conds = ["honest", "visibility"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), sharex=True, squeeze=False)
    for j, cond in enumerate(conds):
        d = agg[agg.condition == cond].sort_values("noise_sigma")
        for reg in ("visible", "hidden"):
            axes[0, j].errorbar(d.noise_sigma, d[f"var_{reg}"], yerr=d[f"var_{reg}_ci"], color=REGION_COLOR[reg], marker="o",
                                ms=4, lw=1.6, capsize=2, label=f"{reg} half", markeredgecolor=SURFACE)
            axes[1, j].plot(d.noise_sigma, d[f"mean_{reg}"], color=REGION_COLOR[reg], marker="o", ms=4, lw=1.6, label=f"{reg} half",
                            markeredgecolor=SURFACE)
        ax2 = axes[0, j].twinx()
        ax2.bar(d.noise_sigma, d.k_frozen, width=d.noise_sigma * 0.25, color="#e1e0d9", zorder=0, label="seeds frozen (of 10)")
        ax2.set_ylim(0, 40); ax2.set_yticks([0, 10]); ax2.tick_params(colors=MUTED); ax2.grid(False)
        ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color("#c3c2b7")
        if j == 1:
            ax2.set_ylabel("seeds frozen (of 10), bars", color=MUTED)
        axes[0, j].set_title(f"{cond} brain", loc="left", fontsize=9)
        axes[0, j].set_xscale("log"); axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel("decoder noise σ")
        axes[0, j].set_xticks(sorted(agg.noise_sigma.unique())); axes[0, j].set_xticklabels([f"{s:g}" for s in sorted(agg.noise_sigma.unique())])
    axes[0, 0].set_ylabel("per-step agent-reward variance (log)")
    axes[1, 0].set_ylabel("per-step agent-reward mean")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Reward statistics by region during the first 20% of training (Q-learning, α = 1, centered; 10 seeds)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. fixes
# ---------------------------------------------------------------------------
def fixes_summary() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "fixes_runs.csv")
    r = runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5,
                    hazard_route=(runs.n_hazard > 0.5) & (runs.reached_goal >= 0.5))
    agg = r.groupby(["condition", "variant", "noise_sigma"]).agg(
        n_seeds=("seed", "count"), k_frozen=("frozen", "sum"), k_goal=("goal", "sum"), k_hazard_route=("hazard_route", "sum"),
        true_return=("true_return", "mean"), true_return_ci=("true_return", ci95), stuckness_late=("stuckness_late", "mean"),
        goal_rate_late=("goal_rate_late", "mean"), hidden_share_late=("late_hidden_share", "mean")).reset_index()
    agg.to_csv(RESULTS / "summary_fixes.csv", index=False)
    return agg


def plot_fixes(agg: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], ["visibility", "honest"]):
        for v in VARIANT_ORDER:
            d = agg[(agg.condition == cond) & (agg.variant == v)].sort_values("noise_sigma")
            if d.empty:
                continue
            ax.plot(d.noise_sigma, d.k_frozen, color=VARIANT_COLOR[v], marker=VARIANT_MARKER[v], ms=5, lw=1.6,
                    label=v, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_title(f"{cond} brain", loc="left", fontsize=9)
        ax.set_xlabel("decoder noise σ")
        ax.set_xticks(sorted(agg.noise_sigma.unique()))
        ax.set_ylim(-0.5, 21)
    axes[0, 0].set_ylabel("seeds frozen (of 20)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Known fixes for the Boring Areas Trap: freeze rate per Q-learning variant (α = 1, centered, 20 seeds per point)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. exploration discrepancy
# ---------------------------------------------------------------------------
def explore_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = pd.read_csv(RESULTS / "explore2_curves.csv")
    cv = cv[cv.bin <= 100_000]        # the bin past the end holds only the final truncated episode
    runs = pd.read_csv(RESULTS / "explore2_runs.csv")
    r = runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5)
    agg = r.groupby(["exploration", "q_eps_end", "q_eps_fraction"]).agg(
        n_seeds=("seed", "count"), k_frozen=("frozen", "sum"), k_goal=("goal", "sum"), true_return=("true_return", "mean"),
        hidden_share_late=("late_hidden_share", "mean"), q_visible=("q_visible", "mean"), q_hidden=("q_hidden", "mean")).reset_index()
    agg.to_csv(RESULTS / "summary_explore2.csv", index=False)
    st = region_stats(cv)
    curves = st.groupby(["exploration", "bin"]).agg(
        hidden_share=("hidden_share", "mean"), hidden_share_ci=("hidden_share", ci95),
        q_visible=("q_visible", "mean"), q_visible_ci=("q_visible", ci95), q_hidden=("q_hidden", "mean"), q_hidden_ci=("q_hidden", ci95),
        mean_visible=("mean_visible", "mean"), mean_hidden=("mean_hidden", "mean"),
        var_visible=("var_visible", "mean"), var_hidden=("var_hidden", "mean"), n_errors=("n_errors", "mean"),
        reached_goal=("reached_goal", "mean")).reset_index()
    return agg, curves


def plot_explore(curves: pd.DataFrame, out: Path) -> None:
    order = [s for s in ("low", "default", "high") if s in set(curves.exploration)]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), squeeze=False)
    for sched in order:
        d = curves[curves.exploration == sched].sort_values("bin")
        c = SCHEDULE_COLOR[sched]
        axes[0, 0].plot(d.bin, d.hidden_share, color=c, lw=1.6, label=sched)
        axes[0, 0].fill_between(d.bin, d.hidden_share - d.hidden_share_ci, d.hidden_share + d.hidden_share_ci, color=c, alpha=0.15, lw=0)
        axes[0, 1].plot(d.bin, d.q_visible, color=c, lw=1.6, label=f"{sched}: visible half")
        axes[0, 1].plot(d.bin, d.q_hidden, color=c, lw=1.6, ls="--", label=f"{sched}: hidden half")
        axes[0, 2].plot(d.bin, d.mean_visible, color=c, lw=1.6, label=f"{sched}: visible half")
        axes[0, 2].plot(d.bin, d.mean_hidden, color=c, lw=1.6, ls="--", label=f"{sched}: hidden half")
    axes[0, 0].set_title("share of training steps spent in the hidden half", loc="left", fontsize=9)
    axes[0, 1].set_title("mean state value max_a Q(s, a) over each region's cells", loc="left", fontsize=9)
    axes[0, 2].set_title("mean per-step agent reward received in each region", loc="left", fontsize=9)
    for ax in axes[0]:
        ax.set_xlabel("training steps")
    axes[0, 0].legend(fontsize=8, title="ε schedule", title_fontsize=8)
    axes[0, 1].legend(fontsize=7, ncol=1)
    fig.suptitle("Exploration discrepancy: visibility flaw, α = 1, σ = 0.3, centered (Q-learning, 20 seeds, mean ± 95% CI)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. absent supervisor
# ---------------------------------------------------------------------------
def absent_summary() -> pd.DataFrame | None:
    p = RESULTS / "absent_runs.csv"
    if not p.exists():
        return None
    runs = pd.read_csv(p)
    r = runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5,
                    shortcut=(runs.n_hazard > 0.25) & (runs.reached_goal >= 0.5))
    agg = r.groupby(["condition", "variant", "noise_sigma"]).agg(
        n_seeds=("seed", "count"), k_goal=("goal", "sum"), k_shortcut=("shortcut", "sum"), k_frozen=("frozen", "sum"),
        true_return=("true_return", "mean"), true_return_ci=("true_return", ci95), hazard_entries=("n_hazard", "mean"),
        steps=("steps", "mean")).reset_index()
    agg.to_csv(RESULTS / "summary_absent.csv", index=False)
    return agg


def main() -> None:
    pd.set_option("display.width", 250)
    acc = pd.read_csv(RESULTS / "decoder_accuracy.csv")
    v = variance_check(acc)
    print("== variance check (first 20% of training)")
    print(v.round(4).to_string(index=False))
    plot_variance(v, RESULTS / "variance_by_region.png")
    f = fixes_summary()
    print("== fixes")
    print(f.round(3).to_string(index=False))
    plot_fixes(f, RESULTS / "fixes.png")
    e, curves = explore_summary()
    print("== exploration")
    print(e.round(3).to_string(index=False))
    plot_explore(curves, RESULTS / "exploration_regions.png")
    a = absent_summary()
    if a is not None:
        print("== absent supervisor")
        print(a.round(3).to_string(index=False))
    print(f"wrote to {RESULTS}/")


if __name__ == "__main__":
    main()
