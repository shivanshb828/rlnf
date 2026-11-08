"""Part 1 finishing analyses: PPO critic diagnosis, DQN vs Double DQN, observability-aware reward.

    python experiments/part1.py

Reads results/{ppo_diag,dqn,observ}_*.csv. Writes figures/part1_*.png and results/summary_{ppo_diag,dqn,observ}.csv.
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
REGION_COLOR = {"hidden": "#eb6834", "visible": "#2a78d6", "safe_route": "#1baf7a"}
MODE_COLOR = dict(zip(["baseline", "masked", "imputed", "pessimistic"], ["#898781", "#2a78d6", "#eb6834", "#1baf7a"]))
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def ci95(v: pd.Series) -> float:
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def flags(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5,
                       shortcut=(runs.n_hazard > 0.25) & (runs.reached_goal >= 0.5))


def agg(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return flags(runs).groupby(keys).agg(
        n_seeds=("seed", "count"), k_frozen=("frozen", "sum"), k_goal=("goal", "sum"), k_shortcut=("shortcut", "sum"),
        true_return=("true_return", "mean"), true_return_ci=("true_return", ci95),
        stuckness_late=("stuckness_late", "mean"), goal_rate_late=("goal_rate_late", "mean")).reset_index()


# ---------------------------------------------------------------------------
# 1a. PPO critic diagnosis
# ---------------------------------------------------------------------------
def ppo_diag() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "ppo_diag_runs.csv")
    ck = pd.read_csv(RESULTS / "ppo_diag_checkpoints.csv")
    summary = agg(runs, ["condition", "variant", "noise_sigma"])
    # critic bias = V - MC at the last 5 checkpoints, per region
    c = ck[ck.phase.fillna("checkpoint") != "update"] if "phase" in ck else ck
    c = c[c.v_hidden.notna()]
    late = c[c.step > 0.75 * c.step.max()]
    bias = late.assign(**{f"bias_{r}": late[f"v_{r}"] - late[f"mc_{r}"] for r in REGION_COLOR})
    b = bias.groupby(["condition", "variant", "noise_sigma"])[[f"bias_{r}" for r in REGION_COLOR] + [f"mc_{r}" for r in REGION_COLOR]].mean().reset_index()
    summary = summary.merge(b, on=["condition", "variant", "noise_sigma"], how="left")
    up = ck[ck.phase == "update"] if "phase" in ck else ck.iloc[0:0]
    if not up.empty:
        early = up[up.step <= 0.2 * up.step.max()]
        a = early.groupby(["condition", "variant", "noise_sigma"])[["adv_mean_hidden", "adv_std_hidden", "adv_mean_visible", "adv_std_visible"]].mean().reset_index()
        summary = summary.merge(a, on=["condition", "variant", "noise_sigma"], how="left")
    summary.to_csv(RESULTS / "summary_ppo_diag.csv", index=False)
    plot_ppo_diag(c, FIGURES / "part1_ppo_critic.png")
    return summary


def plot_ppo_diag(c: pd.DataFrame, out: Path) -> None:
    d = c[c.variant == "default"]
    sigmas = sorted(d.noise_sigma.unique()); conds = ["honest", "visibility"]
    fig, axes = plt.subplots(len(sigmas), len(conds), figsize=(4.6 * len(conds), 3.2 * len(sigmas)), sharex=True, squeeze=False)
    for i, sig in enumerate(sigmas):
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            g = d[(d.noise_sigma == sig) & (d.condition == cond)].groupby("step")
            for r, col in REGION_COLOR.items():
                m = g[[f"v_{r}", f"mc_{r}"]].mean()
                ax.plot(m.index, m[f"v_{r}"], color=col, lw=1.6, label=f"critic V, {r.replace('_', ' ')}")
                ax.plot(m.index, m[f"mc_{r}"], color=col, lw=1.2, ls="--", label=f"Monte-Carlo return, {r.replace('_', ' ')}")
            ax.set_title(f"{cond} brain, σ = {sig:g}", loc="left", fontsize=9)
            if i == len(sigmas) - 1:
                ax.set_xlabel("training steps")
    axes[0, 0].set_ylabel("discounted agent return from region states")
    axes[1, 0].set_ylabel("discounted agent return from region states")
    axes[0, 0].legend(fontsize=6.5, ncol=1)
    fig.suptitle("PPO critic vs Monte-Carlo return of the current policy, by region (default PPO, α = 1, 10 seeds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. DQN vs Double DQN
# ---------------------------------------------------------------------------
def dqn() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "dqn_runs.csv")
    summary = agg(runs, ["condition", "variant", "noise_sigma"])
    summary.to_csv(RESULTS / "summary_dqn.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# 1c. observability-aware reward
# ---------------------------------------------------------------------------
def observ() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "observ_runs.csv")
    summary = agg(runs, ["env_name", "condition", "noise_sigma", "variant", "observability"])
    summary.to_csv(RESULTS / "summary_observ.csv", index=False)
    plot_observ(summary, FIGURES / "part1_observability_fix.png")
    return summary


def plot_observ(s: pd.DataFrame, out: Path) -> None:
    d = s[(s.condition == "visibility") & (s.noise_sigma == 0.3)]
    panels = [("maze", "k_frozen", "seeds frozen (of 20)"), ("absent_supervisor", "k_shortcut", "seeds taking the shortcut (of 20)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), squeeze=False)
    width = 0.2
    for ax, (env, metric, ylabel) in zip(axes[0], panels):
        de = d[d.env_name == env]
        for k, mode in enumerate(MODE_COLOR):
            vals = [de[(de.variant == v) & (de.observability == mode)][metric].sum() for v in ("plain", "double")]
            ax.bar(np.arange(2) + (k - 1.5) * width, vals, width, color=MODE_COLOR[mode], label=mode, edgecolor=SURFACE)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["plain Q", "Double Q"])
        ax.set_title(f"{env.replace('_', ' ')}: visibility brain, σ = 0.3", loc="left", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_ylim(0, 21)
    axes[0, 0].legend(fontsize=8, title="unobserved steps get", title_fontsize=8)
    fig.suptitle("Observability-aware reward (system knows which steps were seen): 20 seeds per bar", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1d. real ErrP noise
# ---------------------------------------------------------------------------
def realnoise() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "realnoise_runs.csv")
    acc = pd.read_csv(RESULTS / "real_decoder_accuracy.csv")
    runs = runs.merge(acc[["subject", "balanced_accuracy", "auc"]], on="subject")
    by_subject = agg(runs, ["condition", "variant", "subject", "balanced_accuracy"])
    overall = agg(runs, ["condition", "variant"])
    by_subject.to_csv(RESULTS / "summary_realnoise.csv", index=False)
    plot_realnoise(by_subject, FIGURES / "part1_real_noise_cliff.png")
    return pd.concat([overall.assign(subject="all"), by_subject], ignore_index=True)


def plot_realnoise(s: pd.DataFrame, out: Path) -> None:
    """Freeze rate vs real per-subject decoder accuracy, next to the Gaussian-noise cliff from v2.1."""
    gauss = pd.read_csv(RESULTS / "summary_noise.csv")
    gauss = gauss[gauss.alpha == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=False, squeeze=False)
    ax = axes[0, 0]
    ax.axvspan(65, 91, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
    for cond, col in (("honest", "#2a78d6"), ("visibility", "#eb6834")):
        g = gauss[gauss.condition == cond].sort_values("balanced_accuracy")
        ax.plot(100 * g.balanced_accuracy, g.k_frozen / 20, color=col, marker="o", ms=4, lw=1.6, label=f"{cond}, Gaussian noise (plain Q)")
    ax.set_title("Gaussian decoder noise (v2.1 sweep, 20 seeds per point)", loc="left", fontsize=9)
    ax.set_xlabel("decoder balanced accuracy (%)"); ax.set_ylabel("fraction of seeds frozen"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.axvspan(65, 91, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
    ax.text(78, 0.8, "approx. real-decoder band\n65-91% (Iturrate 2010, Kim 2017)", ha="center", fontsize=7, color="#52514e")
    for (cond, var), (col, mk) in {("honest", "plain"): ("#2a78d6", "o"), ("visibility", "plain"): ("#eb6834", "s"),
                                   ("honest", "double"): ("#86b6ef", "o"), ("visibility", "double"): ("#f2a37f", "s")}.items():
        g = s[(s.condition == cond) & (s.variant == var)].sort_values("balanced_accuracy")
        ax.plot(100 * g.balanced_accuracy, g.k_frozen / g.n_seeds, color=col, marker=mk, ms=5, lw=1.4, label=f"{cond}, {var} Q",
                markeredgecolor=SURFACE, ls="-" if var == "plain" else "--")
        for _, r in g.iterrows():
            ax.annotate(f"S{int(r.subject)}", (100 * r.balanced_accuracy, r.k_frozen / r.n_seeds), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=MUTED)
    ax.set_title("Real ErrP decoder outputs, BNCI 013-2015 (one subject per seed, 3-4 seeds each)", loc="left", fontsize=9)
    ax.set_xlabel("held-out balanced accuracy of that subject's decoder (%)"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)
    fig.suptitle("Does the freeze cliff hold with real decoder noise? (α = 1, centered)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learners(out: Path) -> None:
    """Freeze rate (visibility brain) per learner: tabular plain/double (v3 fixes), DQN/DDQN, PPO variants."""
    rows = []
    fx = pd.read_csv(RESULTS / "summary_fixes.csv")
    for v in ("plain", "double"):
        for sig in (0.3,):
            g = fx[(fx.condition == "visibility") & (fx.variant == v) & (fx.noise_sigma == sig)]
            rows.append(dict(learner=f"tabular {v} Q", sigma=sig, frac=g.k_frozen.sum() / g.n_seeds.sum(), n=int(g.n_seeds.sum())))
    for name, fn in (("dqn", "summary_dqn.csv"), ("ppo", "summary_ppo_diag.csv")):
        p = RESULTS / fn
        if p.exists():
            d = pd.read_csv(p)
            d = d[d.condition == "visibility"]
            for v, g in d.groupby("variant"):
                for sig, gg in g.groupby("noise_sigma"):
                    label = {"dqn": "DQN", "ddqn": "Double DQN", "default": "PPO (ours)", "gae0.8": "PPO, GAE λ 0.8", "critics3": "PPO, 3 critics"}.get(v, v)
                    rows.append(dict(learner=label, sigma=sig, frac=gg.k_frozen.sum() / gg.n_seeds.sum(), n=int(gg.n_seeds.sum())))
    d = pd.DataFrame(rows)
    order = ["tabular plain Q", "tabular double Q", "DQN", "Double DQN", "PPO (ours)", "PPO, GAE λ 0.8", "PPO, 3 critics"]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for sig, col, off in ((0.03, "#86b6ef", -0.18), (0.3, "#2a78d6", 0.18)):
        g = d[d.sigma == sig].set_index("learner").reindex(order)
        ax.bar(np.arange(len(order)) + off, g.frac, 0.36, color=col, label=f"σ = {sig:g}", edgecolor=SURFACE)
        for i, (f, n) in enumerate(zip(g.frac, g.n)):
            if not np.isnan(f):
                ax.text(i + off, f + 0.02, f"{int(round(f * n))}/{int(n)}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7.5, rotation=15)
    ax.set_ylabel("fraction of seeds frozen"); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, title="decoder noise", title_fontsize=8)
    fig.suptitle("Visibility-flaw freeze rate by learner (α = 1, centered; tabular at σ = 0.3 only)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    for name, fn in (("ppo_diag", ppo_diag), ("dqn", dqn), ("observ", observ), ("realnoise", realnoise)):
        if (RESULTS / f"{name}_runs.csv").exists():
            print(f"== {name}")
            print(fn().round(3).to_string(index=False))
    plot_learners(FIGURES / "part1_learners.png")
    import shutil
    shutil.copy(RESULTS / "fixes.png", FIGURES / "part1_fixes_comparison.png")


if __name__ == "__main__":
    main()
