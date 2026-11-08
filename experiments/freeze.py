"""v2.1 analysis: is visibility-flaw freezing a trap or an optimum? Reads results/{noise,explore,ppo_norm}_runs.csv.

    python experiments/freeze.py

Writes
  results/decoder_accuracy.csv     decoder accuracy / balanced accuracy / F1 vs noise sigma (held-out honest data)
  results/noise_dose_response.png  headline: freeze rate (k of 20) vs decoder balanced accuracy, per alpha, per condition
  results/learner_sensitivity.png  freeze rate and goal rate over learning rate x exploration schedule
  results/summary_noise.csv, summary_explore.csv, summary_ppo_norm.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlnf.brain import collect_honest_samples, fit_decoder  # noqa: E402
from rlnf.config import BrainConfig, EnvConfig, TrainConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FROZEN = 0.3
COND_COLOR = {"honest": "#2a78d6", "visibility": "#eb6834"}
COND_MARKER = {"honest": "o", "visibility": "s"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LIT_BAND = (65.0, 80.0)   # approximate single-trial ErrP decoder accuracy range reported in the literature

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def decoder_accuracy(sigmas, seeds=range(5), n_test: int = 10_000) -> pd.DataFrame:
    """Fit the decoder as the runs do (seed s), score it on held-out honest data (seed 100 + s)."""
    rows = []
    for sigma in sigmas:
        for s in seeds:
            cfg = BrainConfig(noise_sigma=sigma)
            dec = fit_decoder(EnvConfig(), cfg, n_steps=TrainConfig.decoder_samples, seed=s)
            amps, y = collect_honest_samples(EnvConfig(), cfg, n_test, seed=100 + s)
            pred = dec.predict_proba(amps) > 0.5
            tp, tn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
            fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
            rows.append(dict(noise_sigma=sigma, seed=s, accuracy=(tp + tn) / len(y),
                             balanced_accuracy=0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                             f1=2 * tp / (2 * tp + fp + fn), p_clean=dec.p_clean))
    df = pd.DataFrame(rows).groupby("noise_sigma").mean().drop(columns="seed").reset_index()
    df.to_csv(RESULTS / "decoder_accuracy.csv", index=False)
    return df


def ci95(v: pd.Series) -> float:
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def summarize(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    r = runs.assign(frozen=runs.stuckness > FROZEN, goal=runs.reached_goal >= 0.5,
                    hazard_route=(runs.n_hazard > 0.5) & (runs.reached_goal >= 0.5))
    return r.groupby(keys).agg(
        n_seeds=("seed", "count"), k_frozen=("frozen", "sum"), k_goal=("goal", "sum"), k_hazard_route=("hazard_route", "sum"),
        true_return=("true_return", "mean"), neural_return=("neural_return", "mean"), stuckness=("stuckness", "mean"),
        stuckness_late=("stuckness_late", "mean"), goal_rate_late=("goal_rate_late", "mean"),
        frac_out_fov_late=("frac_out_fov_late", "mean"),
    ).reset_index()


def plot_noise(summary: pd.DataFrame, out: Path) -> None:
    d = summary.copy()
    d["acc_pct"] = 100 * d.balanced_accuracy
    alphas = sorted(d.alpha.unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.4 * len(alphas), 3.6), sharey=True, squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        ax.axvspan(*LIT_BAND, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
        ax.text(np.mean(LIT_BAND), 10.0, "approx. range of real\nsingle-trial ErrP decoders", ha="center", va="center",
                fontsize=7, color="#52514e")
        for cond in ("honest", "visibility"):
            g = d[(d.alpha == alpha) & (d.condition == cond)].sort_values("acc_pct")
            ax.plot(g.acc_pct, g.k_frozen, color=COND_COLOR[cond], marker=COND_MARKER[cond], ms=5, lw=1.8,
                    label=cond, markeredgecolor=SURFACE, markeredgewidth=0.8)
            if cond == "visibility":
                for _, r in g.iterrows():
                    ax.annotate(f"σ={r.noise_sigma:g}", (r.acc_pct, r.k_frozen), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=6.5, color=MUTED)
        n = int(d.n_seeds.max())
        ax.set_title(f"α = {alpha:g}", loc="left", fontsize=9)
        ax.set_xlabel("decoder balanced accuracy on held-out honest data (%)")
        ax.set_ylim(-0.5, n + 1)
        ax.set_xlim(48, 102)
    axes[0, 0].set_ylabel(f"seeds frozen (of {n})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Freeze rate vs decoder accuracy (Q-learning, centered reward; 20 seeds per point; noise σ labelled)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_explore(summary: pd.DataFrame, out: Path) -> None:
    order = ["low", "default", "high"]
    lrs = sorted(summary.q_lr.unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), squeeze=False)
    for ax, (metric, title) in zip(axes[0], [("k_frozen", "seeds frozen (of 20)"), ("k_goal", "seeds reaching the goal (of 20)")]):
        m = summary.pivot(index="q_lr", columns="exploration", values=metric).reindex(index=lrs, columns=order)
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=20, aspect="auto")
        for i, lr in enumerate(lrs):
            for j, ex in enumerate(order):
                v = m.loc[lr, ex]
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10, color="#ffffff" if v > 10 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{e}\nε→{summary[summary.exploration == e].q_eps_end.iloc[0]:g} over {int(100 * summary[summary.exploration == e].q_eps_fraction.iloc[0])}%" for e in order], fontsize=7.5)
        ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"lr = {lr:g}" for lr in lrs])
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False)
        ax.set_xlabel("exploration schedule")
    fig.suptitle("Learner sensitivity: visibility flaw, α = 1, σ = 0.3, centered (Q-learning, 20 seeds per cell)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 250)
    noise = pd.read_csv(RESULTS / "noise_runs.csv")
    acc = decoder_accuracy(sorted(noise.noise_sigma.unique()))
    print(acc.round(3).to_string(index=False))
    s_noise = summarize(noise, ["condition", "alpha", "noise_sigma"]).merge(acc, on="noise_sigma")
    s_noise.to_csv(RESULTS / "summary_noise.csv", index=False)
    print(s_noise[["condition", "alpha", "noise_sigma", "balanced_accuracy", "k_frozen", "k_goal", "k_hazard_route", "true_return", "stuckness_late"]].round(3).to_string(index=False))
    plot_noise(s_noise, RESULTS / "noise_dose_response.png")

    explore = pd.read_csv(RESULTS / "explore_runs.csv")
    s_explore = summarize(explore, ["q_lr", "exploration", "q_eps_end", "q_eps_fraction"])
    s_explore.to_csv(RESULTS / "summary_explore.csv", index=False)
    print(s_explore[["q_lr", "exploration", "k_frozen", "k_goal", "true_return", "stuckness_late", "goal_rate_late"]].round(3).to_string(index=False))
    plot_explore(s_explore, RESULTS / "learner_sensitivity.png")

    p = RESULTS / "ppo_norm_runs.csv"
    if p.exists():
        ppo = pd.read_csv(p)
        s_ppo = summarize(ppo, ["condition", "alpha"])
        s_ppo.to_csv(RESULTS / "summary_ppo_norm.csv", index=False)
        print(s_ppo[["condition", "alpha", "n_seeds", "k_frozen", "k_goal", "true_return", "neural_return", "stuckness_late", "goal_rate_late"]].round(3).to_string(index=False))
    print(f"wrote to {RESULTS}/")


if __name__ == "__main__":
    main()
