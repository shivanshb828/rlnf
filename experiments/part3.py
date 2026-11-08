"""Part 3 (attention manipulation) analyses. Reads results/att_{distractor,occluder,fixes}_runs.csv.

    python experiments/part3.py

Writes figures/part3_distractor.png, figures/part3_fixes.png and results/summary_att_*.csv.
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
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})
SETTING_ORDER = ["gaze_k2", "gaze_k5", "gaze_k10", "control_inert_wave", "control_attentive_honest",
                 "occluder", "occluder_transparent", "occluder_attentive_honest"]


def ci95(v):
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def agg(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    r = runs.assign(goal=runs.reached_goal >= 0.5, shortcut=(runs.n_hazard > 0.5) & (runs.reached_goal >= 0.5),
                    uses_distractor=runs.waves > 0.5, uses_box=runs.pushes > 0.5, frozen=runs.stuckness > 0.3)
    return r.groupby(keys).agg(
        n_seeds=("seed", "count"), k_goal=("goal", "sum"), k_shortcut=("shortcut", "sum"), k_uses_distractor=("uses_distractor", "sum"),
        k_uses_box=("uses_box", "sum"), k_frozen=("frozen", "sum"), true_return=("true_return", "mean"), true_return_ci=("true_return", ci95),
        waves=("waves", "mean"), pushes=("pushes", "mean"), hazards=("n_hazard", "mean"),
        frac_hazards_after_wave=("frac_hazards_after_wave", "mean"), frac_hazards_unobserved=("frac_hazards_unobserved", "mean"),
        steps=("steps", "mean")).reset_index()


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    pd.set_option("display.width", 260)
    frames = []
    for name in ("att_distractor", "att_occluder", "att_fixes"):
        p = RESULTS / f"{name}_runs.csv"
        if p.exists():
            s = agg(pd.read_csv(p), ["setting", "observability", "variant"])
            s.to_csv(RESULTS / f"summary_{name}.csv", index=False)
            print(f"== {name}"); print(s.round(3).to_string(index=False))
            frames.append(s.assign(experiment=name))
    if not frames:
        return
    s = pd.concat(frames, ignore_index=True)
    # Figure 1: distractor use, shortcut rate and error timing vs controls (baseline reward)
    d = s[(s.observability == "baseline")]
    settings = [x for x in SETTING_ORDER if x in set(d.setting)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), squeeze=False)
    for ax, (metric, label) in zip(axes[0], [("k_shortcut", "seeds taking the shortcut (of 20)"), ("k_uses_distractor", "seeds using the distractor / box (of 20)"),
                                             ("frac_hazards_after_wave", "share of hazard entries within k steps after a wave")]):
        for v, col, off in (("plain", "#eb6834", -0.18), ("double", "#2a78d6", 0.18)):
            g = d[d.variant == v].set_index("setting").reindex(settings)
            y = g[metric] if metric != "k_uses_distractor" else g.k_uses_distractor.where(g.index.str.startswith("gaze") | g.index.str.startswith("control"), g.k_uses_box)
            ax.bar(np.arange(len(settings)) + off, y, 0.36, color=col, label=f"{v} Q", edgecolor=SURFACE)
        short = {"gaze_k2": "wave\nk=2", "gaze_k5": "wave\nk=5", "gaze_k10": "wave\nk=10", "control_inert_wave": "control:\ninert wave",
                 "control_attentive_honest": "control:\nattentive", "occluder": "box", "occluder_transparent": "box:\ntransparent",
                 "occluder_attentive_honest": "box:\nattentive"}
        ax.set_xticks(range(len(settings))); ax.set_xticklabels([short.get(x, x) for x in settings], fontsize=7)
        ax.set_title(label, loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Attention manipulation: gaze observer with a distractor action or a pushable occluder (α = 1, σ = 0.3, 20 seeds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "part3_distractor.png", dpi=150); plt.close(fig)
    # Figure 2: fixes
    f = s[s.setting.isin(["gaze_k10", "occluder"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, squeeze=False)
    modes = ["baseline", "masked", "imputed"]
    for ax, setting in zip(axes[0], ["gaze_k10", "occluder"]):
        for v, col, off in (("plain", "#eb6834", -0.18), ("double", "#2a78d6", 0.18)):
            g = f[(f.setting == setting) & (f.variant == v)].set_index("observability").reindex(modes)
            ax.bar(np.arange(len(modes)) + off, g.k_shortcut, 0.36, color=col, label=f"{v} Q", edgecolor=SURFACE)
        ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes)
        ax.set_title(f"{setting.replace('_', ' ')}: seeds taking the shortcut (of 20)", loc="left", fontsize=9)
        ax.set_xlabel("reward for unattended steps")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixes: observability-aware (masked) and decoupled / imputed rewards against attention manipulation", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "part3_fixes.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
