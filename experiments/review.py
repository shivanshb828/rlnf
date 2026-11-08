"""Review-round analyses (Phases A-D). Reads results/{layouts,fixes2,qinit,double300k,realnoise20,praise_fixes,deep_fixes,shaping,distractor_robust}_runs.csv.

    python experiments/review.py

Writes results/summary_<experiment>.csv, results/summary_layouts_classified.csv, figures/review_*.png.
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
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def ci95(v):
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def flags(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs.copy()
    r["goal"] = r.reached_goal >= 0.5
    r["frozen"] = r.stuckness > FROZEN
    r["failed"] = ~r.goal
    r["hazard_route"] = (r.n_hazard > 0.5) & r.goal
    r["ends_hidden"] = r.hidden_share > 0.5
    if "baseline" in r.columns:
        r["cond"] = np.where(r.baseline.fillna("") == "alpha0", "alpha0", r.condition)
    else:
        r["cond"] = r.condition
    return r


def agg(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    r = flags(runs)
    cols = dict(n_seeds=("seed", "count"), k_goal=("goal", "sum"), k_failed=("failed", "sum"), k_frozen=("frozen", "sum"),
                k_hazard_route=("hazard_route", "sum"), k_ends_hidden=("ends_hidden", "sum"),
                true_return=("true_return", "mean"), true_return_ci=("true_return", ci95), hidden_share=("hidden_share", "mean"),
                goal_rate_late=("goal_rate_late", "mean"), stuckness_late=("stuckness_late", "mean"))
    for k in ("imputed_from_table_frac", "waves", "frac_hazards_after_wave", "praise_return", "good_events", "n_hazard"):
        if k in r.columns:
            cols[k] = (k, "mean")
    return r.groupby(keys).agg(**cols).reset_index()


# ---------------------------------------------------------------------------
# Phase A: two failure types
# ---------------------------------------------------------------------------
def classify_layouts() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "layouts_runs.csv")
    keys = ["layout", "variant", "q_init", "noise_sigma"]
    s = agg(runs, keys + ["cond"])
    s.to_csv(RESULTS / "summary_layouts.csv", index=False)
    wide = s.pivot_table(index=keys, columns="cond", values=["k_goal", "k_frozen", "k_ends_hidden", "true_return"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    # Classification of the visibility failures in each cell. The safe route is observed in off_route, unobserved in
    # on_route, half-observed in half_route.
    route_hidden = {"off_route": "no", "on_route": "yes", "half_route": "half"}
    rows = []
    for _, w in wide.iterrows():
        n = 20
        vis_fail = n - w.k_goal_visibility
        a0_ok, hon_ok = w.k_goal_alpha0 >= n / 2, w.k_goal_honest >= n / 2
        if vis_fail == 0:
            label = "no failure"
        elif a0_ok:
            label = "trap (solvable without guidance)"
        elif route_hidden[w.layout] == "no":
            label = "trap (guidance available on the route)"
        elif route_hidden[w.layout] == "yes":
            label = "lost guidance (route unobserved)"
        else:
            label = "mixed (half the route unobserved)"
        rows.append(dict(layout=w.layout, variant=w.variant, q_init=w.q_init, noise_sigma=w.noise_sigma,
                         goal_honest=int(w.k_goal_honest), goal_alpha0=int(w.k_goal_alpha0), goal_visibility=int(w.k_goal_visibility),
                         visibility_failed=int(vis_fail), visibility_frozen=int(w.k_frozen_visibility), visibility_ends_hidden=int(w.k_ends_hidden_visibility),
                         classification=label, true_visibility=round(w.true_return_visibility, 2), true_honest=round(w.true_return_honest, 2)))
    cls = pd.DataFrame(rows).sort_values(["layout", "variant", "q_init", "noise_sigma"])
    cls.to_csv(RESULTS / "summary_layouts_classified.csv", index=False)
    plot_layouts(cls, FIGURES / "review_two_failure_types.png")
    return cls


def plot_layouts(cls: pd.DataFrame, out: Path) -> None:
    layouts = ["off_route", "on_route", "half_route"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True, squeeze=False)
    for ax, lay in zip(axes[0], layouts):
        d = cls[(cls.layout == lay) & (cls.noise_sigma == 0.3)]
        labels, x = [], 0
        for v in ("plain", "double"):
            for qi in (-1.0, 0.0):
                r = d[(d.variant == v) & (d.q_init == qi)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.bar(x - 0.27, r.goal_honest, 0.25, color="#2a78d6", edgecolor=SURFACE, label="honest" if x == 0 else None)
                ax.bar(x, r.goal_alpha0, 0.25, color="#898781", edgecolor=SURFACE, label="α = 0" if x == 0 else None)
                ax.bar(x + 0.27, r.goal_visibility, 0.25, color="#eb6834", edgecolor=SURFACE, label="visibility" if x == 0 else None)
                ax.text(x + 0.27, r.goal_visibility + 0.4, f"{r.visibility_ends_hidden} in\nhidden", ha="center", fontsize=6, color=MUTED)
                labels.append(f"{v} Q\nq_init {qi:g}"); x += 1
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(f"{lay.replace('_', ' ')} blind spot", loc="left", fontsize=9)
        ax.set_ylim(0, 23)
    axes[0, 0].set_ylabel("seeds reaching the goal (of 20)"); axes[0, 0].legend(fontsize=8)
    fig.suptitle("Two failure types: goal rate by brain, per layout (σ = 0.3, α = 1; text = failed visibility seeds that end in the hidden region)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Phase B: fixes by mechanism
# ---------------------------------------------------------------------------
def fixes2() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "fixes2_runs.csv")
    s = agg(runs, ["layout", "condition", "variant", "observability"])
    s.to_csv(RESULTS / "summary_fixes2.csv", index=False)
    plot_fixes2(s, FIGURES / "review_fixes_by_mechanism.png")
    return s


def plot_fixes2(s: pd.DataFrame, out: Path) -> None:
    modes = ["baseline", "denoise", "tax", "pessimistic", "imputed"]
    layouts = ["off_route", "on_route", "half_route", "absent_supervisor"]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.2), sharey="row", squeeze=False)
    for j, lay in enumerate(layouts):
        d = s[(s.layout == lay) & (s.condition == "visibility")]
        metric = "k_hazard_route" if lay == "absent_supervisor" else "k_failed"
        for i, (met, ylabel) in enumerate([(metric, "seeds " + ("taking the shortcut" if lay == "absent_supervisor" else "failing to reach the goal")), ("k_hazard_route", "seeds reaching the goal via hazards")]):
            ax = axes[i, j]
            for v, col, off in (("plain", "#eb6834", -0.18), ("double", "#2a78d6", 0.18)):
                g = d[d.variant == v].set_index("observability").reindex(modes)
                ax.bar(np.arange(len(modes)) + off, g[met], 0.36, color=col, label=f"{v} Q", edgecolor=SURFACE)
            ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes, fontsize=7.5)
            if i == 0:
                ax.set_title(lay.replace("_", " "), loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel + " (of 20)")
            ax.set_ylim(0, 21)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("What each fix does (visibility brain, σ = 0.3, α = 1, q_init -1): failure (top) and hazard-route side effect (bottom)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Phase C / D tables
# ---------------------------------------------------------------------------
def simple(name: str, keys: list[str]) -> pd.DataFrame | None:
    p = RESULTS / f"{name}_runs.csv"
    if not p.exists():
        return None
    s = agg(pd.read_csv(p), keys)
    s.to_csv(RESULTS / f"summary_{name}.csv", index=False)
    return s


def plot_qinit(s: pd.DataFrame, out: Path) -> None:
    tests = [("main", "k_frozen", "Part 1 freeze (σ 0.3): seeds frozen"), ("praise_farm", "k_failed", "praise farm (σ 0.03 / 0.3): seeds not reaching the goal"),
             ("distract", "k_hazard_route", "distract-then-err (k 10): seeds taking the shortcut")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), sharey=True, squeeze=False)
    for ax, (t, met, title) in zip(axes[0], tests):
        d = s[s.test == t]
        for v, col in (("plain", "#eb6834"), ("double", "#2a78d6")):
            for sg, ls in ((0.03, "--"), (0.3, "-")):
                g = d[(d.variant == v) & (d.noise_sigma == sg)].sort_values("q_init")
                if g.empty:
                    continue
                ax.plot(g.q_init, g[met], color=col, ls=ls, marker="o", ms=5, lw=1.6, label=f"{v} Q, σ {sg:g}", markeredgecolor=SURFACE)
        ax.set_xticks([-1, -0.5, 0]); ax.set_xlabel("Q-table init"); ax.set_title(title, loc="left", fontsize=8.5); ax.set_ylim(-0.5, 21)
    axes[0, 0].set_ylabel("seeds (of 20)"); axes[0, 2].legend(fontsize=7)
    fig.suptitle("q_init sensitivity of the three headline exploits (visibility brain, α = 1, 100k steps, 20 seeds per point)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    pd.set_option("display.width", 260)
    if (RESULTS / "layouts_runs.csv").exists():
        print("== Phase A: layouts, classified"); print(classify_layouts().to_string(index=False))
    if (RESULTS / "fixes2_runs.csv").exists():
        print("== Phase B: fixes2"); print(fixes2().round(3).to_string(index=False))
    for name, keys in [("qinit", ["test", "variant", "q_init", "noise_sigma"]), ("double300k", ["test", "variant", "noise_sigma"]),
                       ("realnoise20", ["condition", "variant", "subject"]), ("praise_fixes", ["variant", "observability"]),
                       ("deep_fixes", ["variant", "observability"]), ("shaping", ["condition", "praise_kind", "variant"]),
                       ("distractor_robust", ["distractor_cost", "distracted_observable", "variant"])]:
        s = simple(name, keys)
        if s is not None:
            print(f"== {name}"); print(s.round(3).to_string(index=False))
            if name == "qinit":
                plot_qinit(s, FIGURES / "review_qinit_sensitivity.png")


if __name__ == "__main__":
    main()
