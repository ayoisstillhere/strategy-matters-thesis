"""
Publication-Quality Figure Generation
======================================
Generates all thesis figures from posthoc_scores.csv and analysis results.

Figures produced:
1. Score distributions per condition and dimension (RQ1)
2. Trade-off heatmap (RQ2)
3. Forest plot of strategy effects vs. no moderation (RQ2)
4. Temporal trajectories with CIs (RQ3)
5. Judge reliability scatter plots (RQ4)
6. Intervention frequency chart
7. Composite score comparison bar chart
8. Moderation effect by topic type (RQ5)

Usage:
    python experiments/generate_figures.py

Output:
    paper/figures/  (PNG + PDF for each figure)
"""

import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ─── Style ───────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.0)
# Figures are drawn at the printed width (about 6.3 in), so these sizes are the printed sizes.
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (6.3, 4),
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
POSTHOC_CSV = ROOT / "runs" / "posthoc_scores" / "posthoc_scores.csv"
RESULTS_JSON = ROOT / "experiments" / "outputs" / \
    "comparative" / "comparative_results.json"
HUMAN_RESULTS = ROOT / "experiments" / "outputs" / \
    "human_validation" / "human_validation_results.json"
HUMAN_TIERS = ROOT / "experiments" / "outputs" / \
    "judge_validity" / "human_tiers.json"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation"
]

DIM_SHORT = {
    "civility": "Civility",
    "relevance": "Relevance",
    "logical_consistency": "Log. consistency",
    "argument_strength": "Arg. strength",
    "document_grounding": "Doc.-grounding",
    "responsiveness": "Responsiveness",
    "stance_differentiation": "Stance diff.",
    "composite": "Composite",
}

CONDITION_LABELS = {
    "baseline_1": "None",
    "baseline_2": "Nudge",
    "baseline_3": "Habermas",
    "baseline_4": "Random",
    "strategy_a": "De-escalation",
    "strategy_b": "Reframing",
    "strategy_c": "Fact-reminder",
    "strategy_d": "Common-ground",
}

CONDITION_ORDER = [
    "baseline_1", "baseline_2", "baseline_3", "baseline_4",
    "strategy_a", "strategy_b", "strategy_c", "strategy_d"
]

BASELINE_COLORS = {
    "baseline_1": "#1e293b",
    "baseline_2": "#64748b",
    "baseline_3": "#a16207",
    "baseline_4": "#c2410c",
}

STRATEGY_COLORS = {
    "strategy_a": "#dc2626",
    "strategy_b": "#2563eb",
    "strategy_c": "#16a34a",
    "strategy_d": "#7c3aed",
}

ALL_COLORS = {**BASELINE_COLORS, **STRATEGY_COLORS}

TOPIC_TYPES = {
    "mindestlohn": "empirical",
    "rentenpolitik": "empirical",
    "sozialpolitik": "values",
    "migrationspolitik": "values",
}


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {name}.png/.pdf")


# ─── Load Data ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(POSTHOC_CSV)
    # Match comparative_analysis.py: drop the reversed-order ablation runs and the
    # opening turn's responsiveness (always 1 by construction).
    df = df[~df["run_file"].str.contains(
        "ablation_turn_order", regex=False)].copy()
    opening = (df["round_number"] == 1) & (df["turn_in_round"] == 1)
    df["responsiveness"] = df["responsiveness"].astype(float).where(~opening)
    df["topic_type"] = df["topic_id"].map(TOPIC_TYPES)
    df["condition_label"] = df["condition_id"].map(CONDITION_LABELS)
    df["is_strategy"] = df["condition_id"].str.startswith("strategy_")

    results = None
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            results = json.load(f)

    human = None
    if HUMAN_RESULTS.exists():
        with open(HUMAN_RESULTS, encoding="utf-8") as f:
            human = json.load(f)

    tiers = None
    if HUMAN_TIERS.exists():
        with open(HUMAN_TIERS, encoding="utf-8") as f:
            tiers = json.load(f)

    return df, results, human, tiers


# ─── Debate-level bootstrap helpers ──────────────────────────────────────────

N_BOOT = 2000
SEED = 2026


def debate_means(df, metric, by=("run_id",)):
    """Mean of `metric` per debate (plus any extra grouping keys)."""
    keys = list(by) + ["condition_id", "topic_id"]
    return df.dropna(subset=[metric]).groupby(keys, as_index=False)[metric].mean()


def boot_mean_ci(runs, metric, rng):
    """Mean over debates with a 95% CI that resamples debates within topics."""
    cells = [g[metric].to_numpy() for _, g in runs.groupby("topic_id")]
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        boots[b] = np.concatenate([rng.choice(c, len(c))
                                  for c in cells]).mean()
    return runs[metric].mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def boot_diff_ci(runs_a, runs_b, metric, rng):
    """Difference in debate means (A - B) with a 95% CI resampling debates within topics."""
    cells_a = [g[metric].to_numpy() for _, g in runs_a.groupby("topic_id")]
    cells_b = [g[metric].to_numpy() for _, g in runs_b.groupby("topic_id")]
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        a = np.concatenate([rng.choice(c, len(c)) for c in cells_a]).mean()
        bb = np.concatenate([rng.choice(c, len(c)) for c in cells_b]).mean()
        boots[b] = a - bb
    diff = runs_a[metric].mean() - runs_b[metric].mean()
    return diff, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ─── Figure 1: Score distributions (RQ1) ─────────────────────────────────────

SCORE_COLORS = {1: "#8c510a", 2: "#d8b365",
                3: "#d9d9d9", 4: "#5ab4ac", 5: "#01665e"}


def fig_rq1_distributions(df):
    """Diverging stacked bars of the 1-5 score distribution per condition and dimension."""
    print("Generating RQ1 score distributions...")

    fig, axes = plt.subplots(2, 4, figsize=(6.3, 4.4), sharex=True)
    axes = axes.flatten()
    ordered = CONDITION_ORDER[::-1]
    y = np.arange(len(ordered))

    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        for j, cid in enumerate(ordered):
            vals = df[df["condition_id"] ==
                      cid][dim].dropna().round().astype(int)
            share = vals.value_counts(normalize=True).reindex(
                range(1, 6), fill_value=0) * 100
            left = -(share[1] + share[2] + share[3] / 2)
            for s in range(1, 6):
                ax.barh(j, share[s], left=left, color=SCORE_COLORS[s], height=0.75,
                        edgecolor="white", linewidth=0.3)
                left += share[s]
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(DIM_SHORT[dim], fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([CONDITION_LABELS[c]
                           for c in ordered] if i % 4 == 0 else [])
        ax.set_xlim(-100, 100)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(["100", "50", "0", "50", "100"])
        ax.grid(axis="y", visible=False)
        if i >= 3:
            ax.set_xlabel("% of turns")
            ax.tick_params(labelbottom=True)

    axes[7].axis("off")
    handles = [mpatches.Patch(color=SCORE_COLORS[s],
                              label=f"Score {s}") for s in range(1, 6)]
    axes[7].legend(handles=handles, loc="center", frameon=False,
                   title="Bars centred on\nthe middle of score 3", title_fontsize=8)

    plt.tight_layout(w_pad=0.6, h_pad=0.8)
    save_fig(fig, "rq1_distributions")


# ─── Figure 2: Trade-off heatmap (RQ2) ───────────────────────────────────────

def fig_rq2_heatmap(df):
    """Heatmap of mean scores: conditions × dimensions."""
    print("Generating RQ2 heatmap...")

    matrix = []
    for cid in CONDITION_ORDER:
        cond = df[df["condition_id"] == cid]
        row = [cond[dim].mean() for dim in DIMENSIONS]
        matrix.append(row)

    matrix = np.array(matrix)
    labels_y = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    labels_x = [DIM_SHORT[d] for d in DIMENSIONS]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=2.0, vmax=5.0)

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.set_yticks(range(len(CONDITION_ORDER)))
    ax.set_yticklabels(labels_y)

    # Add value annotations
    for i in range(len(CONDITION_ORDER)):
        for j in range(len(DIMENSIONS)):
            val = matrix[i, j]
            color = "white" if val < 3.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center",
                    va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Mean Score (1–5)", shrink=0.8)
    ax.set_title(
        "RQ2: Mean Discourse Quality Scores — Conditions × Dimensions", fontweight="bold")

    # Draw dividing line between baselines and strategies
    ax.axhline(3.5, color="white", linewidth=2)

    plt.tight_layout()
    save_fig(fig, "rq2_heatmap")


# ─── Figure 3: Forest plot of strategy effects (RQ2) ─────────────────────────

def fig_rq2_forest(results):
    """Difference from no moderation per dimension, with debate-level 95% CIs."""
    print("Generating RQ2 forest plot...")

    if results is None:
        print("  Skipping — no results data.")
        return

    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    fig, axes = plt.subplots(2, 2, figsize=(
        6.3, 4.6), sharex=True, sharey=True)
    y = np.arange(len(DIMENSIONS))[::-1]

    for ax, cid in zip(axes.flatten(), strategy_ids):
        color = STRATEGY_COLORS[cid]
        for yi, dim in zip(y, DIMENSIONS):
            r = results["rq2"][cid]["vs_none"][dim]
            lo, hi = r["diff_ci"]
            sig = r["p_holm"] < 0.05
            ax.plot([lo, hi], [yi, yi], color=color,
                    linewidth=1.6, solid_capstyle="butt")
            ax.plot(r["diff"], yi, "o", markersize=5, color=color,
                    markerfacecolor=color if sig else "white", markeredgewidth=1.3)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(CONDITION_LABELS[cid], fontweight="bold", color=color)
        ax.set_yticks(y)
        ax.set_yticklabels([DIM_SHORT[d] for d in DIMENSIONS])
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-0.6, len(DIMENSIONS) - 0.4)
        ax.grid(axis="y", visible=False)

    for ax in axes[1]:
        ax.set_xlabel("Difference from no moderation (score points)")

    handles = [
        Line2D([], [], marker="o", color="#374151", markerfacecolor="#374151", linestyle="-",
               label="$p_{\\mathrm{Holm}} < 0.05$"),
        Line2D([], [], marker="o", color="#374151", markerfacecolor="white", linestyle="-",
               label="not significant after correction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.04, 1, 1], w_pad=1.0, h_pad=1.0)
    save_fig(fig, "rq2_forest")


# ─── Figure 4: Temporal trajectories (RQ3) ───────────────────────────────────

def fig_rq3_trajectories(df, results):
    """Per-round means per condition with debate-level 95% CI bands."""
    print("Generating RQ3 trajectory plots...")

    rng = np.random.default_rng(SEED)
    max_round = int(df["round_number"].max())
    rounds = list(range(1, max_round + 1))
    key_dims = ["responsiveness", "document_grounding",
                "stance_differentiation", "composite"]

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), sharex=True)
    axes = axes.flatten()

    for idx, dim in enumerate(key_dims):
        ax = axes[idx]
        per_round = debate_means(df, dim, by=("run_id", "round_number"))
        for cid in CONDITION_ORDER:
            cond = per_round[per_round["condition_id"] == cid]
            stats_ = [boot_mean_ci(
                cond[cond["round_number"] == r], dim, rng) for r in rounds]
            mean, lo, hi = (np.array(v) for v in zip(*stats_))
            is_strat = cid.startswith("strategy_")
            color = ALL_COLORS[cid]
            ax.fill_between(rounds, lo, hi, color=color,
                            alpha=0.10, linewidth=0)
            ax.plot(rounds, mean, color=color,
                    linewidth=1.5 if is_strat else 1.3,
                    linestyle="-" if is_strat else (0, (4, 2)),
                    marker="o" if is_strat else None, markersize=2.5,
                    label=CONDITION_LABELS[cid])

        ax.set_title(DIM_SHORT[dim], fontweight="bold")
        if idx >= 2:
            ax.set_xlabel("Round")
        if idx % 2 == 0:
            ax.set_ylabel("Mean score")
        ax.set_xlim(0.7, max_round + 0.3)
        ax.set_xticks(rounds)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.0, h_pad=1.0)
    save_fig(fig, "rq3_trajectories")


# ─── Figure 5: Judge reliability scatter (RQ4) ───────────────────────────────

def fig_rq4_scatter(human, tiers):
    """Per-dimension inter-annotator and human-LLM agreement."""
    print("Generating RQ4 scatter plot...")

    if human is None:
        print("  Skipping — no human validation data.")
        return

    per_dim = human["per_dimension"]
    dims_list = [r["dimension"] for r in per_dim]
    colors = sns.color_palette("colorblind", len(dims_list))

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))

    # Label offsets (points, horizontal alignment) chosen to avoid overlaps.
    left_offsets = {"civility": (0, -11, "center"), "document_grounding": (6, -3, "left"),
                    "stance_differentiation": (5, 4, "left"), "responsiveness": (6, -3, "left"),
                    "relevance": (5, -9, "left")}
    right_offsets = {"argument_strength": (-5, 3, "right"), "logical_consistency": (0, 7, "center"),
                     "stance_differentiation": (-6, -3, "right"), "document_grounding": (6, -6, "left")}

    def scatter(ax, xs, ys, offsets):
        for i, (x, yv, d) in enumerate(zip(xs, ys, dims_list)):
            dx, dy, ha = offsets.get(d, (5, 3, "left"))
            ax.scatter(x, yv, color=colors[i], s=28, zorder=5)
            ax.annotate(DIM_SHORT[d], (x, yv), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, ha=ha)

    ax = axes[0]
    scatter(ax, [r["alpha"] for r in per_dim], [r["pct_within1"] * 100 for r in per_dim],
            left_offsets)
    ax.axvline(0.667, color="#b91c1c", linestyle="--", linewidth=0.9,
               label="$\\alpha = 0.667$\n(tentative conclusions)")
    ax.set_xlim(-0.15, 0.75)
    ax.set_xlabel("Krippendorff's $\\alpha$")
    ax.set_ylabel("$\\pm 1$ agreement (%)")
    ax.set_title("Between annotators", fontweight="bold")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.62), frameon=True)

    ax = axes[1]
    scatter(ax, [r["spearman_rho"] for r in per_dim],
            [r["within1_vs_llm"] * 100 for r in per_dim], right_offsets)
    if tiers is not None:
        chance = tiers["all"]["within1_chance"] * 100
        ax.axhline(chance, color="#b91c1c", linestyle="--", linewidth=0.9,
                   label=f"Chance ({chance:.1f}%)")
        ax.legend(loc="center right", frameon=True)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Spearman $\\rho$ (human mean vs. judge)")
    ax.set_ylabel("$\\pm 1$ agreement with judge (%)")
    ax.set_title("Annotators vs. judge", fontweight="bold")

    plt.tight_layout(w_pad=1.5)
    save_fig(fig, "rq4_judge_reliability")


# ─── Figure 6: Intervention frequency ────────────────────────────────────────

def fig_intervention_frequency(results):
    """Bar chart of intervention counts per condition."""
    print("Generating intervention frequency chart...")

    if results is None or "intervention_summary" not in results:
        print("  Skipping — no results data.")
        return

    per_cond = results["intervention_summary"]["per_condition"]
    if not per_cond:
        print("  No intervention data.")
        return

    conditions = [c for c in CONDITION_ORDER if c in per_cond]
    counts = [per_cond.get(c, 0) for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]
    colors = [ALL_COLORS[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Total Interventions (active + silent)")
    ax.set_title("Intervention Frequency per Condition", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_fig(fig, "intervention_frequency")


# ─── Figure 7: Composite comparison bar chart ─────────────────────────────────

def fig_composite_bars(df):
    """Simple bar chart of composite scores per condition."""
    print("Generating composite bar chart...")

    means = []
    stds = []
    for cid in CONDITION_ORDER:
        vals = df[df["condition_id"] == cid]["composite"]
        means.append(vals.mean())
        stds.append(vals.std())

    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    colors = [ALL_COLORS[c] for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=3, edgecolor="white", error_kw=dict(lw=1))

    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Composite Score (mean ± SD)")
    ax.set_ylim(2.5, 4.5)
    ax.set_title("Composite Discourse Quality Score per Condition",
                 fontweight="bold")

    # Draw dividing line
    ax.axvline(3.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(1.5, 4.3, "Baselines", ha="center", fontsize=9, color="gray")
    ax.text(5.5, 4.3, "Strategies", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    save_fig(fig, "composite_comparison")


# ─── Figure 8: RQ5 topic type interaction ─────────────────────────────────────

def fig_rq5_topic_interaction(df, results):
    """Moderation effect (strategies pooled minus no moderation) by topic type, with CIs."""
    print("Generating RQ5 topic interaction chart...")

    rng = np.random.default_rng(SEED)
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    types = [("empirical", "Empirical topics", "#0f766e", "o"),
             ("values", "Values-driven topics", "#b45309", "s")]

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    y = np.arange(len(DIMENSIONS))[::-1]

    for k, (ttype, label, color, marker) in enumerate(types):
        offset = 0.17 if k == 0 else -0.17
        for yi, dim in zip(y, DIMENSIONS):
            runs = debate_means(df[df["topic_type"] == ttype], dim)
            diff, lo, hi = boot_diff_ci(runs[runs["condition_id"].isin(strategy_ids)],
                                        runs[runs["condition_id"] == "baseline_1"], dim, rng)
            ax.plot([lo, hi], [yi + offset] * 2, color=color, linewidth=1.5)
            ax.plot(diff, yi + offset, marker, color=color, markersize=4.5,
                    label=label if yi == y[0] else None)

    if results is not None:
        for yi, dim in zip(y, DIMENSIONS):
            ax.text(1.02, yi, f"$p = {results['rq5'][dim]['p']:.2f}$",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([DIM_SHORT[d] for d in DIMENSIONS])
    ax.set_xlabel("Strategies (pooled) minus no moderation (score points)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_fig(fig, "rq5_topic_interaction")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    df, results, human, tiers = load_data()
    print(f"Loaded {len(df)} turns\n")

    fig_rq1_distributions(df)
    fig_rq2_heatmap(df)
    fig_rq2_forest(results)
    fig_rq3_trajectories(df, results)
    fig_rq4_scatter(human, tiers)
    fig_intervention_frequency(results)
    fig_composite_bars(df)
    fig_rq5_topic_interaction(df, results)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
