"""
Publication-Quality Figure Generation
======================================
Generates all thesis figures from posthoc_scores.csv and analysis results.

Figures produced:
1. Per-dimension comparison boxplots (RQ1)
2. Trade-off heatmap (RQ2)
3. Radar/spider charts per strategy (RQ2)
4. Temporal trajectory line plots (RQ3)
5. Judge reliability scatter plots (RQ4)
6. Intervention frequency chart
7. Composite score comparison bar chart

Usage:
    python experiments/generate_figures.py

Output:
    paper/figures/  (PNG + PDF for each figure)
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# ─── Style ───────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.figsize": (10, 6),
    "savefig.bbox": "tight",
})

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
POSTHOC_CSV = ROOT / "runs" / "posthoc_scores" / "posthoc_scores.csv"
RESULTS_JSON = ROOT / "experiments" / "outputs" / "comparative" / "comparative_results.json"
HUMAN_RESULTS = ROOT / "experiments" / "outputs" / "human_validation" / "human_validation_results.json"
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
    "logical_consistency": "Log. Consist.",
    "argument_strength": "Arg. Strength",
    "document_grounding": "Doc. Ground.",
    "responsiveness": "Responsive.",
    "stance_differentiation": "Stance Diff.",
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
    "baseline_1": "#94a3b8",
    "baseline_2": "#a78bfa",
    "baseline_3": "#fbbf24",
    "baseline_4": "#fb923c",
}

STRATEGY_COLORS = {
    "strategy_a": "#ef4444",
    "strategy_b": "#3b82f6",
    "strategy_c": "#22c55e",
    "strategy_d": "#8b5cf6",
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

    return df, results, human


# ─── Figure 1: Per-dimension boxplots (RQ1) ──────────────────────────────────

def fig_rq1_boxplots(df):
    """Boxplots comparing all 8 conditions per dimension."""
    print("Generating RQ1 boxplots...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    ordered_labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    palette = [ALL_COLORS[c] for c in CONDITION_ORDER]

    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        data_for_plot = []
        labels_for_plot = []
        for cid in CONDITION_ORDER:
            vals = df[df["condition_id"] == cid][dim].dropna()
            data_for_plot.append(vals.values)
            labels_for_plot.append(CONDITION_LABELS[cid])

        bp = ax.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))

        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(DIM_SHORT[dim], fontweight="bold")
        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel("Score" if i % 4 == 0 else "")
        ax.tick_params(axis="x", rotation=45)

    # Use last subplot for legend
    axes[7].axis("off")
    handles = [mpatches.Patch(color=ALL_COLORS[c], alpha=0.7, label=CONDITION_LABELS[c])
               for c in CONDITION_ORDER]
    axes[7].legend(handles=handles, loc="center", fontsize=9, ncol=2)

    fig.suptitle("Discourse Quality by Condition — Per-Dimension Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "rq1_boxplots")


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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Mean Score (1–5)", shrink=0.8)
    ax.set_title("RQ2: Mean Discourse Quality Scores — Conditions × Dimensions", fontweight="bold")

    # Draw dividing line between baselines and strategies
    ax.axhline(3.5, color="white", linewidth=2)

    plt.tight_layout()
    save_fig(fig, "rq2_heatmap")


# ─── Figure 3: Radar charts per strategy (RQ2) ───────────────────────────────

def fig_rq2_radar(df):
    """Spider/radar charts comparing each strategy against baseline_1."""
    print("Generating RQ2 radar charts...")

    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    n = len(strategy_ids)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    # Baseline reference
    bl = df[df["condition_id"] == "baseline_1"]
    bl_vals = [bl[dim].mean() for dim in DIMENSIONS]
    bl_vals += bl_vals[:1]

    for i, cid in enumerate(strategy_ids):
        ax = axes[i]
        cond = df[df["condition_id"] == cid]
        vals = [cond[dim].mean() for dim in DIMENSIONS]
        vals += vals[:1]

        ax.plot(angles, bl_vals, "o--", color="#94a3b8", linewidth=1, markersize=3, label="No moderation")
        ax.fill(angles, bl_vals, color="#94a3b8", alpha=0.1)
        ax.plot(angles, vals, "o-", color=STRATEGY_COLORS[cid], linewidth=2, markersize=4,
                label=CONDITION_LABELS[cid])
        ax.fill(angles, vals, color=STRATEGY_COLORS[cid], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIM_SHORT[d][:6] for d in DIMENSIONS], fontsize=7)
        ax.set_ylim(2, 5)
        ax.set_yticks([2, 3, 4, 5])
        ax.set_yticklabels(["2", "3", "4", "5"], fontsize=7)
        ax.set_title(CONDITION_LABELS[cid], fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=6)

    fig.suptitle("RQ2: Strategy Profiles vs No-Moderation Baseline", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "rq2_radar")


# ─── Figure 4: Temporal trajectories (RQ3) ───────────────────────────────────

def fig_rq3_trajectories(df, results):
    """Line plots of per-round mean scores per condition."""
    print("Generating RQ3 trajectory plots...")

    max_round = int(df["round_number"].max())
    rounds = list(range(1, max_round + 1))

    # One figure per key dimension pair
    key_dims = ["responsiveness", "civility", "stance_differentiation", "composite"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, dim in enumerate(key_dims):
        ax = axes[idx]
        for cid in CONDITION_ORDER:
            cond = df[df["condition_id"] == cid]
            traj = []
            for r in rounds:
                rd = cond[cond["round_number"] == r]
                if not rd.empty:
                    traj.append(rd[dim].mean() if dim != "composite" else rd["composite"].mean())
                else:
                    traj.append(np.nan)

            is_strat = cid.startswith("strategy_")
            ax.plot(rounds, traj,
                    color=ALL_COLORS[cid],
                    linewidth=2 if is_strat else 1,
                    linestyle="-" if is_strat else "--",
                    alpha=0.9 if is_strat else 0.5,
                    marker="o" if is_strat else None,
                    markersize=3,
                    label=CONDITION_LABELS[cid])

        title = DIM_SHORT.get(dim, "Composite")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Score")
        ax.set_xlim(0.5, max_round + 0.5)
        ax.set_ylim(1.5, 5.0)
        ax.set_xticks(rounds)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("RQ3: Temporal Quality Trajectories per Condition", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    save_fig(fig, "rq3_trajectories")


# ─── Figure 5: Judge reliability scatter (RQ4) ───────────────────────────────

def fig_rq4_scatter(human):
    """Scatter plots of human vs LLM agreement per dimension."""
    print("Generating RQ4 scatter plot...")

    if human is None:
        print("  Skipping — no human validation data.")
        return

    per_dim = human["per_dimension"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: alpha vs ±1 agreement
    ax = axes[0]
    dims_list = [r["dimension"] for r in per_dim]
    alphas = [r["alpha"] for r in per_dim]
    within1 = [r["pct_within1"] * 100 for r in per_dim]

    colors = sns.color_palette("husl", len(DIMENSIONS))
    for i, (a, w, d) in enumerate(zip(alphas, within1, dims_list)):
        ax.scatter(a, w, color=colors[i], s=80, zorder=5)
        ax.annotate(DIM_SHORT[d], (a, w), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    ax.axhline(67, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6, label="67% threshold")
    ax.axvline(0.67, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6, label="α=0.67")
    ax.set_xlabel("Krippendorff's α")
    ax.set_ylabel("±1 Agreement (%)")
    ax.set_title("Inter-Annotator Agreement")
    ax.legend(fontsize=7)

    # Right: Spearman ρ vs ±1 vs LLM
    ax = axes[1]
    rhos = [r["spearman_rho"] for r in per_dim]
    within1_llm = [r["within1_vs_llm"] * 100 for r in per_dim]

    for i, (rho, w, d) in enumerate(zip(rhos, within1_llm, dims_list)):
        ax.scatter(rho, w, color=colors[i], s=80, zorder=5)
        ax.annotate(DIM_SHORT[d], (rho, w), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    ax.axhline(67, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6, label="ρ=0.5")
    ax.set_xlabel("Spearman ρ (Human mean vs LLM)")
    ax.set_ylabel("±1 Agreement vs LLM (%)")
    ax.set_title("Human–LLM Agreement")
    ax.legend(fontsize=7)

    fig.suptitle("RQ4: LLM Judge Reliability — Human Validation", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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
    ax.set_title("Composite Discourse Quality Score per Condition", fontweight="bold")

    # Draw dividing line
    ax.axvline(3.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(1.5, 4.3, "Baselines", ha="center", fontsize=9, color="gray")
    ax.text(5.5, 4.3, "Strategies", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    save_fig(fig, "composite_comparison")


# ─── Figure 8: RQ5 topic type interaction ─────────────────────────────────────

def fig_rq5_topic_interaction(df):
    """Grouped bars: strategy scores for empirical vs values-driven topics."""
    print("Generating RQ5 topic interaction chart...")

    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    key_dims = ["civility", "responsiveness", "stance_differentiation", "document_grounding"]

    fig, axes = plt.subplots(1, len(key_dims), figsize=(16, 4))

    for idx, dim in enumerate(key_dims):
        ax = axes[idx]
        x = np.arange(len(strategy_ids))
        width = 0.35

        emp_vals = [df[(df["condition_id"] == c) & (df["topic_type"] == "empirical")][dim].mean()
                    for c in strategy_ids]
        val_vals = [df[(df["condition_id"] == c) & (df["topic_type"] == "values")][dim].mean()
                    for c in strategy_ids]

        ax.bar(x - width/2, emp_vals, width, label="Empirical", color="#3b82f6", alpha=0.7)
        ax.bar(x + width/2, val_vals, width, label="Values-driven", color="#ef4444", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c][:8] for c in strategy_ids], rotation=30, ha="right")
        ax.set_title(DIM_SHORT[dim], fontweight="bold")
        ax.set_ylim(2, 5)
        if idx == 0:
            ax.set_ylabel("Mean Score")
        if idx == len(key_dims) - 1:
            ax.legend(fontsize=8)

    fig.suptitle("RQ5: Strategy Scores by Topic Type (Empirical vs Values-Driven)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, "rq5_topic_interaction")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    df, results, human = load_data()
    print(f"Loaded {len(df)} turns\n")

    fig_rq1_boxplots(df)
    fig_rq2_heatmap(df)
    fig_rq2_radar(df)
    fig_rq3_trajectories(df, results)
    fig_rq4_scatter(human)
    fig_intervention_frequency(results)
    fig_composite_bars(df)
    fig_rq5_topic_interaction(df)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
