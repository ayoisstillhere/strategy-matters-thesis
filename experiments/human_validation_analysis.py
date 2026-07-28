"""
Human Validation Study — Analysis Script
=========================================
Loads the annotation export from the HAKI annotation app and computes:
1. Inter-annotator agreement (Krippendorff's alpha, % exact, % within ±1)
2. LLM-vs-Human agreement (Spearman ρ, % exact, % within ±1)
3. Per-annotator scoring patterns (mean, std, bias tendency)
4. Per-dimension breakdown tables (LaTeX-ready)
5. Revision analysis (Phase 2 anchoring effects)
6. Perceived bias analysis
7. Publication-quality figures

Usage:
    python experiments/human_validation_analysis.py

Output:
    experiments/outputs/human_validation/  (tables, figures, summary)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# Optional: matplotlib for figures
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['figure.dpi'] = 150
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping figures")

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
EXPORT_PATH = ROOT.parent / "annotation-app" / "exports" / "annotation-export-2026-07-28.json"
OUTPUT_DIR = ROOT / "experiments" / "outputs" / "human_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation"
]

DIM_LABELS = {
    "civility": "Civility",
    "relevance": "Relevance",
    "logical_consistency": "Logical Consistency",
    "argument_strength": "Argument Strength",
    "document_grounding": "Document-Grounding",
    "responsiveness": "Responsiveness",
    "stance_differentiation": "Stance Differentiation",
}


# ─── Load Data ───────────────────────────────────────────────────────────────

def load_export(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframes(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build annotators, annotations, and exchanges DataFrames."""
    annotators_df = pd.DataFrame(data["annotators"])
    annotations_df = pd.DataFrame(data["annotations"])
    exchanges_df = pd.DataFrame(data["exchanges"])

    # Expand scores dict into separate columns
    scores_expanded = pd.json_normalize(annotations_df["scores"])
    annotations_df = pd.concat([annotations_df.drop(columns=["scores"]), scores_expanded], axis=1)

    return annotators_df, annotations_df, exchanges_df


# ─── Krippendorff's Alpha ────────────────────────────────────────────────────

def krippendorffs_alpha(reliability_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha for interval data.
    reliability_matrix: shape (n_annotators, n_items), NaN for missing.
    Uses proper 1/(m_u - 1) per-unit weighting per Krippendorff (2011).
    """
    n_ann, n_items = reliability_matrix.shape
    if n_ann < 2 or n_items < 2:
        return np.nan

    Do = 0.0
    n = 0
    pairable_values = []

    for item in range(n_items):
        values = reliability_matrix[:, item]
        values = values[~np.isnan(values)]
        mu = len(values)
        if mu < 2:
            continue
        n += mu
        pairable_values.extend(values)

        # Within-unit pairwise disagreement
        for i in range(mu):
            for j in range(i + 1, mu):
                Do += (values[i] - values[j]) ** 2 * 2 / (mu - 1)

    if n < 2:
        return np.nan

    # Expected disagreement from marginals
    pv = np.array(pairable_values)
    De = 0.0
    for i in range(len(pv)):
        for j in range(i + 1, len(pv)):
            De += (pv[i] - pv[j]) ** 2

    if De == 0:
        return 1.0

    return 1 - ((n - 1) * Do) / (2 * De)


# ─── Pairwise Agreement ──────────────────────────────────────────────────────

def pairwise_agreement(reliability_matrix: np.ndarray, tolerance: int = 0) -> float:
    """
    Compute % pairwise agreement within ±tolerance.
    tolerance=0 → exact agreement, tolerance=1 → within ±1.
    """
    n_ann, n_items = reliability_matrix.shape
    total_pairs = 0
    agree_pairs = 0

    for item in range(n_items):
        values = reliability_matrix[:, item]
        values = values[~np.isnan(values)]
        mu = len(values)
        if mu < 2:
            continue
        for i in range(mu):
            for j in range(i + 1, mu):
                total_pairs += 1
                if abs(values[i] - values[j]) <= tolerance:
                    agree_pairs += 1

    return agree_pairs / total_pairs if total_pairs > 0 else 0.0


# ─── Human vs LLM Agreement ──────────────────────────────────────────────────

def human_vs_llm_agreement(
    annotations_df: pd.DataFrame,
    exchanges_df: pd.DataFrame,
    dim: str,
    tolerance: int = 0
) -> float:
    """% of individual human scores within ±tolerance of LLM score."""
    total = 0
    agree = 0

    for _, ex in exchanges_df.iterrows():
        eid = ex["exchange_id"]
        llm_score = ex["llm_scores"].get(dim)
        if llm_score is None:
            continue
        human_scores = annotations_df[
            (annotations_df["exchange_id"] == eid) &
            (annotations_df["phase"] == "initial") &
            (~annotations_df["perceived_bias"])
        ][dim].dropna()

        for s in human_scores:
            total += 1
            if abs(s - llm_score) <= tolerance:
                agree += 1

    return agree / total if total > 0 else 0.0


# ─── Main Analysis ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HUMAN VALIDATION STUDY — ANALYSIS")
    print("=" * 60)

    # Load
    if not EXPORT_PATH.exists():
        print(f"ERROR: Export file not found at {EXPORT_PATH}")
        sys.exit(1)

    data = load_export(EXPORT_PATH)
    annotators_df, annotations_df, exchanges_df = build_dataframes(data)

    # Filter to included annotators only
    excluded_ids = set(data["excluded_annotator_ids"])
    included_annotators = annotators_df[~annotators_df["id"].isin(excluded_ids)]
    included_ids = set(included_annotators["id"])

    print(f"\nAnnotators: {len(annotators_df)} total, {len(included_annotators)} included")
    print(f"Excluded: {[a['name'] for a in data['annotators'] if a['excluded']]}")

    # Filter annotations to included annotators, initial phase, non-bias
    initial_anns = annotations_df[
        (annotations_df["annotator_id"].isin(included_ids)) &
        (annotations_df["phase"] == "initial") &
        (~annotations_df["perceived_bias"])
    ].copy()

    revised_anns = annotations_df[
        (annotations_df["annotator_id"].isin(included_ids)) &
        (annotations_df["phase"] == "revised")
    ].copy()

    print(f"Initial annotations (non-bias): {len(initial_anns)}")
    print(f"Revised annotations: {len(revised_anns)}")

    # Get ordered annotator and exchange lists
    annotator_ids = sorted(included_ids)
    exchange_ids = sorted(initial_anns["exchange_id"].unique())
    n_ann = len(annotator_ids)
    n_items = len(exchange_ids)

    print(f"Annotators: {n_ann}, Items: {n_items}")

    # ─── Per-Dimension Metrics ────────────────────────────────────────────────

    results = []
    print(f"\n{'Dimension':<25} {'α':>8} {'Exact%':>8} {'±1%':>8} {'ρ':>8} {'ExLLM%':>8} {'±1LLM%':>8} {'μ_H':>7} {'μ_L':>7} {'Δ':>6}")
    print("-" * 100)

    for dim in DIMENSIONS:
        # Build reliability matrix (annotators × items)
        matrix = np.full((n_ann, n_items), np.nan)
        for i, ann_id in enumerate(annotator_ids):
            ann_data = initial_anns[initial_anns["annotator_id"] == ann_id]
            for j, ex_id in enumerate(exchange_ids):
                row = ann_data[ann_data["exchange_id"] == ex_id]
                if not row.empty and not pd.isna(row.iloc[0][dim]):
                    matrix[i, j] = row.iloc[0][dim]

        alpha = krippendorffs_alpha(matrix)
        pct_exact = pairwise_agreement(matrix, tolerance=0)
        pct_within1 = pairwise_agreement(matrix, tolerance=1)

        # Human means per exchange for Spearman
        human_means = []
        llm_vals = []
        for ex_id in exchange_ids:
            col_idx = exchange_ids.tolist().index(ex_id) if hasattr(exchange_ids, 'tolist') else list(exchange_ids).index(ex_id)
            vals = matrix[:, col_idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            human_means.append(np.mean(vals))
            ex_row = exchanges_df[exchanges_df["exchange_id"] == ex_id]
            if not ex_row.empty:
                llm_vals.append(ex_row.iloc[0]["llm_scores"].get(dim, 0))

        if len(human_means) >= 3:
            rho, p_val = stats.spearmanr(human_means, llm_vals)
        else:
            rho, p_val = np.nan, np.nan

        # Human vs LLM agreement
        exact_vs_llm = human_vs_llm_agreement(annotations_df[annotations_df["annotator_id"].isin(included_ids)], exchanges_df, dim, tolerance=0)
        within1_vs_llm = human_vs_llm_agreement(annotations_df[annotations_df["annotator_id"].isin(included_ids)], exchanges_df, dim, tolerance=1)

        mean_human = np.nanmean(human_means) if human_means else 0
        mean_llm = np.mean(llm_vals) if llm_vals else 0
        delta = mean_llm - mean_human

        results.append({
            "dimension": dim,
            "label": DIM_LABELS[dim],
            "alpha": alpha,
            "pct_exact": pct_exact,
            "pct_within1": pct_within1,
            "spearman_rho": rho,
            "spearman_p": p_val,
            "exact_vs_llm": exact_vs_llm,
            "within1_vs_llm": within1_vs_llm,
            "mean_human": mean_human,
            "mean_llm": mean_llm,
            "delta": delta,
        })

        print(f"{DIM_LABELS[dim]:<25} {alpha:>8.3f} {pct_exact*100:>7.1f}% {pct_within1*100:>7.1f}% {rho:>8.3f} {exact_vs_llm*100:>7.1f}% {within1_vs_llm*100:>7.1f}% {mean_human:>7.2f} {mean_llm:>7.2f} {delta:>+5.2f}")

    results_df = pd.DataFrame(results)

    # ─── Summary Statistics ───────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean α across dimensions:       {results_df['alpha'].mean():.3f}")
    print(f"Mean ±1 agreement (inter-ann):  {results_df['pct_within1'].mean()*100:.1f}%")
    print(f"Mean ±1 agreement (vs LLM):     {results_df['within1_vs_llm'].mean()*100:.1f}%")
    print(f"Mean Spearman ρ:                {results_df['spearman_rho'].mean():.3f}")
    print(f"Mean LLM bias (LLM - Human):    {results_df['delta'].mean():+.2f}")

    # ─── Per-Annotator Analysis ───────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PER-ANNOTATOR SCORING PATTERNS")
    print("=" * 60)

    annotator_stats = []
    for ann_id in annotator_ids:
        ann_data = initial_anns[initial_anns["annotator_id"] == ann_id]
        ann_info = included_annotators[included_annotators["id"] == ann_id].iloc[0]
        scores_all = ann_data[DIMENSIONS].values.flatten()
        scores_all = scores_all[~np.isnan(scores_all)]

        annotator_stats.append({
            "name": ann_info["name"],
            "n_annotations": len(ann_data),
            "mean_score": np.mean(scores_all),
            "std_score": np.std(scores_all),
            "median_score": np.median(scores_all),
            "time_hours": None,
        })

        if ann_info["started_at"] and ann_info["finished_at"]:
            start = pd.Timestamp(ann_info["started_at"])
            end = pd.Timestamp(ann_info["finished_at"])
            annotator_stats[-1]["time_hours"] = (end - start).total_seconds() / 3600

    ann_stats_df = pd.DataFrame(annotator_stats)
    print(f"\n{'Name':<20} {'N':>4} {'Mean':>6} {'Std':>6} {'Median':>7} {'Time':>8}")
    print("-" * 60)
    for _, row in ann_stats_df.iterrows():
        time_str = f"{row['time_hours']:.1f}h" if row['time_hours'] else "—"
        print(f"{row['name']:<20} {row['n_annotations']:>4} {row['mean_score']:>6.2f} {row['std_score']:>6.2f} {row['median_score']:>7.1f} {time_str:>8}")

    # ─── Revision Analysis (Phase 2) ─────────────────────────────────────────

    print("\n" + "=" * 60)
    print("REVISION ANALYSIS (Phase 2 — Anchoring Effect)")
    print("=" * 60)

    total_revised = len(revised_anns)
    changed = revised_anns["changed_after_reveal"].sum()
    revision_rate = changed / total_revised if total_revised > 0 else 0
    print(f"Total Phase 2 annotations: {total_revised}")
    print(f"Changed after LLM reveal:  {changed} ({revision_rate*100:.1f}%)")

    # Direction of change analysis
    if total_revised > 0:
        # Merge initial and revised on (annotator_id, exchange_id)
        initial_for_merge = initial_anns[["annotator_id", "exchange_id"] + DIMENSIONS].copy()
        initial_for_merge.columns = ["annotator_id", "exchange_id"] + [f"{d}_initial" for d in DIMENSIONS]

        revised_for_merge = revised_anns[["annotator_id", "exchange_id"] + DIMENSIONS].copy()
        revised_for_merge.columns = ["annotator_id", "exchange_id"] + [f"{d}_revised" for d in DIMENSIONS]

        merged = initial_for_merge.merge(revised_for_merge, on=["annotator_id", "exchange_id"])

        print(f"\n{'Dimension':<25} {'→LLM':>6} {'←LLM':>6} {'Net':>6}")
        print("-" * 50)
        for dim in DIMENSIONS:
            initial_col = f"{dim}_initial"
            revised_col = f"{dim}_revised"
            diffs = merged[revised_col] - merged[initial_col]
            # Need LLM scores to determine direction
            toward_llm = 0
            away_llm = 0
            for _, row in merged.iterrows():
                d = row[revised_col] - row[initial_col]
                if d == 0:
                    continue
                ex_row = exchanges_df[exchanges_df["exchange_id"] == row["exchange_id"]]
                if ex_row.empty:
                    continue
                llm_score = ex_row.iloc[0]["llm_scores"].get(dim, 0)
                # Toward LLM if revised is closer to LLM than initial
                dist_initial = abs(row[initial_col] - llm_score)
                dist_revised = abs(row[revised_col] - llm_score)
                if dist_revised < dist_initial:
                    toward_llm += 1
                elif dist_revised > dist_initial:
                    away_llm += 1

            total_changes = toward_llm + away_llm
            net = toward_llm - away_llm
            print(f"{DIM_LABELS[dim]:<25} {toward_llm:>6} {away_llm:>6} {net:>+6}")

    # ─── Perceived Bias Analysis ──────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PERCEIVED BIAS ANALYSIS")
    print("=" * 60)

    bias_anns = annotations_df[
        (annotations_df["annotator_id"].isin(included_ids)) &
        (annotations_df["phase"] == "initial") &
        (annotations_df["perceived_bias"])
    ]
    total_initial_all = annotations_df[
        (annotations_df["annotator_id"].isin(included_ids)) &
        (annotations_df["phase"] == "initial")
    ]
    bias_rate = len(bias_anns) / len(total_initial_all) if len(total_initial_all) > 0 else 0
    print(f"Bias-flagged exchanges: {len(bias_anns)} / {len(total_initial_all)} ({bias_rate*100:.1f}%)")

    if len(bias_anns) > 0:
        # Which exchanges were flagged most
        bias_counts = bias_anns["exchange_id"].value_counts().head(10)
        print(f"\nMost-flagged exchanges:")
        for ex_id, count in bias_counts.items():
            ex_row = exchanges_df[exchanges_df["exchange_id"] == ex_id]
            agent = ex_row.iloc[0]["agent_name"] if not ex_row.empty else "?"
            print(f"  Exchange {ex_id} ({agent}): {count}/{n_ann} annotators flagged")

    # ─── LaTeX Table Output ───────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("LATEX TABLE (for expose.tex)")
    print("=" * 60)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Human Validation Results: Inter-Annotator Agreement and LLM Judge Correlation ($n=5$ annotators, 50 exchanges, 7 dimensions)}
\label{tab:human-validation}
\small
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{\textbf{Inter-Annotator}} & \multicolumn{2}{c}{\textbf{Human vs.\ LLM}} & \multicolumn{2}{c}{\textbf{Means}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\textbf{Dimension} & $\alpha$ & $\pm 1$\% & $\rho$ & $\pm 1$\% & $\mu_H$ & $\mu_L$ \\
\midrule
"""
    for _, row in results_df.iterrows():
        latex += f"{row['label']} & {row['alpha']:.3f} & {row['pct_within1']*100:.1f} & {row['spearman_rho']:.3f} & {row['within1_vs_llm']*100:.1f} & {row['mean_human']:.2f} & {row['mean_llm']:.2f} \\\\\n"

    latex += r"""\midrule
\textbf{Mean} & """ + f"{results_df['alpha'].mean():.3f} & {results_df['pct_within1'].mean()*100:.1f} & {results_df['spearman_rho'].mean():.3f} & {results_df['within1_vs_llm'].mean()*100:.1f} & {results_df['mean_human'].mean():.2f} & {results_df['mean_llm'].mean():.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""

    print(latex)

    # Save LaTeX table
    (OUTPUT_DIR / "human_validation_table.tex").write_text(latex, encoding="utf-8")

    # ─── Save Results JSON ────────────────────────────────────────────────────

    output_summary = {
        "study_params": {
            "n_annotators": n_ann,
            "n_items": n_items,
            "n_annotations_initial": len(initial_anns),
            "n_annotations_revised": len(revised_anns),
            "revision_rate": revision_rate,
            "bias_rate": bias_rate,
            "avg_completion_hours": ann_stats_df["time_hours"].mean(),
        },
        "per_dimension": results_df.to_dict(orient="records"),
        "per_annotator": ann_stats_df.to_dict(orient="records"),
        "summary": {
            "mean_alpha": results_df["alpha"].mean(),
            "mean_within1_inter": results_df["pct_within1"].mean(),
            "mean_within1_vs_llm": results_df["within1_vs_llm"].mean(),
            "mean_spearman": results_df["spearman_rho"].mean(),
            "mean_llm_bias": results_df["delta"].mean(),
        }
    }

    with open(OUTPUT_DIR / "human_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, default=str)

    # ─── Figures ──────────────────────────────────────────────────────────────

    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        dims_short = [d.replace("_", "\n") for d in DIMENSIONS]

        # 1. Alpha + ±1 agreement
        ax = axes[0]
        x = np.arange(len(DIMENSIONS))
        ax.bar(x - 0.2, results_df["alpha"], 0.4, label="Krippendorff's α", color="#ef4444", alpha=0.8)
        ax.bar(x + 0.2, results_df["pct_within1"], 0.4, label="±1 Agreement", color="#22c55e", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dims_short, fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title("Inter-Annotator Agreement")
        ax.legend(fontsize=8)
        ax.axhline(0.67, color="#f59e0b", linestyle="--", linewidth=0.8, label="α=0.67")
        ax.set_ylim(-0.2, 1.0)

        # 2. Spearman ρ + ±1 vs LLM
        ax = axes[1]
        ax.bar(x - 0.2, results_df["spearman_rho"], 0.4, label="Spearman ρ", color="#3b82f6", alpha=0.8)
        ax.bar(x + 0.2, results_df["within1_vs_llm"], 0.4, label="±1 vs LLM", color="#a855f7", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dims_short, fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title("Human vs LLM Agreement")
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="#f59e0b", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.0)

        # 3. Mean comparison (Human vs LLM)
        ax = axes[2]
        ax.bar(x - 0.2, results_df["mean_human"], 0.4, label="Human Mean", color="#64748b", alpha=0.8)
        ax.bar(x + 0.2, results_df["mean_llm"], 0.4, label="LLM Score", color="#f97316", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dims_short, fontsize=7)
        ax.set_ylabel("Mean Score (1–5)")
        ax.set_title("Score Level Comparison")
        ax.legend(fontsize=8)
        ax.set_ylim(1, 5)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "human_validation_overview.png", bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "human_validation_overview.pdf", bbox_inches="tight")
        print(f"\nFigures saved to {OUTPUT_DIR}")
        plt.close()

    # ─── Save CSV for further analysis ────────────────────────────────────────

    results_df.to_csv(OUTPUT_DIR / "per_dimension_metrics.csv", index=False)
    ann_stats_df.to_csv(OUTPUT_DIR / "per_annotator_stats.csv", index=False)
    initial_anns.to_csv(OUTPUT_DIR / "initial_annotations.csv", index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
