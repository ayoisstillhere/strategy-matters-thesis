"""
Comparative Analysis — Full RQ1–RQ5 Analysis
=============================================
Loads posthoc_scores.csv and intervention data from experiment JSON files.
Performs all statistical tests and generates LaTeX tables for the thesis.

RQ1: Moderated vs Baselines (per-dimension)
RQ2: Per-strategy trade-off characterization
RQ3: Temporal quality trajectories
RQ4: (Handled in human_validation_analysis.py)
RQ5: Topic type interaction analysis
+ Within-condition causal analysis (silent control)
+ Speaking-order ablation

Usage:
    python experiments/comparative_analysis.py

Output:
    experiments/outputs/comparative/  (LaTeX tables, JSON summaries)
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
POSTHOC_CSV = ROOT / "runs" / "posthoc_scores" / "posthoc_scores.csv"
EXPERIMENT_DIR = ROOT / "runs" / "experiment"
ABLATION_DIR = ROOT / "runs" / "ablation_turn_order"
OUTPUT_DIR = ROOT / "experiments" / "outputs" / "comparative"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation"
]

DIM_LABELS = {
    "civility": "Civility",
    "relevance": "Relevance",
    "logical_consistency": "Log.\ Consistency",
    "argument_strength": "Arg.\ Strength",
    "document_grounding": "Doc.\ Grounding",
    "responsiveness": "Responsiveness",
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

TOPIC_TYPES = {
    "mindestlohn": "empirical",
    "rentenpolitik": "empirical",
    "sozialpolitik": "values",
    "migrationspolitik": "values",
}


# ─── Load Data ───────────────────────────────────────────────────────────────

def load_posthoc() -> pd.DataFrame:
    df = pd.read_csv(POSTHOC_CSV)
    df["topic_type"] = df["topic_id"].map(TOPIC_TYPES)
    df["condition_label"] = df["condition_id"].map(CONDITION_LABELS)
    df["is_strategy"] = df["condition_id"].str.startswith("strategy_")
    df["is_moderated"] = df["condition_id"] != "baseline_1"
    return df


def load_interventions() -> pd.DataFrame:
    """Load all interventions from experiment JSON files."""
    rows = []
    for f in glob.glob(str(EXPERIMENT_DIR / "**" / "run_*.json"), recursive=True):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        config = d["config"]
        for iv in d.get("interventions", []):
            rows.append({
                "run_id": d["run_id"],
                "condition_id": config["condition_id"],
                "topic_id": config["topic_id"],
                "topic_type": TOPIC_TYPES[config["topic_id"]],
                "run_number": config["run_number"],
                "round_number": iv["round_number"],
                "trigger_dimension": iv.get("trigger_dimension"),
                "trigger_score": iv.get("trigger_score"),
                "silent_control": iv.get("silent_control", False),
                "strategy": iv.get("strategy"),
                "has_text": bool(iv.get("intervention_text")),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_ablation() -> pd.DataFrame:
    """Load ablation turn-order runs."""
    rows = []
    for f in glob.glob(str(ABLATION_DIR / "**" / "run_*.json"), recursive=True):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        config = d["config"]
        for turn in d["turns"]:
            rows.append({
                "run_id": d["run_id"],
                "condition_id": config["condition_id"],
                "topic_id": config["topic_id"],
                "run_number": config["run_number"],
                "round_number": turn["round_number"],
                "turn_in_round": turn["turn_in_round"],
                "agent_name": turn["agent_name"],
                **{dim: turn["scores"][dim] for dim in DIMENSIONS},
                "composite": sum(turn["scores"][d] for d in DIMENSIONS) / len(DIMENSIONS),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Effect Size ──────────────────────────────────────────────────────────────

def cliff_delta(x, y):
    """Cliff's delta effect size for ordinal data."""
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (n_x * n_y)


def cliff_magnitude(d):
    d = abs(d)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


# ─── RQ1: Moderated vs Baselines ─────────────────────────────────────────────

def rq1_analysis(df: pd.DataFrame) -> dict:
    """Compare aggregated strategy conditions vs each baseline per dimension."""
    print("\n" + "=" * 70)
    print("RQ1: MODERATED (Strategies A–D) vs BASELINES")
    print("=" * 70)

    strategies = df[df["is_strategy"]]
    results = {}

    for baseline_id in ["baseline_1", "baseline_2", "baseline_3", "baseline_4"]:
        baseline = df[df["condition_id"] == baseline_id]
        bl_label = CONDITION_LABELS[baseline_id]
        print(f"\n--- Strategies vs {bl_label} (Baseline) ---")
        print(f"{'Dimension':<20} {'Strat μ':>8} {'BL μ':>8} {'U':>10} {'p':>10} {'Δ_cliff':>8} {'Mag':>12}")
        print("-" * 80)

        bl_results = {}
        for dim in DIMENSIONS:
            s_vals = strategies[dim].dropna()
            b_vals = baseline[dim].dropna()
            u_stat, p_val = stats.mannwhitneyu(s_vals, b_vals, alternative="two-sided")
            delta = cliff_delta(s_vals.values, b_vals.values)
            mag = cliff_magnitude(delta)

            bl_results[dim] = {
                "strategy_mean": float(s_vals.mean()),
                "baseline_mean": float(b_vals.mean()),
                "U": float(u_stat),
                "p": float(p_val),
                "cliff_delta": float(delta),
                "magnitude": mag,
                "significant": p_val < 0.05,
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"{DIM_LABELS[dim]:<20} {s_vals.mean():>8.3f} {b_vals.mean():>8.3f} {u_stat:>10.0f} {p_val:>10.4f}{sig} {delta:>+8.3f} {mag:>12}")

        results[baseline_id] = bl_results

    # Also: Kruskal-Wallis across ALL 8 conditions
    print(f"\n--- Kruskal-Wallis (all 8 conditions) ---")
    print(f"{'Dimension':<20} {'H':>10} {'p':>10} {'Sig':>5}")
    print("-" * 50)
    kw_results = {}
    for dim in DIMENSIONS:
        groups = [df[df["condition_id"] == c][dim].dropna().values for c in CONDITION_ORDER]
        h_stat, p_val = stats.kruskal(*groups)
        kw_results[dim] = {"H": float(h_stat), "p": float(p_val)}
        sig = "*" if p_val < 0.05 else ""
        print(f"{DIM_LABELS[dim]:<20} {h_stat:>10.2f} {p_val:>10.4f}{sig}")

    results["kruskal_wallis"] = kw_results
    return results


# ─── RQ2: Per-strategy trade-off characterization ────────────────────────────

def rq2_analysis(df: pd.DataFrame, interventions_df: pd.DataFrame) -> dict:
    """Per-strategy per-dimension analysis + post-intervention deltas."""
    print("\n" + "=" * 70)
    print("RQ2: PER-STRATEGY TRADE-OFF CHARACTERIZATION")
    print("=" * 70)

    results = {}

    # Per-strategy mean scores
    print(f"\n{'Condition':<15}", end="")
    for dim in DIMENSIONS:
        print(f" {DIM_LABELS[dim][:8]:>8}", end="")
    print(f" {'Composite':>9}")
    print("-" * 95)

    for cid in CONDITION_ORDER:
        cond = df[df["condition_id"] == cid]
        means = {dim: float(cond[dim].mean()) for dim in DIMENSIONS}
        composite = float(cond["composite"].mean())
        means["composite"] = composite
        results[cid] = {"means": means}

        print(f"{CONDITION_LABELS[cid]:<15}", end="")
        for dim in DIMENSIONS:
            print(f" {means[dim]:>8.3f}", end="")
        print(f" {composite:>9.3f}")

    # Pairwise strategy comparisons
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    print(f"\n--- Pairwise Mann-Whitney U between strategies ---")
    for dim in DIMENSIONS:
        print(f"\n  {DIM_LABELS[dim]}:")
        for s1, s2 in combinations(strategy_ids, 2):
            v1 = df[df["condition_id"] == s1][dim].dropna()
            v2 = df[df["condition_id"] == s2][dim].dropna()
            u, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            delta = cliff_delta(v1.values, v2.values)
            sig = "*" if p < 0.05 else ""
            if p < 0.05:
                print(f"    {CONDITION_LABELS[s1]:>15} vs {CONDITION_LABELS[s2]:<15}: Δ={delta:+.3f} p={p:.4f}{sig}")

    # Post-intervention quality deltas
    if not interventions_df.empty:
        print(f"\n--- Post-intervention quality deltas (next round - trigger round) ---")
        active_ivs = interventions_df[~interventions_df["silent_control"]]
        strategy_ivs = active_ivs[active_ivs["condition_id"].str.startswith("strategy_")]

        for cid in strategy_ids:
            cond_ivs = strategy_ivs[strategy_ivs["condition_id"] == cid]
            if cond_ivs.empty:
                continue

            deltas_by_dim = defaultdict(list)
            for _, iv in cond_ivs.iterrows():
                run_data = df[(df["run_id"] == iv["run_id"])]
                pre = run_data[run_data["round_number"] == iv["round_number"]]
                post = run_data[run_data["round_number"] == iv["round_number"] + 1]
                if pre.empty or post.empty:
                    continue
                for dim in DIMENSIONS:
                    d = post[dim].mean() - pre[dim].mean()
                    deltas_by_dim[dim].append(d)

            if deltas_by_dim:
                print(f"\n  {CONDITION_LABELS[cid]} (n={len(next(iter(deltas_by_dim.values())))} interventions):")
                for dim in DIMENSIONS:
                    vals = deltas_by_dim[dim]
                    if vals:
                        mean_d = np.mean(vals)
                        print(f"    {DIM_LABELS[dim]:<20}: Δ={mean_d:+.3f}")
                results[cid]["post_intervention_deltas"] = {
                    dim: float(np.mean(deltas_by_dim[dim])) for dim in DIMENSIONS if deltas_by_dim[dim]
                }

    # Topic type breakdown
    print(f"\n--- Topic type breakdown (empirical vs values-driven) ---")
    for cid in strategy_ids:
        cond = df[df["condition_id"] == cid]
        for tt in ["empirical", "values"]:
            subset = cond[cond["topic_type"] == tt]
            means = {dim: float(subset[dim].mean()) for dim in DIMENSIONS}
            results[cid][f"topic_type_{tt}"] = means
        print(f"  {CONDITION_LABELS[cid]}:")
        for dim in DIMENSIONS:
            emp = results[cid]["topic_type_empirical"][dim]
            val = results[cid]["topic_type_values"][dim]
            diff = val - emp
            marker = "▲" if diff > 0.1 else "▼" if diff < -0.1 else "≈"
            print(f"    {DIM_LABELS[dim]:<20}: emp={emp:.3f} val={val:.3f} ({diff:+.3f}) {marker}")

    return results


# ─── RQ3: Temporal quality trajectories ───────────────────────────────────────

def rq3_analysis(df: pd.DataFrame) -> dict:
    """Per-condition per-round mean scores for temporal trajectory analysis."""
    print("\n" + "=" * 70)
    print("RQ3: TEMPORAL QUALITY TRAJECTORIES")
    print("=" * 70)

    results = {}
    max_round = int(df["round_number"].max())

    for cid in CONDITION_ORDER:
        cond = df[df["condition_id"] == cid]
        trajectories = {}

        for dim in DIMENSIONS:
            traj = []
            for r in range(1, max_round + 1):
                round_data = cond[cond["round_number"] == r]
                if not round_data.empty:
                    traj.append(float(round_data[dim].mean()))
                else:
                    traj.append(None)
            trajectories[dim] = traj

        # Composite trajectory
        comp_traj = []
        for r in range(1, max_round + 1):
            round_data = cond[cond["round_number"] == r]
            if not round_data.empty:
                comp_traj.append(float(round_data["composite"].mean()))
            else:
                comp_traj.append(None)
        trajectories["composite"] = comp_traj

        results[cid] = trajectories

        # Print summary: first round, last round, trend
        print(f"\n  {CONDITION_LABELS[cid]}:")
        for dim in DIMENSIONS:
            t = [x for x in trajectories[dim] if x is not None]
            if len(t) >= 2:
                trend = t[-1] - t[0]
                direction = "↑" if trend > 0.1 else "↓" if trend < -0.1 else "→"
                print(f"    {DIM_LABELS[dim]:<20}: R1={t[0]:.3f} → R{len(t)}={t[-1]:.3f} ({trend:+.3f}) {direction}")

    return results


# ─── Causal Analysis: Silent Control ─────────────────────────────────────────

def causal_analysis(df: pd.DataFrame, interventions_df: pd.DataFrame) -> dict:
    """Compare post-trigger trajectories: intervention vs silent control."""
    print("\n" + "=" * 70)
    print("CAUSAL ANALYSIS: INTERVENTION vs SILENT CONTROL")
    print("=" * 70)

    if interventions_df.empty:
        print("  No intervention data available.")
        return {}

    results = {}
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]

    for cid in strategy_ids:
        cond_ivs = interventions_df[interventions_df["condition_id"] == cid]
        active = cond_ivs[~cond_ivs["silent_control"]]
        silent = cond_ivs[cond_ivs["silent_control"]]

        if active.empty or silent.empty:
            print(f"  {CONDITION_LABELS[cid]}: insufficient data (active={len(active)}, silent={len(silent)})")
            continue

        def get_deltas(iv_subset):
            deltas = defaultdict(list)
            for _, iv in iv_subset.iterrows():
                run_data = df[df["run_id"] == iv["run_id"]]
                pre = run_data[run_data["round_number"] == iv["round_number"]]
                post = run_data[run_data["round_number"] == iv["round_number"] + 1]
                if pre.empty or post.empty:
                    continue
                for dim in DIMENSIONS:
                    deltas[dim].append(post[dim].mean() - pre[dim].mean())
            return deltas

        active_deltas = get_deltas(active)
        silent_deltas = get_deltas(silent)

        print(f"\n  {CONDITION_LABELS[cid]} (active={len(active)}, silent={len(silent)}):")
        cid_results = {}
        for dim in DIMENSIONS:
            a_vals = active_deltas.get(dim, [])
            s_vals = silent_deltas.get(dim, [])
            if len(a_vals) >= 2 and len(s_vals) >= 2:
                u, p = stats.mannwhitneyu(a_vals, s_vals, alternative="two-sided")
                sig = "*" if p < 0.05 else ""
                print(f"    {DIM_LABELS[dim]:<20}: active Δ={np.mean(a_vals):+.3f}, silent Δ={np.mean(s_vals):+.3f}, p={p:.4f}{sig}")
                cid_results[dim] = {
                    "active_delta": float(np.mean(a_vals)),
                    "silent_delta": float(np.mean(s_vals)),
                    "p": float(p),
                    "significant": p < 0.05,
                }
            elif a_vals or s_vals:
                am = np.mean(a_vals) if a_vals else float("nan")
                sm = np.mean(s_vals) if s_vals else float("nan")
                print(f"    {DIM_LABELS[dim]:<20}: active Δ={am:+.3f} (n={len(a_vals)}), silent Δ={sm:+.3f} (n={len(s_vals)}) — too few for test")
                cid_results[dim] = {
                    "active_delta": float(am) if a_vals else None,
                    "silent_delta": float(sm) if s_vals else None,
                    "p": None,
                    "significant": None,
                }

        results[cid] = cid_results

    return results


# ─── RQ5: Topic Type Interaction ──────────────────────────────────────────────

def rq5_analysis(df: pd.DataFrame) -> dict:
    """Test if moderation effects differ between empirical and values-driven topics."""
    print("\n" + "=" * 70)
    print("RQ5: TOPIC TYPE INTERACTION ANALYSIS")
    print("=" * 70)

    results = {}

    # For each dimension: compare (strategy - baseline_1) gap between empirical and values topics
    baseline = df[df["condition_id"] == "baseline_1"]
    strategies = df[df["is_strategy"]]

    print(f"\n{'Dimension':<20} {'Emp Δ':>8} {'Val Δ':>8} {'Interaction':>12} {'p':>10}")
    print("-" * 65)

    for dim in DIMENSIONS:
        # Moderation effect for empirical topics
        bl_emp = baseline[baseline["topic_type"] == "empirical"][dim].dropna()
        st_emp = strategies[strategies["topic_type"] == "empirical"][dim].dropna()
        delta_emp = float(st_emp.mean() - bl_emp.mean())

        # Moderation effect for values topics
        bl_val = baseline[baseline["topic_type"] == "values"][dim].dropna()
        st_val = strategies[strategies["topic_type"] == "values"][dim].dropna()
        delta_val = float(st_val.mean() - bl_val.mean())

        # Test interaction via 2-way comparison
        # Strategy scores for empirical vs values, controlling for baseline
        # Simpler: compare strategy-baseline deltas
        # More robust: Mann-Whitney on strategy scores between topic types
        u, p = stats.mannwhitneyu(st_emp, st_val, alternative="two-sided")
        interaction = delta_val - delta_emp
        sig = "*" if p < 0.05 else ""

        results[dim] = {
            "delta_empirical": delta_emp,
            "delta_values": delta_val,
            "interaction": float(interaction),
            "p": float(p),
            "significant": p < 0.05,
        }
        print(f"{DIM_LABELS[dim]:<20} {delta_emp:>+8.3f} {delta_val:>+8.3f} {interaction:>+12.3f} {p:>10.4f}{sig}")

    return results


# ─── Ablation Analysis ───────────────────────────────────────────────────────

def ablation_analysis(df: pd.DataFrame, ablation_df: pd.DataFrame) -> dict:
    """Compare normal vs reversed turn order."""
    print("\n" + "=" * 70)
    print("ABLATION: SPEAKING-ORDER EFFECT")
    print("=" * 70)

    if ablation_df.empty:
        print("  No ablation data available.")
        return {}

    # Normal order: baseline_1, mindestlohn from experiment
    normal = df[(df["condition_id"] == "baseline_1") & (df["topic_id"] == "mindestlohn")]
    reversed_order = ablation_df

    results = {}
    print(f"\n{'Dimension':<20} {'Normal μ':>10} {'Reversed μ':>12} {'Δ':>8} {'U':>10} {'p':>10}")
    print("-" * 75)

    for dim in DIMENSIONS:
        n_vals = normal[dim].dropna()
        r_vals = reversed_order[dim].dropna()
        u, p = stats.mannwhitneyu(n_vals, r_vals, alternative="two-sided")
        delta = float(r_vals.mean() - n_vals.mean())
        sig = "*" if p < 0.05 else ""

        results[dim] = {
            "normal_mean": float(n_vals.mean()),
            "reversed_mean": float(r_vals.mean()),
            "delta": delta,
            "U": float(u),
            "p": float(p),
            "significant": p < 0.05,
        }
        print(f"{DIM_LABELS[dim]:<20} {n_vals.mean():>10.3f} {r_vals.mean():>12.3f} {delta:>+8.3f} {u:>10.0f} {p:>10.4f}{sig}")

    return results


# ─── LaTeX Table Generators ──────────────────────────────────────────────────

def generate_rq1_table(rq1: dict) -> str:
    """RQ1: Moderated vs None baseline table."""
    bl1 = rq1["baseline_1"]
    latex = r"""\begin{table}[htbp]
\centering
\caption{RQ1: Moderated conditions (Strategies A--D) vs.\ No-moderation baseline. Mann-Whitney $U$ test with Cliff's $\delta$ effect size.}
\label{tab:rq1}
\small
\begin{tabular}{l cc cc c}
\toprule
\textbf{Dimension} & $\mu_{\text{Strat}}$ & $\mu_{\text{None}}$ & $U$ & $p$ & $\delta$ \\
\midrule
"""
    for dim in DIMENSIONS:
        r = bl1[dim]
        sig = "$^{*}$" if r["significant"] else ""
        latex += f"{DIM_LABELS[dim]} & {r['strategy_mean']:.3f} & {r['baseline_mean']:.3f} & {r['U']:.0f} & {r['p']:.4f}{sig} & {r['cliff_delta']:+.3f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_rq2_table(rq2: dict) -> str:
    """RQ2: Per-condition mean scores table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{RQ2: Per-condition mean scores across discourse quality dimensions (1--5 scale). Bold = highest per dimension.}
\label{tab:rq2-means}
\small
\begin{tabular}{l """ + "c " * len(DIMENSIONS) + r"""c}
\toprule
\textbf{Condition} """
    for dim in DIMENSIONS:
        latex += f"& \\textbf{{{DIM_LABELS[dim][:8]}}} "
    latex += r"""& \textbf{Comp.} \\
\midrule
"""
    # Find max per dimension
    max_vals = {}
    for dim in DIMENSIONS + ["composite"]:
        max_vals[dim] = max(rq2[cid]["means"][dim] for cid in CONDITION_ORDER)

    for cid in CONDITION_ORDER:
        label = CONDITION_LABELS[cid]
        latex += f"{label} "
        for dim in DIMENSIONS:
            v = rq2[cid]["means"][dim]
            bold = v == max_vals[dim]
            fmt = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
            latex += f"& {fmt} "
        comp = rq2[cid]["means"]["composite"]
        bold = comp == max_vals["composite"]
        fmt = f"\\textbf{{{comp:.2f}}}" if bold else f"{comp:.2f}"
        latex += f"& {fmt} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_ablation_table(abl: dict) -> str:
    """Ablation: speaking-order comparison table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Speaking-order ablation: normal (CDU/CSU$\to$AfD) vs.\ reversed order, Baseline~1, \textit{Mindestlohn}, 5 runs each.}
\label{tab:ablation}
\small
\begin{tabular}{l cc c c}
\toprule
\textbf{Dimension} & $\mu_{\text{Normal}}$ & $\mu_{\text{Reversed}}$ & $\Delta$ & $p$ \\
\midrule
"""
    for dim in DIMENSIONS:
        r = abl[dim]
        sig = "$^{*}$" if r["significant"] else ""
        latex += f"{DIM_LABELS[dim]} & {r['normal_mean']:.3f} & {r['reversed_mean']:.3f} & {r['delta']:+.3f} & {r['p']:.4f}{sig} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_posthoc()
    interventions_df = load_interventions()
    ablation_df = load_ablation()

    print(f"Loaded {len(df)} turn scores, {len(interventions_df)} interventions, {len(ablation_df)} ablation turns")

    # Run analyses
    rq1 = rq1_analysis(df)
    rq2 = rq2_analysis(df, interventions_df)
    rq3 = rq3_analysis(df)
    causal = causal_analysis(df, interventions_df)
    rq5 = rq5_analysis(df)
    abl = ablation_analysis(df, ablation_df)

    # Generate LaTeX tables
    tables = {
        "rq1_table.tex": generate_rq1_table(rq1),
        "rq2_means_table.tex": generate_rq2_table(rq2),
        "ablation_table.tex": generate_ablation_table(abl),
    }

    for fname, content in tables.items():
        (OUTPUT_DIR / fname).write_text(content, encoding="utf-8")
        print(f"\nSaved: {OUTPUT_DIR / fname}")

    # Save all results as JSON
    all_results = {
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "causal": causal,
        "rq5": rq5,
        "ablation": abl,
        "intervention_summary": {
            "total": len(interventions_df),
            "active": int((~interventions_df["silent_control"]).sum()) if not interventions_df.empty else 0,
            "silent": int(interventions_df["silent_control"].sum()) if not interventions_df.empty else 0,
            "per_condition": interventions_df.groupby("condition_id").size().to_dict() if not interventions_df.empty else {},
        }
    }

    with open(OUTPUT_DIR / "comparative_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
