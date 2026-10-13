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
+ Mixed-model sensitivity analysis

The debate (run) is the unit of analysis: p-values come from topic-stratified
permutation tests on per-debate means, confidence intervals from a topic-stratified
cluster bootstrap over debates, and Holm correction is applied across dimensions.
The 5 reversed-order ablation runs are excluded from all analyses except the
ablation, and the opening turn of each debate (responsiveness = 1 by construction)
is excluded from responsiveness unless stated otherwise.

Usage:
    python experiments/comparative_analysis.py

Output:
    experiments/outputs/comparative/  (LaTeX tables, JSON summaries)
"""

import json
import glob
import math
import warnings
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
    "responsiveness_all": "Resp.\ (incl.\ opening)",
}

# Responsiveness is also reported with the opening turn included (sensitivity check).
RQ1_METRICS = DIMENSIONS + ["responsiveness_all"]

ABLATION_MARKER = "ablation_turn_order"
N_PERM = 10_000
N_BOOT = 2_000
SEED = 2026

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
    """Load all post-hoc scores, including the ablation runs (flagged)."""
    df = pd.read_csv(POSTHOC_CSV)
    df["topic_type"] = df["topic_id"].map(TOPIC_TYPES)
    df["condition_label"] = df["condition_id"].map(CONDITION_LABELS)
    df["is_strategy"] = df["condition_id"].str.startswith("strategy_")
    df["is_moderated"] = df["condition_id"] != "baseline_1"
    df["is_ablation"] = df["run_file"].str.contains(ABLATION_MARKER, regex=False)
    # The first speaker of a debate has nothing to respond to and always scores 1.
    df["is_opening_turn"] = (df["round_number"] == 1) & (df["turn_in_round"] == 1)
    df["responsiveness_all"] = df["responsiveness"].astype(float)
    df["responsiveness"] = df["responsiveness"].astype(float).where(~df["is_opening_turn"])
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


# ─── Effect Size ──────────────────────────────────────────────────────────────

def cliff_delta(x, y):
    """Cliff's delta effect size for ordinal data: P(X > Y) - P(X < Y)."""
    x, y = np.asarray(x, dtype=float), np.sort(np.asarray(y, dtype=float))
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = np.searchsorted(y, x, side="left").sum()
    less = (n_y - np.searchsorted(y, x, side="right")).sum()
    return float((more - less) / (n_x * n_y))


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


# ─── Debate-level inference helpers ──────────────────────────────────────────

def holm(pvals) -> list:
    """Holm-Bonferroni adjusted p-values (None entries are passed through)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    adjusted = [None] * len(pvals)
    running = 0.0
    m = len(idx)
    for rank, i in enumerate(sorted(idx, key=lambda k: pvals[k])):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def run_table(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """Per-debate means plus the per-debate turn scores for one metric."""
    sub = df.dropna(subset=[metric])
    runs = sub.groupby("run_id").agg(
        condition_id=("condition_id", "first"),
        topic_id=("topic_id", "first"),
        value=(metric, "mean"),
    )
    turns = {rid: g[metric].to_numpy() for rid, g in sub.groupby("run_id")}
    return runs, turns


def stratified_perm_p(values, in_a, strata, observed, n_perm=N_PERM, seed=SEED) -> float:
    """Two-sided permutation p for a difference in means; labels shuffled within strata."""
    rng = np.random.default_rng(seed)
    labels = np.empty((n_perm, len(values)), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        labels[:, idx] = rng.permuted(np.tile(in_a[idx], (n_perm, 1)), axis=1)
    a_mean = (labels * values).sum(1) / labels.sum(1)
    b_mean = (~labels * values).sum(1) / (~labels).sum(1)
    extreme = np.abs(a_mean - b_mean) >= abs(observed) - 1e-12
    return float((extreme.sum() + 1) / (n_perm + 1))


def exact_perm_p(a, b, max_enum=50_000, n_perm=N_PERM, seed=SEED) -> float:
    """Two-sided permutation p for a difference in means between two small samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(a)
    observed = abs(a.mean() - b.mean())
    total = pooled.sum()
    if math.comb(n, k) <= max_enum:
        diffs = []
        for combo in combinations(range(n), k):
            s = pooled[list(combo)].sum()
            diffs.append(abs(s / k - (total - s) / (n - k)))
        diffs = np.asarray(diffs)
        return float(np.mean(diffs >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    perm = np.array([rng.permutation(pooled) for _ in range(n_perm)])
    diffs = np.abs(perm[:, :k].mean(1) - perm[:, k:].mean(1))
    return float(((diffs >= observed - 1e-12).sum() + 1) / (n_perm + 1))


def cluster_bootstrap(runs: pd.DataFrame, turns: dict, n_boot=N_BOOT, seed=SEED) -> dict:
    """95% CIs for the mean difference and Cliff's delta, resampling debates within
    each (group, topic) cell so that turns from one debate stay together."""
    rng = np.random.default_rng(seed)
    cells = [(in_a, g.index.to_numpy())
             for (in_a, _), g in runs.groupby(["in_a", "topic_id"])]
    value = runs["value"]
    diffs, deltas = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        a_ids, b_ids = [], []
        for in_a, ids in cells:
            draw = rng.choice(ids, size=len(ids), replace=True)
            (a_ids if in_a else b_ids).extend(draw)
        diffs[b] = value.loc[a_ids].mean() - value.loc[b_ids].mean()
        deltas[b] = cliff_delta(np.concatenate([turns[r] for r in a_ids]),
                                np.concatenate([turns[r] for r in b_ids]))
    return {
        "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


def compare_groups(df: pd.DataFrame, metric: str, cond_a: list, cond_b: list,
                   bootstrap: bool = True) -> dict:
    """Debate-level comparison of condition group A vs group B on one metric."""
    sub = df[df["condition_id"].isin(cond_a + cond_b)]
    runs, turns = run_table(sub, metric)
    runs["in_a"] = runs["condition_id"].isin(cond_a)
    a_runs, b_runs = runs[runs["in_a"]], runs[~runs["in_a"]]
    observed = float(a_runs["value"].mean() - b_runs["value"].mean())
    a_turns = np.concatenate([turns[r] for r in a_runs.index])
    b_turns = np.concatenate([turns[r] for r in b_runs.index])
    _, p_turn = stats.mannwhitneyu(a_turns, b_turns, alternative="two-sided")
    delta = cliff_delta(a_turns, b_turns)
    result = {
        "a_mean": float(a_runs["value"].mean()),
        "b_mean": float(b_runs["value"].mean()),
        "diff": observed,
        "cliff_delta": delta,
        "magnitude": cliff_magnitude(delta),
        "p_perm": stratified_perm_p(runs["value"].to_numpy(), runs["in_a"].to_numpy(),
                                    runs["topic_id"].to_numpy(), observed),
        "p_turn_mwu": float(p_turn),
        "n_runs_a": int(len(a_runs)),
        "n_runs_b": int(len(b_runs)),
        "n_turns_a": int(len(a_turns)),
        "n_turns_b": int(len(b_turns)),
    }
    if bootstrap:
        result.update(cluster_bootstrap(runs, turns))
    return result


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sn(x: float, nd: int = 2) -> str:
    """Signed number for LaTeX tables, with a proper minus sign and no '-0.00'."""
    s = f"{x:+.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
    return f"${s}$"


def ci(lo: float, hi: float, nd: int = 2) -> str:
    return f"[{sn(lo, nd)}, {sn(hi, nd)}]"


# ─── RQ1: Moderated vs Baselines ─────────────────────────────────────────────

def rq1_analysis(df: pd.DataFrame) -> dict:
    """Pooled strategies vs each baseline, per dimension, at the debate level.

    The comparison with the no-moderation baseline is the confirmatory test
    (Holm across the seven dimensions); comparisons with the other baselines
    are exploratory.
    """
    print("\n" + "=" * 70)
    print("RQ1: MODERATED (Strategies A–D) vs BASELINES  [unit: debate]")
    print("=" * 70)

    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    results = {}

    for baseline_id in ["baseline_1", "baseline_2", "baseline_3", "baseline_4"]:
        bl_label = CONDITION_LABELS[baseline_id]
        print(f"\n--- Strategies vs {bl_label} ---")
        print(f"{'Dimension':<26} {'Strat μ':>8} {'BL μ':>8} {'Δ':>7} {'95% CI':>17} "
              f"{'δ':>7} {'95% CI':>17} {'p_perm':>8} {'p_Holm':>8}")
        print("-" * 115)

        bl_results = {m: compare_groups(df, m, strategy_ids, [baseline_id])
                      for m in RQ1_METRICS}
        adjusted = holm([bl_results[d]["p_perm"] for d in DIMENSIONS])
        for d, p_adj in zip(DIMENSIONS, adjusted):
            bl_results[d]["p_holm"] = p_adj
            bl_results[d]["significant"] = p_adj < 0.05
        bl_results["responsiveness_all"]["p_holm"] = None
        bl_results["responsiveness_all"]["significant"] = None

        for m in RQ1_METRICS:
            r = bl_results[m]
            ph = f"{r['p_holm']:.4f}" if r["p_holm"] is not None else "   —"
            print(f"{DIM_LABELS[m]:<26} {r['a_mean']:>8.3f} {r['b_mean']:>8.3f} {r['diff']:>+7.3f} "
                  f"[{r['diff_ci'][0]:+.3f},{r['diff_ci'][1]:+.3f}] {r['cliff_delta']:>+7.3f} "
                  f"[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}] {r['p_perm']:>8.4f} {ph:>8}")
        results[baseline_id] = bl_results

    print(f"\n--- Kruskal-Wallis on debate means (all 8 conditions) ---")
    kw_results = {}
    for dim in DIMENSIONS:
        runs, _ = run_table(df, dim)
        groups = [runs[runs["condition_id"] == c]["value"].to_numpy() for c in CONDITION_ORDER]
        h_stat, p_val = stats.kruskal(*groups)
        kw_results[dim] = {"H": float(h_stat), "p": float(p_val)}
        print(f"{DIM_LABELS[dim]:<20} H={h_stat:>8.2f} p={p_val:.4f}")

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

    # Each strategy vs no moderation (20 vs 20 debates), Holm across all 28 tests
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    print(f"\n--- Each strategy vs None [unit: debate; Holm across 4 x 7 tests] ---")
    vs_none = {cid: {dim: compare_groups(df, dim, [cid], ["baseline_1"]) for dim in DIMENSIONS}
               for cid in strategy_ids}
    keys = [(cid, dim) for cid in strategy_ids for dim in DIMENSIONS]
    adjusted = holm([vs_none[c][d]["p_perm"] for c, d in keys])
    for (cid, dim), p_adj in zip(keys, adjusted):
        vs_none[cid][dim]["p_holm"] = p_adj
        vs_none[cid][dim]["significant"] = p_adj < 0.05
    for cid in strategy_ids:
        results[cid]["vs_none"] = vs_none[cid]
        print(f"\n  {CONDITION_LABELS[cid]}:")
        for dim in DIMENSIONS:
            r = vs_none[cid][dim]
            print(f"    {DIM_LABELS[dim]:<20}: Δ={r['diff']:+.3f} [{r['diff_ci'][0]:+.3f},{r['diff_ci'][1]:+.3f}] "
                  f"δ={r['cliff_delta']:+.3f} p_perm={r['p_perm']:.4f} p_Holm={r['p_holm']:.4f}")

    # Pairwise strategy comparisons (exploratory, debate-level permutation test)
    print(f"\n--- Pairwise strategy comparisons [exploratory; raw p < 0.05 shown] ---")
    pairwise = {}
    for dim in DIMENSIONS:
        for s1, s2 in combinations(strategy_ids, 2):
            r = compare_groups(df, dim, [s1], [s2], bootstrap=False)
            pairwise[f"{s1}|{s2}|{dim}"] = r
            if r["p_perm"] < 0.05:
                print(f"    {DIM_LABELS[dim]:<20} {CONDITION_LABELS[s1]:>15} vs {CONDITION_LABELS[s2]:<15}: "
                      f"Δ={r['diff']:+.3f} δ={r['cliff_delta']:+.3f} p_perm={r['p_perm']:.4f}")
    results["pairwise_strategies"] = pairwise

    # Post-intervention quality deltas
    if not interventions_df.empty:
        print(f"\n--- Post-intervention quality deltas (next round - trigger round) ---")
        active_ivs = interventions_df[~interventions_df["silent_control"]]
        strategy_ivs = active_ivs[active_ivs["condition_id"].str.startswith(
            "strategy_")]

        for cid in strategy_ids:
            cond_ivs = strategy_ivs[strategy_ivs["condition_id"] == cid]
            if cond_ivs.empty:
                continue

            deltas_by_dim = defaultdict(list)
            for _, iv in cond_ivs.iterrows():
                run_data = df[(df["run_id"] == iv["run_id"])]
                pre = run_data[run_data["round_number"] == iv["round_number"]]
                post = run_data[run_data["round_number"]
                                == iv["round_number"] + 1]
                if pre.empty or post.empty:
                    continue
                for dim in DIMENSIONS:
                    d = post[dim].mean() - pre[dim].mean()
                    deltas_by_dim[dim].append(d)

            if deltas_by_dim:
                print(
                    f"\n  {CONDITION_LABELS[cid]} (n={len(next(iter(deltas_by_dim.values())))} interventions):")
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
            print(
                f"    {DIM_LABELS[dim]:<20}: emp={emp:.3f} val={val:.3f} ({diff:+.3f}) {marker}")

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
                print(
                    f"    {DIM_LABELS[dim]:<20}: R1={t[0]:.3f} → R{len(t)}={t[-1]:.3f} ({trend:+.3f}) {direction}")

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
            print(
                f"  {CONDITION_LABELS[cid]}: insufficient data (active={len(active)}, silent={len(silent)})")
            continue

        def get_deltas(iv_subset):
            deltas = defaultdict(list)
            for _, iv in iv_subset.iterrows():
                run_data = df[df["run_id"] == iv["run_id"]]
                pre = run_data[run_data["round_number"] == iv["round_number"]]
                post = run_data[run_data["round_number"]
                                == iv["round_number"] + 1]
                if pre.empty or post.empty:
                    continue
                for dim in DIMENSIONS:
                    deltas[dim].append(post[dim].mean() - pre[dim].mean())
            return deltas

        active_deltas = get_deltas(active)
        silent_deltas = get_deltas(silent)

        print(
            f"\n  {CONDITION_LABELS[cid]} (active={len(active)}, silent={len(silent)}):")
        cid_results = {"n_active": int(len(active)), "n_silent": int(len(silent))}
        for dim in DIMENSIONS:
            a_vals = active_deltas.get(dim, [])
            s_vals = silent_deltas.get(dim, [])
            if len(a_vals) >= 2 and len(s_vals) >= 2:
                _, p_mwu = stats.mannwhitneyu(
                    a_vals, s_vals, alternative="two-sided")
                p_exact = exact_perm_p(a_vals, s_vals)
                cid_results[dim] = {
                    "active_delta": float(np.mean(a_vals)),
                    "silent_delta": float(np.mean(s_vals)),
                    "n_active": len(a_vals),
                    "n_silent": len(s_vals),
                    "p": float(p_mwu),
                    "p_perm": p_exact,
                }
            elif a_vals or s_vals:
                am = np.mean(a_vals) if a_vals else float("nan")
                sm = np.mean(s_vals) if s_vals else float("nan")
                cid_results[dim] = {
                    "active_delta": float(am) if a_vals else None,
                    "silent_delta": float(sm) if s_vals else None,
                    "n_active": len(a_vals),
                    "n_silent": len(s_vals),
                    "p": None,
                    "p_perm": None,
                }

        tested = [d for d in DIMENSIONS if d in cid_results and cid_results[d]["p"] is not None]
        for d, p_adj in zip(tested, holm([cid_results[d]["p"] for d in tested])):
            cid_results[d]["p_holm_within"] = p_adj
        for dim in DIMENSIONS:
            r = cid_results.get(dim)
            if r is None:
                continue
            if r["p"] is not None:
                sig = "*" if r["p"] < 0.05 else ""
                print(f"    {DIM_LABELS[dim]:<20}: active Δ={r['active_delta']:+.3f}, silent Δ={r['silent_delta']:+.3f}, "
                      f"p_MWU={r['p']:.4f}{sig} p_perm={r['p_perm']:.4f} p_Holm={r['p_holm_within']:.4f}")
            else:
                print(f"    {DIM_LABELS[dim]:<20}: too few events for a test")

        results[cid] = cid_results

    all_tests = [(c, d) for c in results for d in DIMENSIONS
                 if d in results[c] and results[c][d]["p"] is not None]
    for (c, d), p_adj in zip(all_tests, holm([results[c][d]["p"] for c, d in all_tests])):
        results[c][d]["p_holm_all"] = p_adj
    n_sig = sum(results[c][d]["p"] < 0.05 for c, d in all_tests)
    results["summary"] = {
        "test": "two-sided Mann-Whitney U on per-event change in the debate's round mean "
                "(next round minus trigger round); exact permutation p as a check",
        "n_tests": len(all_tests),
        "expected_false_positives": 0.05 * len(all_tests),
        "n_raw_significant": int(n_sig),
        "n_holm_significant": int(sum(results[c][d]["p_holm_all"] < 0.05 for c, d in all_tests)),
    }
    print(f"\n  {len(all_tests)} tests, {n_sig} with raw p < 0.05 "
          f"(≈{0.05 * len(all_tests):.1f} expected by chance); "
          f"{results['summary']['n_holm_significant']} survive Holm across all tests.")

    return results


# ─── RQ5: Topic Type Interaction ──────────────────────────────────────────────

def interaction_permutation_p(df: pd.DataFrame, dim: str, observed: float,
                              n_perm: int = 10_000, seed: int = 2026) -> float:
    """Two-sided permutation p for (strategy - none) x (values - empirical).

    Debate runs are the exchangeable unit: topic-type labels are shuffled among
    runs within the strategy group and within the no-moderation group.
    """
    sub = df[(df["condition_id"] == "baseline_1") |
             df["is_strategy"]].dropna(subset=[dim])
    runs = sub.groupby("run_id").agg(
        is_strategy=("is_strategy", "first"),
        is_values=("topic_type", lambda s: s.iloc[0] == "values"),
        total=(dim, "sum"),
        n=(dim, "size"),
    )
    rng = np.random.default_rng(seed)

    def group_delta(g: pd.DataFrame) -> np.ndarray:
        total, n = g["total"].to_numpy(), g["n"].to_numpy()
        labels = np.tile(g["is_values"].to_numpy(), (n_perm, 1))
        labels = rng.permuted(labels, axis=1)
        val_mean = (labels * total).sum(1) / (labels * n).sum(1)
        emp_mean = (~labels * total).sum(1) / (~labels * n).sum(1)
        return val_mean - emp_mean

    perm = group_delta(runs[runs["is_strategy"]]) - \
        group_delta(runs[~runs["is_strategy"]])
    return float((np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (n_perm + 1))


def rq5_analysis(df: pd.DataFrame) -> dict:
    """Test if moderation effects differ between empirical and values-driven topics."""
    print("\n" + "=" * 70)
    print("RQ5: TOPIC TYPE INTERACTION ANALYSIS")
    print("=" * 70)

    results = {}

    # For each dimension: compare (strategy - baseline_1) gap between empirical and values topics
    baseline = df[df["condition_id"] == "baseline_1"]
    strategies = df[df["is_strategy"]]

    print(
        f"\n{'Dimension':<20} {'Emp Δ':>8} {'Val Δ':>8} {'Interaction':>12} {'p':>10}")
    print("-" * 65)

    for dim in DIMENSIONS:
        # Moderation effect for empirical topics
        bl_emp = baseline[baseline["topic_type"] == "empirical"][dim].dropna()
        st_emp = strategies[strategies["topic_type"]
                            == "empirical"][dim].dropna()
        delta_emp = float(st_emp.mean() - bl_emp.mean())

        # Moderation effect for values topics
        bl_val = baseline[baseline["topic_type"] == "values"][dim].dropna()
        st_val = strategies[strategies["topic_type"] == "values"][dim].dropna()
        delta_val = float(st_val.mean() - bl_val.mean())

        # Topic-type difference in strategy scores only (not an interaction test)
        _, p_topic = stats.mannwhitneyu(
            st_emp, st_val, alternative="two-sided")
        interaction = delta_val - delta_emp
        p = interaction_permutation_p(df, dim, interaction)
        sig = "*" if p < 0.05 else ""

        results[dim] = {
            "delta_empirical": delta_emp,
            "delta_values": delta_val,
            "interaction": float(interaction),
            "p": float(p),
            "p_topic_mwu": float(p_topic),
            "significant": p < 0.05,
        }
        print(
            f"{DIM_LABELS[dim]:<20} {delta_emp:>+8.3f} {delta_val:>+8.3f} {interaction:>+12.3f} {p:>10.4f}{sig}")

    for dim, p_adj in zip(DIMENSIONS, holm([results[d]["p"] for d in DIMENSIONS])):
        results[dim]["p_holm"] = p_adj

    return results


# ─── Ablation Analysis ───────────────────────────────────────────────────────

def ablation_analysis(full_df: pd.DataFrame) -> dict:
    """Normal vs reversed speaking order: 5 vs 5 Mindestlohn debates (no pooling)."""
    print("\n" + "=" * 70)
    print("ABLATION: SPEAKING-ORDER EFFECT  [unit: debate; exact permutation]")
    print("=" * 70)

    b1_mindestlohn = full_df[(full_df["condition_id"] == "baseline_1")
                             & (full_df["topic_id"] == "mindestlohn")]
    if not b1_mindestlohn["is_ablation"].any():
        print("  No ablation data available.")
        return {}

    rng = np.random.default_rng(SEED)
    results = {}
    print(f"\n{'Dimension':<20} {'Normal μ':>10} {'Reversed μ':>12} {'Δ':>8} {'95% CI':>18} {'p':>8}")
    print("-" * 80)

    for dim in DIMENSIONS:
        runs = b1_mindestlohn.groupby("run_id").agg(
            value=(dim, "mean"), is_ablation=("is_ablation", "first"))
        normal = runs[~runs["is_ablation"]]["value"].to_numpy()
        reversed_order = runs[runs["is_ablation"]]["value"].to_numpy()
        delta = float(reversed_order.mean() - normal.mean())
        boot = [rng.choice(reversed_order, len(reversed_order)).mean()
                - rng.choice(normal, len(normal)).mean() for _ in range(N_BOOT)]
        p = exact_perm_p(reversed_order, normal)
        results[dim] = {
            "normal_mean": float(normal.mean()),
            "reversed_mean": float(reversed_order.mean()),
            "delta": delta,
            "delta_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": p,
            "n_normal": int(len(normal)),
            "n_reversed": int(len(reversed_order)),
        }
        r = results[dim]
        print(f"{DIM_LABELS[dim]:<20} {r['normal_mean']:>10.3f} {r['reversed_mean']:>12.3f} {delta:>+8.3f} "
              f"[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}] {p:>8.4f}")

    for dim, p_adj in zip(DIMENSIONS, holm([results[d]["p"] for d in DIMENSIONS])):
        results[dim]["p_holm"] = p_adj
    return results


# ─── Mixed-model sensitivity analysis ─────────────────────────────────────────

def mixed_model_analysis(df: pd.DataFrame) -> dict:
    """Linear mixed models on turn scores with a random intercept per debate and
    fixed effects for topic and agent (too few levels to estimate their variances).

    Model 1 (RQ1): score ~ moderated + topic + agent, strategies and None only.
    Model 2 (RQ2): score ~ condition + topic + agent, each strategy vs None.
    """
    import statsmodels.formula.api as smf

    print("\n" + "=" * 70)
    print("SENSITIVITY: LINEAR MIXED MODELS (random intercept per debate)")
    print("=" * 70)

    sub = df[df["is_strategy"] | (df["condition_id"] == "baseline_1")].copy()
    sub["moderated"] = sub["is_strategy"].astype(int)
    sub["agent"] = sub["agent_name"]
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    results = {"pooled": {}, "per_strategy": {}}

    for dim in DIMENSIONS:
        data = sub.dropna(subset=[dim])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit1 = smf.mixedlm(f"{dim} ~ moderated + C(topic_id) + C(agent)",
                               data, groups=data["run_id"]).fit(reml=True)
            fit2 = smf.mixedlm(f"{dim} ~ C(condition_id, Treatment('baseline_1')) + C(topic_id) + C(agent)",
                               data, groups=data["run_id"]).fit(reml=True)
        ci1 = fit1.conf_int().loc["moderated"]
        results["pooled"][dim] = {
            "beta": float(fit1.params["moderated"]),
            "ci": [float(ci1[0]), float(ci1[1])],
            "p": float(fit1.pvalues["moderated"]),
            "converged": bool(fit1.converged),
        }
        results["per_strategy"][dim] = {}
        for cid in strategy_ids:
            term = f"C(condition_id, Treatment('baseline_1'))[T.{cid}]"
            ci2 = fit2.conf_int().loc[term]
            results["per_strategy"][dim][cid] = {
                "beta": float(fit2.params[term]),
                "ci": [float(ci2[0]), float(ci2[1])],
                "p": float(fit2.pvalues[term]),
            }

    for dim, p_adj in zip(DIMENSIONS, holm([results["pooled"][d]["p"] for d in DIMENSIONS])):
        results["pooled"][dim]["p_holm"] = p_adj
    keys = [(d, c) for d in DIMENSIONS for c in strategy_ids]
    for (d, c), p_adj in zip(keys, holm([results["per_strategy"][d][c]["p"] for d, c in keys])):
        results["per_strategy"][d][c]["p_holm"] = p_adj

    print(f"\n{'Dimension':<20} {'β (mod.)':>9} {'95% CI':>18} {'p':>9} {'p_Holm':>8}")
    for dim in DIMENSIONS:
        r = results["pooled"][dim]
        print(f"{DIM_LABELS[dim]:<20} {r['beta']:>+9.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
              f"{r['p']:>9.4f} {r['p_holm']:>8.4f}")
    return results


# ─── LaTeX Table Generators ──────────────────────────────────────────────────

def generate_rq1_table(rq1: dict) -> str:
    """RQ1: Pooled strategies vs no-moderation baseline, debate-level inference."""
    bl1 = rq1["baseline_1"]
    n_a, n_b = bl1["civility"]["n_runs_a"], bl1["civility"]["n_runs_b"]
    latex = r"""\begin{table}[htbp]
\centering
\caption[RQ1: moderated conditions vs.\ no moderation]{RQ1: Moderated conditions (Strategies A--D, """ + str(n_a) + r""" debates) vs.\ no moderation (""" + str(n_b) + r""" debates). $\Delta$ = difference in mean score; $\delta$ = Cliff's $\delta$ on turn scores. 95\% CIs from a cluster bootstrap over debates (stratified by topic, 2,000 resamples); $p$ from a permutation test on debate means with labels shuffled within topic (10,000 permutations), Holm-adjusted across the seven dimensions. Responsiveness excludes the opening turn of each debate; the last row includes it.}
\label{tab:rq1}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l cc c c c c}
\toprule
\textbf{Dimension} & $\mu_{\text{Strat}}$ & $\mu_{\text{None}}$ & $\Delta$ [95\% CI] & $\delta$ [95\% CI] & $p$ & $p_{\text{Holm}}$ \\
\midrule
"""
    for m in RQ1_METRICS:
        r = bl1[m]
        if m == "responsiveness_all":
            latex += r"\midrule" + "\n"
        ph = fmt_p(r["p_holm"]) if r["p_holm"] is not None else "---"
        if r["p_holm"] is not None and r["p_holm"] < 0.05:
            ph = f"\\textbf{{{ph}}}"
        latex += (f"{DIM_LABELS[m]} & {r['a_mean']:.2f} & {r['b_mean']:.2f} & "
                  f"{sn(r['diff'])} {ci(*r['diff_ci'])} & "
                  f"{sn(r['cliff_delta'], 3)} {ci(*r['delta_ci'])} & "
                  f"{fmt_p(r['p_perm'])} & {ph} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_rq2_tests_table(rq2: dict) -> str:
    """RQ2: Each strategy vs no moderation, debate level, Holm across 28 tests."""
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_")]
    latex = r"""\begin{table}[htbp]
\centering
\caption[RQ2: each strategy vs.\ no moderation]{RQ2: Difference in mean score between each strategy and no moderation (20 vs.\ 20 debates), with 95\% cluster-bootstrap CIs. $p$-values from topic-stratified permutation tests on debate means, Holm-adjusted across all 28 tests; bold marks $p_{\text{Holm}} < 0.05$, $^{\dagger}$ marks raw $p < 0.05$ that does not survive correction.}
\label{tab:rq2-tests}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l """ + "c " * len(strategy_ids) + r"""}
\toprule
\textbf{Dimension} """ + "".join(f"& \\textbf{{{CONDITION_LABELS[c]}}} " for c in strategy_ids) + r"""\\
\midrule
"""
    for dim in DIMENSIONS:
        latex += f"{DIM_LABELS[dim]} "
        for cid in strategy_ids:
            r = rq2[cid]["vs_none"][dim]
            cell = f"{sn(r['diff'])} {ci(*r['diff_ci'])}"
            if r["p_holm"] < 0.05:
                cell = f"{{\\bfseries\\boldmath {cell}}}"
            elif r["p_perm"] < 0.05:
                cell += r"$^{\dagger}$"
            latex += f"& {cell} "
        latex += "\\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_rq2_table(rq2: dict) -> str:
    """RQ2: Per-condition mean scores table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-condition mean discourse quality scores (1--5 scale; 20 debates per condition). Bold indicates the highest score per dimension across all conditions. Responsiveness excludes the opening turn of each debate; the composite is computed over all turns.}
\label{tab:rq2-means}
\small
\begin{tabular}{l """ + "c" * len(DIMENSIONS) + r"""c}
\toprule
\textbf{Condition} """
    short = {"civility": "Civ.", "relevance": "Rel.", "logical_consistency": "Log.C.",
             "argument_strength": "Arg.S.", "document_grounding": "Doc.G.",
             "responsiveness": "Resp.", "stance_differentiation": "Sta.D."}
    for dim in DIMENSIONS:
        latex += f"& \\textbf{{{short.get(dim, DIM_LABELS[dim])}}} "
    latex += r"""& \textbf{Comp.} \\
\midrule
"""
    # Find max per dimension
    max_vals = {}
    for dim in DIMENSIONS + ["composite"]:
        max_vals[dim] = max(rq2[cid]["means"][dim] for cid in CONDITION_ORDER)

    for cid in CONDITION_ORDER:
        if cid == "strategy_a":
            latex += "\\midrule\n"
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
    """Ablation: speaking-order comparison table (debate level, no pooling)."""
    latex = r"""\begin{table}[htbp]
\centering
\caption[Speaking-order ablation]{Speaking-order ablation: normal (CDU/CSU$\to$AfD) vs.\ reversed order, no moderation, \textit{Mindestlohn}, 5 debates each. $\Delta$ = reversed $-$ normal, with 95\% bootstrap CI over debates; $p$ from an exact permutation test on debate means (252 splits), unadjusted. Responsiveness excludes the opening turn.}
\label{tab:ablation}
\small
\begin{tabular}{l cc c c}
\toprule
\textbf{Dimension} & $\mu_{\text{Normal}}$ & $\mu_{\text{Reversed}}$ & $\Delta$ [95\% CI] & $p$ \\
\midrule
"""
    for dim in DIMENSIONS:
        r = abl[dim]
        latex += (f"{DIM_LABELS[dim]} & {r['normal_mean']:.3f} & {r['reversed_mean']:.3f} & "
                  f"{sn(r['delta'], 3)} {ci(*r['delta_ci'])} & {r['p']:.3f} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_causal_table(causal: dict) -> str:
    """Silent-control comparison: all strategy x dimension tests (exploratory)."""
    strategy_ids = [c for c in CONDITION_ORDER if c.startswith("strategy_") and c in causal]
    s = causal["summary"]
    latex = r"""\begin{table}[htbp]
\centering
\caption[Silent-control comparison (exploratory)]{Silent-control comparison (exploratory). For each trigger event, $\Delta$ is the change in the debate's mean score from the trigger round to the next round. $p$: two-sided Mann--Whitney $U$ test of active vs.\ silent events; $p_{\text{perm}}$: exact permutation test on the difference in means; $p_{\text{Holm}}$: Holm-adjusted across all """ + str(s["n_tests"]) + r""" tests. About """ + f"{s['expected_false_positives']:.1f}" + r""" raw $p < 0.05$ are expected by chance; """ + str(s["n_raw_significant"]) + r""" were observed.}
\label{tab:causal}
\scriptsize
\begin{tabular}{l c l cc ccc}
\toprule
\textbf{Strategy} & $n_{\text{act}}/n_{\text{sil}}$ & \textbf{Dimension} & $\Delta_{\text{active}}$ & $\Delta_{\text{silent}}$ & $p$ & $p_{\text{perm}}$ & $p_{\text{Holm}}$ \\
\midrule
"""
    for cid in strategy_ids:
        c = causal[cid]
        first = True
        for dim in DIMENSIONS:
            r = c.get(dim)
            if r is None or r["p"] is None:
                continue
            label = CONDITION_LABELS[cid] if first else ""
            counts = f"{c['n_active']}/{c['n_silent']}" if first else ""
            first = False
            latex += (f"{label} & {counts} & {DIM_LABELS[dim]} & {sn(r['active_delta'])} & {sn(r['silent_delta'])} & "
                      f"{r['p']:.3f} & {r['p_perm']:.3f} & {r['p_holm_all']:.2f} \\\\\n")
        if cid != strategy_ids[-1]:
            latex += r"\midrule" + "\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


def generate_mixed_table(mixed: dict) -> str:
    """Mixed-model sensitivity table: pooled moderation effect per dimension."""
    latex = r"""\begin{table}[htbp]
\centering
\caption[Mixed-model sensitivity analysis]{Sensitivity analysis: linear mixed models of turn scores with a random intercept per debate and fixed effects for topic and agent. $\beta$ = estimated effect of moderation (Strategies A--D vs.\ no moderation) on the 1--5 scale; $p$ Holm-adjusted across the seven dimensions.}
\label{tab:mixed}
\small
\begin{tabular}{l c c c}
\toprule
\textbf{Dimension} & $\beta$ [95\% CI] & $p$ & $p_{\text{Holm}}$ \\
\midrule
"""
    for dim in DIMENSIONS:
        r = mixed["pooled"][dim]
        latex += (f"{DIM_LABELS[dim]} & {sn(r['beta'], 3)} {ci(*r['ci'])} & "
                  f"{fmt_p(r['p'])} & {fmt_p(r['p_holm'])} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    return latex


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    full_df = load_posthoc()
    df = full_df[~full_df["is_ablation"]].copy()
    interventions_df = load_interventions()

    print(f"Loaded {len(full_df)} turn scores ({len(df)} main-experiment turns in "
          f"{df['run_id'].nunique()} debates; {int(full_df['is_ablation'].sum())} ablation turns), "
          f"{len(interventions_df)} interventions")

    # Run analyses
    rq1 = rq1_analysis(df)
    rq2 = rq2_analysis(df, interventions_df)
    rq3 = rq3_analysis(df)
    causal = causal_analysis(df, interventions_df)
    rq5 = rq5_analysis(df)
    abl = ablation_analysis(full_df)
    mixed = mixed_model_analysis(df)

    # Generate LaTeX tables
    tables = {
        "rq1_table.tex": generate_rq1_table(rq1),
        "rq2_means_table.tex": generate_rq2_table(rq2),
        "rq2_tests_table.tex": generate_rq2_tests_table(rq2),
        "causal_table.tex": generate_causal_table(causal),
        "ablation_table.tex": generate_ablation_table(abl),
        "mixed_model_table.tex": generate_mixed_table(mixed),
    }

    for fname, content in tables.items():
        (OUTPUT_DIR / fname).write_text(content, encoding="utf-8")
        print(f"\nSaved: {OUTPUT_DIR / fname}")

    # Save all results as JSON
    all_results = {
        "method": {
            "unit": "debate (run)",
            "excluded_runs": "5 reversed-order ablation runs (used only in the ablation)",
            "responsiveness": "opening turn of each debate excluded; 'responsiveness_all' includes it",
            "p_values": f"topic-stratified permutation test on debate means ({N_PERM} permutations)",
            "confidence_intervals": f"topic-stratified cluster bootstrap over debates ({N_BOOT} resamples)",
            "multiple_comparisons": "Holm: RQ1 across 7 dims (confirmatory); RQ2 across 28; others exploratory",
            "seed": SEED,
        },
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "causal": causal,
        "rq5": rq5,
        "ablation": abl,
        "mixed_model": mixed,
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
