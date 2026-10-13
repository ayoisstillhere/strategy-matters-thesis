"""Human validation split by sampling tier.

The 50 annotated exchanges were drawn in two tiers (select_annotation_sample.py):
12 turns with the largest in-debate vs. post-hoc disagreement on responsiveness and
stance differentiation, and 38 stratified/random turns. This script rebuilds the
tier membership and reports human-LLM agreement for each tier separately.

Output: experiments/outputs/judge_validity/human_tiers.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import select_annotation_sample as sas  # noqa: E402
from human_validation_analysis import (  # noqa: E402
    DIMENSIONS, EXPORT_PATH, krippendorffs_alpha, pairwise_agreement,
)

OUT_DIR = ROOT / "experiments" / "outputs" / "judge_validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_PERM = 2000
RNG = np.random.default_rng(2026)


def tier1_turn_ids(n_sample: int) -> list[str]:
    rows = sas.load_posthoc_csv()
    enriched = sas.compute_deltas(rows, sas.load_indebate_scores(rows))
    selected = sas.select_sample(enriched, n=n_sample)
    return [r["turn_id"] for r in selected[: min(12, n_sample // 3)]]


def tier_metrics(anns: pd.DataFrame, llm: dict, turn_ids: list[str], annotator_ids: list[str]) -> dict:
    out = {"n_exchanges": len(turn_ids), "per_dimension": {}}
    pooled_hits, pooled_n, pooled_exact = 0, 0, 0
    for dim in DIMENSIONS:
        matrix = np.full((len(annotator_ids), len(turn_ids)), np.nan)
        for i, a in enumerate(annotator_ids):
            sub = anns[anns["annotator_id"] == a].set_index("turn_id")[dim]
            for j, t in enumerate(turn_ids):
                if t in sub.index and not pd.isna(sub[t]):
                    matrix[i, j] = sub[t]
        llm_vec = np.array([llm[t][dim] for t in turn_ids], dtype=float)
        diff = matrix - llm_vec[None, :]
        valid = ~np.isnan(diff)
        hits = int((np.abs(diff[valid]) <= 1).sum())
        exact = int((diff[valid] == 0).sum())
        pooled_hits += hits
        pooled_exact += exact
        pooled_n += int(valid.sum())
        human_means = np.nanmean(matrix, axis=0)
        out["per_dimension"][dim] = {
            "within1_vs_llm": hits / valid.sum(),
            "exact_vs_llm": exact / valid.sum(),
            "alpha_humans": krippendorffs_alpha(matrix),
            "within1_humans": pairwise_agreement(matrix, tolerance=1),
            "mean_human": float(np.nanmean(human_means)),
            "mean_llm": float(llm_vec.mean()),
            "bias": float(llm_vec.mean() - np.nanmean(human_means)),
        }
    out["within1_vs_llm"] = pooled_hits / pooled_n
    out["exact_vs_llm"] = pooled_exact / pooled_n
    out["mean_alpha_humans"] = float(np.nanmean(
        [v["alpha_humans"] for v in out["per_dimension"].values()]))
    out["mean_bias"] = float(np.mean([v["bias"]
                             for v in out["per_dimension"].values()]))

    # Chance baseline: LLM score vectors shuffled across exchanges within the tier.
    hum = {t: anns[anns["turn_id"] == t]
           [DIMENSIONS].to_numpy(dtype=float) for t in turn_ids}
    llm_mat = np.array([[llm[t][d] for d in DIMENSIONS]
                       for t in turn_ids], dtype=float)
    rates = []
    for _ in range(N_PERM):
        perm = RNG.permutation(len(turn_ids))
        h, n = 0, 0
        for j, t in enumerate(turn_ids):
            d = hum[t] - llm_mat[perm[j]][None, :]
            v = ~np.isnan(d)
            h += (np.abs(d[v]) <= 1).sum()
            n += v.sum()
        rates.append(h / n)
    out["within1_chance"] = float(np.mean(rates))
    return out


def main():
    data = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    excluded = set(data["excluded_annotator_ids"])
    anns = pd.DataFrame(data["annotations"])
    anns = pd.concat([anns.drop(columns=["scores"]),
                     pd.json_normalize(anns["scores"])], axis=1)
    anns = anns[(~anns["annotator_id"].isin(excluded)) & (
        anns["phase"] == "initial") & (~anns["perceived_bias"])]
    annotator_ids = sorted(anns["annotator_id"].unique())
    llm = {ex["turn_id"]: ex["llm_scores"] for ex in data["exchanges"]}
    all_ids = [ex["turn_id"] for ex in data["exchanges"]]

    tier1 = tier1_turn_ids(len(all_ids))
    missing = [t for t in tier1 if t not in llm]
    if missing:
        sys.exit(
            f"Tier-1 reconstruction failed: {len(missing)} turns not in the annotated sample")
    rest = [t for t in all_ids if t not in set(tier1)]

    results = {
        "all": tier_metrics(anns, llm, all_ids, annotator_ids),
        "high_disagreement": tier_metrics(anns, llm, tier1, annotator_ids),
        "stratified": tier_metrics(anns, llm, rest, annotator_ids),
        "tier1_turn_ids": tier1,
    }
    (OUT_DIR / "human_tiers.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'Tier':<20}{'n':>4}{'±1 LLM':>9}{'chance':>9}{'exact':>8}{'α hum':>8}{'bias':>8}")
    for k in ["all", "high_disagreement", "stratified"]:
        r = results[k]
        print(f"{k:<20}{r['n_exchanges']:>4}{r['within1_vs_llm']*100:>8.1f}%{r['within1_chance']*100:>8.1f}%"
              f"{r['exact_vs_llm']*100:>7.1f}%{r['mean_alpha_humans']:>8.3f}{r['mean_bias']:>+8.2f}")
    print("\nPer-dimension ±1 / bias (high_disagreement | stratified):")
    for d in DIMENSIONS:
        a = results["high_disagreement"]["per_dimension"][d]
        b = results["stratified"]["per_dimension"][d]
        print(
            f"  {d:<24} {a['within1_vs_llm']*100:5.1f}% {a['bias']:+.2f} | {b['within1_vs_llm']*100:5.1f}% {b['bias']:+.2f}")


if __name__ == "__main__":
    main()
