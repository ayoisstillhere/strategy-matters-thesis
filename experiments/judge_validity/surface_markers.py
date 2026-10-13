"""Rubric-intervention overlap: do surface markers account for the strategy gains?

Reframing tells agents to address "a named opposing party", and the responsiveness
rubric rewards naming the party and claim. Fact-reminder tells agents to anchor
arguments in "documented positions", and the document-grounding rubric rewards
explicit programme anchoring. This script counts the corresponding surface markers
in each turn and checks how much of each strategy's gain over no moderation remains
after adjusting for them.

Output: experiments/outputs/judge_validity/surface_markers.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT_DIR, load_turns  # noqa: E402

PARTY_PATTERNS = {
    "CDU/CSU": r"\bCDU\b|\bCSU\b|\bUnion\b|Christdemokrat",
    "SPD": r"\bSPD\b|Sozialdemokrat",
    "Bündnis 90/Die Grünen": r"Grünen?\b|Bündnis\s*90",
    "FDP": r"\bFDP\b|Freie[n]? Demokrat|Liberalen\b",
    "Die Linke": r"\bLinken?\b",
    "AfD": r"\bAfD\b",
}
# Acknowledgement formula, e.g. "Wir verstehen die Argumente/Bedenken von ..."
ACK = re.compile(
    r"\b(?:verstehen|respektieren|anerkennen|erkennen|teilen|nehmen)\b[^.]{0,40}\b"
    r"(?:Argument\w*|Bedenken|Position\w*|Sorge\w*|Einw\w*|Kritik|Standpunkt\w*|Forderung\w*)",
    re.IGNORECASE,
)
PROGRAMME = re.compile(
    r"Wahlprogramm|Bundestagswahlprogramm|Grundsatzprogramm|Parteiprogramm|Regierungsprogramm"
    r"|Wahl-O-Mat|\bunser(?:em|es)? Programm|dokumentierte\w* Position",
    re.IGNORECASE,
)
STRATEGIES = {
    "strategy_a": "De-escalation", "strategy_b": "Reframing",
    "strategy_c": "Fact-reminder", "strategy_d": "Common-ground",
}


def add_markers(df: pd.DataFrame) -> pd.DataFrame:
    def n_others(row):
        return sum(bool(re.search(p, row["text"])) for party, p in PARTY_PATTERNS.items()
                   if party != row["agent_name"])

    df["n_other_parties"] = df.apply(n_others, axis=1)
    df["names_other"] = (df["n_other_parties"] > 0).astype(int)
    df["ack_formula"] = df["text"].str.contains(ACK).astype(int)
    df["programme_ref"] = df["text"].str.contains(PROGRAMME).astype(int)
    return df


def adjusted_effects(df: pd.DataFrame, outcome: str, markers: list[str]) -> dict:
    """OLS on turns with debate-clustered SEs: strategy coefficient vs. None, without and with markers."""
    out = {}
    for cid, label in STRATEGIES.items():
        sub = df[df["condition_id"].isin(["baseline_1", cid])].dropna(
            subset=[outcome]).copy()
        sub["treat"] = (sub["condition_id"] == cid).astype(int)
        res = {}
        for name, rhs in [("unadjusted", "treat"), ("adjusted", "treat + " + " + ".join(markers))]:
            m = smf.ols(f"{outcome} ~ {rhs} + C(topic_id) + C(agent_name)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["run_id"]})
            ci = m.conf_int().loc["treat"]
            res[name] = {"beta": m.params["treat"], "ci": [
                ci[0], ci[1]], "p": m.pvalues["treat"]}
            if name == "adjusted":
                res["marker_betas"] = {k: m.params[k] for k in markers}
        u, a = res["unadjusted"]["beta"], res["adjusted"]["beta"]
        res["share_explained"] = 1 - a / u if abs(u) > 0.05 else None
        out[label] = res
    return out


def stratified_means(df: pd.DataFrame, outcome: str, marker: str) -> dict:
    """Mean outcome by condition within marker strata (marker absent / present)."""
    out = {}
    for cid in ["baseline_1"] + list(STRATEGIES):
        g = df[df["condition_id"] == cid].dropna(subset=[outcome])
        label = "None" if cid == "baseline_1" else STRATEGIES[cid]
        out[label] = {
            "share_with_marker": g[marker].mean(),
            "mean_without": g.loc[g[marker] == 0, outcome].mean(),
            "mean_with": g.loc[g[marker] == 1, outcome].mean(),
            "n_without": int((g[marker] == 0).sum()),
            "n_with": int((g[marker] == 1).sum()),
        }
    return out


def main():
    df = add_markers(load_turns())

    prevalence = (df.groupby("condition_id")[["names_other", "n_other_parties", "ack_formula", "programme_ref"]]
                  .mean().round(3))
    results = {
        "prevalence": prevalence.to_dict(orient="index"),
        "score_by_marker": {
            "responsiveness_names_other": df.groupby("names_other")["responsiveness"].mean().to_dict(),
            "responsiveness_ack_formula": df.groupby("ack_formula")["responsiveness"].mean().to_dict(),
            "document_grounding_programme_ref": df.groupby("programme_ref")["document_grounding"].mean().to_dict(),
        },
        "responsiveness": {
            "adjusted": adjusted_effects(df, "responsiveness", ["n_other_parties", "ack_formula"]),
            "by_names_other": stratified_means(df, "responsiveness", "names_other"),
            "by_ack_formula": stratified_means(df, "responsiveness", "ack_formula"),
        },
        "document_grounding": {
            "adjusted": adjusted_effects(df, "document_grounding", ["programme_ref"]),
            "by_programme_ref": stratified_means(df, "document_grounding", "programme_ref"),
        },
    }
    (OUT_DIR / "surface_markers.json").write_text(json.dumps(results,
                                                             indent=2, default=float), encoding="utf-8")
    df[["turn_id", "run_id", "condition_id", "topic_id", "agent_name", "round_number", "turn_in_round",
        "n_other_parties", "names_other", "ack_formula", "programme_ref"]].to_csv(
        OUT_DIR / "surface_markers_turns.csv", index=False)

    print("Prevalence by condition:\n", prevalence)
    print("\nMean score by marker:")
    for k, v in results["score_by_marker"].items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:.2f}" for kk, vv in v.items()))
    for outcome in ["responsiveness", "document_grounding"]:
        print(
            f"\n{outcome}: strategy - None (turn OLS, topic+agent FE, debate-clustered SE)")
        for label, r in results[outcome]["adjusted"].items():
            u, a = r["unadjusted"], r["adjusted"]
            se = r["share_explained"]
            print(f"  {label:<14} unadj {u['beta']:+.2f} [{u['ci'][0]:+.2f},{u['ci'][1]:+.2f}]  "
                  f"adj {a['beta']:+.2f} [{a['ci'][0]:+.2f},{a['ci'][1]:+.2f}] p={a['p']:.3g}  "
                  f"explained={'n/a' if se is None else f'{se:.0%}'}  markers={ {k: round(v, 2) for k, v in r['marker_betas'].items()} }")
    for key, (outcome, strat) in {"by_names_other": ("responsiveness", "names_other"),
                                  "by_ack_formula": ("responsiveness", "ack_formula"),
                                  "by_programme_ref": ("document_grounding", "programme_ref")}.items():
        print(f"\n{outcome} {key}:")
        for label, r in results[outcome][key].items():
            print(f"  {label:<14} share={r['share_with_marker']:.2f}  without={r['mean_without']:.2f} (n={r['n_without']})"
                  f"  with={r['mean_with']:.2f} (n={r['n_with']})")


if __name__ == "__main__":
    main()
