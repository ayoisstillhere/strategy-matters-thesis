"""Per-party judge scores by condition (Discussion, Section 6.3).

Outputs:
  experiments/outputs/comparative/party_by_condition.json
  experiments/outputs/comparative/party_by_condition_table.tex

Usage: python experiments/party_by_condition.py
"""

import json

from comparative_analysis import CONDITION_ORDER, OUTPUT_DIR, load_posthoc

PARTY_ORDER = ["CDU/CSU", "SPD",
               "Bündnis 90/Die Grünen", "FDP", "Die Linke", "AfD"]
PARTY_TEX = {"Bündnis 90/Die Grünen": "B90/Gr\\\"{u}ne"}
STRATEGIES = ["strategy_a", "strategy_b", "strategy_c", "strategy_d"]
SHORT = {
    "baseline_1": "None", "baseline_2": "Nudge", "baseline_3": "Haberm.", "baseline_4": "Random",
    "strategy_a": "De-esc.", "strategy_b": "Refr.", "strategy_c": "Fact-r.", "strategy_d": "Comm.-g.",
    "strategies": "Strat.",
}


def main():
    df = load_posthoc()
    df = df[~df["is_ablation"]]
    parties = [p for p in PARTY_ORDER if p in set(df["agent_name"])]
    missing = set(df["agent_name"]) - set(parties)
    assert not missing, f"Unexpected agent names: {missing}"

    comp = df.pivot_table(
        index="agent_name", columns="condition_id", values="composite", aggfunc="mean")
    comp["strategies"] = df[df["condition_id"].isin(
        STRATEGIES)].groupby("agent_name")["composite"].mean()
    overall = df.groupby("agent_name")[
        ["document_grounding", "stance_differentiation"]].mean()

    out = {
        p: {
            "composite": {c: round(float(comp.loc[p, c]), 3) for c in CONDITION_ORDER + ["strategies"]},
            "document_grounding_all": round(float(overall.loc[p, "document_grounding"]), 3),
            "stance_differentiation_all": round(float(overall.loc[p, "stance_differentiation"]), 3),
        }
        for p in parties
    }
    (OUTPUT_DIR / "party_by_condition.json").write_text(json.dumps(out,
                                                                   indent=2, ensure_ascii=False), encoding="utf-8")

    cols = CONDITION_ORDER + ["strategies"]
    header = " & ".join(["\\textbf{Party}"] + [SHORT[c]
                        for c in cols] + ["Doc.G.", "Sta.D."])
    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption[Judge scores by party and condition]{Mean composite score per party and condition (post-hoc judge, 1--5 scale; 20 debates per condition, 10 turns per party per debate). Strat.: the four strategies pooled. The last two columns give each party's mean document-grounding (Doc.G.) and stance differentiation (Sta.D.) across all conditions.}",
        "\\label{tab:party-condition}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l " + "c" * len(cols) + " cc}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for p in parties:
        row = [PARTY_TEX.get(p, p)] + [f"{comp.loc[p, c]:.2f}" for c in cols]
        row += [f"{overall.loc[p, 'document_grounding']:.2f}",
                f"{overall.loc[p, 'stance_differentiation']:.2f}"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "party_by_condition_table.tex").write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
