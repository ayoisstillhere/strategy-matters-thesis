"""Unblind the hand-check codes and compare conditions.

Codes were assigned blind to condition (see hand_check_sheet.py for the rules):
  responsiveness:      S substantive reply / N names or agrees only / X no engagement
  document_grounding:  T traceable claim / R programme reference without traceable claim / X neither
Flags: copied (multi-sentence verbatim copying of another speaker), source_error
(fabricated, misattributed, or misrepresented source reference).

Output: experiments/outputs/judge_validity/hand_check.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT_DIR  # noqa: E402

LABELS = {"baseline_1": "None", "strategy_b": "Reframing",
          "strategy_c": "Fact-reminder"}


def main():
    codes = pd.read_csv(OUT_DIR / "hand_check_codes.csv")
    key = pd.read_csv(OUT_DIR / "hand_check_key.csv")
    d = key.merge(codes, on="item")
    d["condition"] = d["condition_id"].map(LABELS)
    d[["copied", "source_error"]] = d[[
        "copied", "source_error"]].fillna(0).astype(int)

    res = {}
    for dim, top in [("responsiveness", "S"), ("document_grounding", "T")]:
        sub = d[d["dimension"] == dim]
        tab = pd.crosstab(sub["condition"], sub["code"])
        flag = "copied" if dim == "responsiveness" else "source_error"
        r = {"counts": tab.to_dict(orient="index"),
             f"{flag}_by_condition": sub.groupby("condition")[flag].sum().to_dict(),
             "judge_score_by_code": sub.groupby("code")[dim].mean().round(2).to_dict()}
        conds = list(sub["condition"].unique())
        a = sub[sub["condition"] == conds[0]]
        b = sub[sub["condition"] == conds[1]]
        for name, col in [(top, sub["code"] == top), (flag, sub[flag] == 1)]:
            ta = [[int(col[a.index].sum()), int((~col[a.index]).sum())],
                  [int(col[b.index].sum()), int((~col[b.index]).sum())]]
            r[f"fisher_{name}"] = {"table": {conds[0]: ta[0],
                                             conds[1]: ta[1]}, "p": float(fisher_exact(ta)[1])}
        res[dim] = r

    (OUT_DIR / "hand_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
