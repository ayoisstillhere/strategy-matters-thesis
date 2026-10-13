"""Blinded hand-check sheet for rubric-intervention overlap (optional, author-coded).

Draws high-scoring turns from the targeted strategy and from no moderation, shuffles
them, and hides the condition, so the author can code whether a high score reflects
substantive engagement/grounding or surface compliance with the rubric.

  responsiveness: 20 Reframing + 20 None turns with responsiveness >= 4
  grounding:      20 Fact-reminder + 20 None turns with document-grounding >= 4

Outputs (experiments/outputs/judge_validity/):
  hand_check_sheet.csv  - what the coder sees (fill in the `code` column)
  hand_check_key.csv    - condition per item, for unblinding afterwards
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT_DIR, load_turns  # noqa: E402

N = 20
SEED = 2026
CODES = {
    "responsiveness": "S = engages the substance of a specific opposing claim; N = names a party/claim but only restates own position; X = no engagement",
    "document_grounding": "T = specific claim traceable to the party's programme/Wahl-O-Mat; R = refers to the programme without a specific traceable claim; X = neither",
}


def main():
    df = load_turns()
    prev_text = df.sort_values(["run_id", "round_number", "turn_in_round"]).groupby(
        "run_id")["text"].shift(1)
    prev_agent = df.sort_values(["run_id", "round_number", "turn_in_round"]).groupby(
        "run_id")["agent_name"].shift(1)
    df["previous_turn"] = (prev_agent + ": " + prev_text).reindex(df.index)

    items = []
    for dim, target in [("responsiveness", "strategy_b"), ("document_grounding", "strategy_c")]:
        for cid in [target, "baseline_1"]:
            pool = df[(df["condition_id"] == cid) & (
                df[dim] >= 4) & (df["round_number"] >= 2)]
            s = pool.sample(N, random_state=SEED).assign(dimension=dim)
            items.append(s)
    sheet = pd.concat(items).sample(
        frac=1, random_state=SEED).reset_index(drop=True)
    sheet["item"] = range(1, len(sheet) + 1)
    sheet["coding_rule"] = sheet["dimension"].map(CODES)
    sheet["code"] = ""
    sheet[["item", "dimension", "agent_name", "topic_id", "round_number", "previous_turn", "text",
           "coding_rule", "code"]].to_csv(OUT_DIR / "hand_check_sheet.csv", index=False, encoding="utf-8-sig")
    sheet[["item", "turn_id", "condition_id", "dimension", "responsiveness", "document_grounding"]].to_csv(
        OUT_DIR / "hand_check_key.csv", index=False)
    print(f"{len(sheet)} items written to {OUT_DIR / 'hand_check_sheet.csv'}")


if __name__ == "__main__":
    main()
