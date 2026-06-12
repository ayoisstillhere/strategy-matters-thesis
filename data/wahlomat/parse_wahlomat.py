"""
Parse Wahl-O-Mat 2025 CSV dataset into structured JSON for the 6 target parties.

Input:  Wahl-O-Mat Bundestagswahl 2025_Datensatz_v1.02 - Datensatz BTW 2025.csv
Output: wahlomat_2025.json — 38 theses × 6 parties with positions and reasoning

Usage:
    python parse_wahlomat.py
"""

import csv
import json
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "Wahl-O-Mat Bundestagswahl 2025_Datensatz_v1.02 - Datensatz BTW 2025.csv"
OUTPUT_JSON = Path(__file__).parent / "wahlomat_2025.json"

TARGET_PARTIES = {
    "SPD",
    "CDU / CSU",
    "GRÜNE",
    "FDP",
    "AfD",
    "Die Linke",
}

POSITION_MAP = {
    "stimme zu": "agree",
    "stimme nicht zu": "disagree",
    "neutral": "neutral",
}


def parse():
    theses = {}

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            party_short = row["Partei: Kurzbezeichnung"]
            if party_short not in TARGET_PARTIES:
                continue

            thesis_nr = int(row["These: Nr."])
            thesis_title = row["These: Titel"]
            thesis_text = row["These: These"]
            position_raw = row["Position: Position"].strip()
            reasoning = row["Position: Begründung"].strip()

            if thesis_nr not in theses:
                theses[thesis_nr] = {
                    "id": thesis_nr,
                    "title": thesis_title,
                    "text": thesis_text,
                    "parties": {},
                }

            theses[thesis_nr]["parties"][party_short] = {
                "position": POSITION_MAP.get(position_raw, position_raw),
                "reasoning": reasoning,
            }

    # Sort by thesis number and build final structure
    sorted_theses = [theses[k] for k in sorted(theses.keys())]

    output = {
        "election": "Bundestagswahl 2025",
        "source": "Bundeszentrale für politische Bildung (bpb) — Wahl-O-Mat Datensatz v1.02",
        "parties": sorted(TARGET_PARTIES),
        "thesis_count": len(sorted_theses),
        "theses": sorted_theses,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"Parsed {len(sorted_theses)} theses for {len(TARGET_PARTIES)} parties")
    for t in sorted_theses:
        positions = {p: t["parties"][p]["position"] for p in sorted(t["parties"])}
        print(f"  Thesis {t['id']:2d}: {t['title'][:50]:<50s} {positions}")


if __name__ == "__main__":
    parse()
