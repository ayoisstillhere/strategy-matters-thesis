"""
Select stratified sample of ~35 exchanges for human annotation.
================================================================
Produces:
  runs/posthoc_scores/annotation_sample.json   — machine-readable sample
  runs/posthoc_scores/annotation_sample.txt    — human-readable printout

Stratification:
  1. High-disagreement tier: turns with largest |in-debate − posthoc| delta
     on responsiveness and stance_differentiation (priority dimensions).
  2. Stratified-random tier: fill remaining slots ensuring every
     (condition × topic) cell is represented at least once.
  3. Score-range diversity: prefer turns that aren't all-4s.

Usage:
  .venv\\Scripts\\python.exe select_annotation_sample.py
  .venv\\Scripts\\python.exe select_annotation_sample.py --n 40
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(2026)

PROJECT_ROOT = Path(__file__).resolve().parent

DIMS = [
    "civility", "relevance", "logical_consistency", "argument_strength",
    "document_grounding", "responsiveness", "stance_differentiation",
]
PRIORITY_DIMS = ["responsiveness", "stance_differentiation"]
CONTEXT_TURNS = 3  # preceding turns to include for annotator context


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load posthoc scores + in-debate scores
# ═══════════════════════════════════════════════════════════════════════════

def load_posthoc_csv() -> list[dict]:
    """Load posthoc CSV rows (main experiment only, exclude ablation)."""
    path = PROJECT_ROOT / "runs" / "posthoc_scores" / "posthoc_scores.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "ablation" not in row["run_file"]:
                rows.append(row)
    print(f"Loaded {len(rows)} main-experiment turns from posthoc CSV")
    return rows


def load_indebate_scores(posthoc_rows: list[dict]) -> dict[str, dict]:
    """Load in-debate scores from original run JSONs, keyed by turn_id."""
    run_files = set(r["run_file"] for r in posthoc_rows)
    indebate = {}
    for rf in sorted(run_files):
        path = PROJECT_ROOT / rf
        data = json.load(open(path, "r", encoding="utf-8"))
        for t in data["turns"]:
            indebate[t["turn_id"]] = {
                "scores": t.get("scores", {}),
                "text": t["text"],
            }
    print(f"Loaded in-debate scores for {len(indebate)} turns from {len(run_files)} run files")
    return indebate


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Compute disagreement deltas
# ═══════════════════════════════════════════════════════════════════════════

def compute_deltas(posthoc_rows: list[dict], indebate: dict[str, dict]) -> list[dict]:
    """Add per-dimension delta and priority delta to each row."""
    enriched = []
    for row in posthoc_rows:
        tid = row["turn_id"]
        idb = indebate.get(tid, {}).get("scores", {})
        if not idb:
            continue

        row = dict(row)  # copy
        row["_deltas"] = {}
        total_delta = 0
        for dim in DIMS:
            posthoc_val = int(row[dim])
            indebate_val = idb.get(dim, posthoc_val)
            row["_deltas"][dim] = abs(posthoc_val - indebate_val)
            total_delta += abs(posthoc_val - indebate_val)

        # Priority delta: sum of deltas on responsiveness + stance_differentiation
        row["_priority_delta"] = sum(row["_deltas"][d] for d in PRIORITY_DIMS)
        row["_total_delta"] = total_delta

        # Score variance: std of posthoc scores across dims (prefer non-uniform)
        scores = [int(row[d]) for d in DIMS]
        mean_s = sum(scores) / len(scores)
        row["_score_var"] = sum((s - mean_s) ** 2 for s in scores) / len(scores)

        enriched.append(row)

    print(f"Computed deltas for {len(enriched)} turns")
    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Stratified selection
# ═══════════════════════════════════════════════════════════════════════════

def select_sample(enriched: list[dict], n: int = 35) -> list[dict]:
    """Select stratified sample of n exchanges."""

    selected: list[dict] = []
    selected_ids: set[str] = set()
    agent_counts: Counter = Counter()
    MAX_PER_AGENT = n // 6 + 2  # ~7–8 max per agent for n=35

    def cell(row: dict) -> str:
        return f"{row['condition_id']}|{row['topic_id']}"

    def agent_ok(row: dict) -> bool:
        return agent_counts[row['agent_name']] < MAX_PER_AGENT

    # --- Tier 1: High-disagreement turns (priority dims) ---
    # Take top turns by priority_delta, ensuring topic/condition spread
    tier1_target = min(12, n // 3)
    by_delta = sorted(enriched, key=lambda r: (-r["_priority_delta"], -r["_score_var"]))

    tier1_cells: set[str] = set()
    for row in by_delta:
        if row["_priority_delta"] == 0:
            break
        c = cell(row)
        # Allow max 2 per cell in tier 1
        cell_count = sum(1 for s in selected if cell(s) == c)
        if cell_count < 2 and row["turn_id"] not in selected_ids and agent_ok(row):
            selected.append(row)
            selected_ids.add(row["turn_id"])
            tier1_cells.add(c)
            agent_counts[row["agent_name"]] += 1
        if len(selected) >= tier1_target:
            break

    print(f"Tier 1 (high-disagreement): {len(selected)} turns, "
          f"covering {len(tier1_cells)} cells")

    # --- Tier 2: Stratified random fill ---
    # Ensure every (condition × topic) cell has at least 1 turn
    all_conditions = sorted(set(r["condition_id"] for r in enriched))
    all_topics = sorted(set(r["topic_id"] for r in enriched))
    all_cells = {f"{c}|{t}" for c in all_conditions for t in all_topics}
    covered_cells = {cell(s) for s in selected}
    missing_cells = all_cells - covered_cells

    # Group remaining turns by cell
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        if row["turn_id"] not in selected_ids:
            by_cell[cell(row)].append(row)

    # Fill missing cells first (1 per cell, prefer high score variance)
    for mc in sorted(missing_cells):
        candidates = by_cell.get(mc, [])
        if not candidates:
            continue
        # Prefer turns with higher score variance (more interesting to annotate)
        candidates.sort(key=lambda r: -r["_score_var"])
        # Find first candidate that passes agent cap
        pick = None
        for cand in candidates:
            if agent_ok(cand):
                pick = cand
                break
        if pick is None:
            pick = candidates[0]  # fallback: coverage > balance
        selected.append(pick)
        selected_ids.add(pick["turn_id"])
        agent_counts[pick["agent_name"]] += 1
        candidates.remove(pick)

    covered_cells = {cell(s) for s in selected}
    print(f"Tier 2a (cell coverage): {len(selected)} turns, "
          f"covering {len(covered_cells)}/{len(all_cells)} cells")

    # Fill remaining slots with random draws weighted by score variance
    remaining = [r for r in enriched if r["turn_id"] not in selected_ids]
    remaining.sort(key=lambda r: -r["_score_var"])

    while len(selected) < n and remaining:
        # Weighted random from top-50% by variance, respecting agent cap
        pool_size = max(1, len(remaining) // 2)
        pool = [r for r in remaining[:pool_size] if agent_ok(r)]
        if not pool:
            pool = remaining[:pool_size]  # fallback
        pick = random.choice(pool)
        selected.append(pick)
        selected_ids.add(pick["turn_id"])
        agent_counts[pick["agent_name"]] += 1
        remaining.remove(pick)

    print(f"Tier 2b (random fill): {len(selected)} total turns selected")

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Build annotation exchanges (turn + context)
# ═══════════════════════════════════════════════════════════════════════════

def build_exchanges(
    sample: list[dict],
    indebate: dict[str, dict],
) -> list[dict]:
    """For each sampled turn, retrieve preceding context from the run JSON."""

    # Cache run data
    run_cache: dict[str, dict] = {}
    exchanges = []

    for idx, row in enumerate(sample):
        run_file = PROJECT_ROOT / row["run_file"]
        rf_str = str(run_file)
        if rf_str not in run_cache:
            run_cache[rf_str] = json.load(open(run_file, "r", encoding="utf-8"))

        data = run_cache[rf_str]
        all_turns = data["turns"]

        # Find position of target turn
        target_idx = None
        for i, t in enumerate(all_turns):
            if t["turn_id"] == row["turn_id"]:
                target_idx = i
                break

        if target_idx is None:
            print(f"  WARNING: turn_id {row['turn_id']} not found in {row['run_file']}")
            continue

        target_turn = all_turns[target_idx]

        # Get preceding context turns
        context_start = max(0, target_idx - CONTEXT_TURNS)
        context_turns = all_turns[context_start:target_idx]

        exchange = {
            "exchange_id": idx + 1,
            "run_file": row["run_file"],
            "condition_id": row["condition_id"],
            "topic_id": row["topic_id"],
            "run_number": int(row["run_number"]),
            "round_number": int(row["round_number"]),
            "turn_in_round": int(row["turn_in_round"]),
            "agent_name": row["agent_name"],
            "turn_id": row["turn_id"],
            "posthoc_scores": {d: int(row[d]) for d in DIMS},
            "indebate_scores": indebate.get(row["turn_id"], {}).get("scores", {}),
            "deltas": row["_deltas"],
            "priority_delta": row["_priority_delta"],
            "context": [
                {
                    "agent_name": ct["agent_name"],
                    "round_number": ct["round_number"],
                    "text": ct["text"],
                }
                for ct in context_turns
            ],
            "target_text": target_turn["text"],
        }
        exchanges.append(exchange)

    # Shuffle order for annotation (avoid ordering by condition/topic)
    random.shuffle(exchanges)
    # Re-number after shuffle
    for i, ex in enumerate(exchanges):
        ex["exchange_id"] = i + 1

    return exchanges


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Output
# ═══════════════════════════════════════════════════════════════════════════

def write_json(exchanges: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exchanges, f, ensure_ascii=False, indent=2)
    print(f"JSON written: {path} ({len(exchanges)} exchanges)")


def write_readable(exchanges: list[dict], path: Path) -> None:
    """Write human-readable text file for review before building Google Form."""
    lines = []
    for ex in exchanges:
        lines.append(f"\n{'='*80}")
        lines.append(f"EXCHANGE {ex['exchange_id']}/{len(exchanges)}")
        lines.append(f"{'='*80}")
        lines.append(f"Topic: {ex['topic_id']}")
        lines.append(f"Round {ex['round_number']}")
        lines.append(f"[Internal: condition={ex['condition_id']}, "
                     f"run={ex['run_number']}, delta={ex['priority_delta']}]")

        if ex["context"]:
            lines.append(f"\n--- Context (preceding turns) ---")
            for ct in ex["context"]:
                lines.append(f"\n  {ct['agent_name']} (Round {ct['round_number']}):")
                # Truncate context at 600 chars for readability
                text = ct["text"][:600]
                if len(ct["text"]) > 600:
                    text += "..."
                for line in text.split("\n"):
                    lines.append(f"    {line}")

        lines.append(f"\n--- ★ TURN TO SCORE ---")
        lines.append(f"  {ex['agent_name']} (Round {ex['round_number']}):")
        for line in ex["target_text"].split("\n"):
            lines.append(f"    {line}")

        lines.append(f"\n--- SCORES (hidden from annotators) ---")
        lines.append(f"  Posthoc:  {ex['posthoc_scores']}")
        lines.append(f"  In-debate: {ex['indebate_scores']}")
        lines.append(f"  Deltas:   {ex['deltas']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Readable file written: {path}")


def print_summary(exchanges: list[dict]) -> None:
    """Print coverage statistics."""
    print(f"\n{'='*60}")
    print(f"SAMPLE SUMMARY: {len(exchanges)} exchanges")
    print(f"{'='*60}")

    # Topic coverage
    topic_counts = Counter(ex["topic_id"] for ex in exchanges)
    print(f"\nBy topic:")
    for t, c in sorted(topic_counts.items()):
        print(f"  {t}: {c}")

    # Condition coverage
    cond_counts = Counter(ex["condition_id"] for ex in exchanges)
    print(f"\nBy condition:")
    for c, n in sorted(cond_counts.items()):
        print(f"  {c}: {n}")

    # Agent coverage
    agent_counts = Counter(ex["agent_name"] for ex in exchanges)
    print(f"\nBy agent:")
    for a, n in sorted(agent_counts.items()):
        print(f"  {a}: {n}")

    # Priority delta distribution
    deltas = [ex["priority_delta"] for ex in exchanges]
    high_delta = sum(1 for d in deltas if d > 0)
    print(f"\nHigh-disagreement turns (priority_delta > 0): {high_delta}/{len(exchanges)}")

    # Score range coverage (posthoc)
    for dim in DIMS:
        vals = [ex["posthoc_scores"][dim] for ex in exchanges]
        print(f"  {dim}: min={min(vals)}, max={max(vals)}, "
              f"unique={sorted(set(vals))}")

    # Cell coverage
    cells = set(f"{ex['condition_id']}|{ex['topic_id']}" for ex in exchanges)
    print(f"\nCondition×Topic cells covered: {len(cells)}/32")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Select annotation sample")
    parser.add_argument("--n", type=int, default=35,
                        help="Number of exchanges to select (default: 35)")
    args = parser.parse_args()

    # Load data
    posthoc_rows = load_posthoc_csv()
    indebate = load_indebate_scores(posthoc_rows)

    # Compute deltas
    enriched = compute_deltas(posthoc_rows, indebate)

    # Select sample
    sample = select_sample(enriched, n=args.n)

    # Build exchanges with context
    exchanges = build_exchanges(sample, indebate)

    # Output
    out_dir = PROJECT_ROOT / "runs" / "posthoc_scores"
    write_json(exchanges, out_dir / "annotation_sample.json")
    write_readable(exchanges, out_dir / "annotation_sample.txt")

    # Summary
    print_summary(exchanges)


if __name__ == "__main__":
    main()
