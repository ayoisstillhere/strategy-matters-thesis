"""
Judge Reliability Pilot
========================
Scores 18 turns from existing runs 3× each with the EvaluationJudge
(temperature=0, deterministic). Computes per-dimension intra-judge
consistency stats.

Metrics (per dimension):
  - Exact agreement rate  : % of turns where all 3 scores are identical
  - Within-±1 rate        : % of turns where max score − min score ≤ 1
  - Mean std              : average standard deviation of 3 scores per turn
    (0.0 = perfect, ~0.5 = one-step variation, >1.0 = unreliable)

Decision criterion (per action plan):
  Flag any dimension with exact agreement < 90%.
  Pay particular attention to document_grounding and stance_differentiation.

Turn sampling:
  Loads the 3 most recent run JSONs (across any topic/condition).
  Takes 3 turns per party (CDU/CSU, SPD, Grünen, FDP, Linke, AfD) =
  18 turns total. Prefers turns from different rounds for diversity.

Usage:
    cd strategy-matters-thesis
    python pilots/judge_reliability.py

Output:
  runs/judge_reliability/reliability_scores_<timestamp>.json
  Console: per-dimension reliability table + flagged dimensions
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient, JUDGE_MODEL
from src.judge import EvaluationJudge
from src.models import Turn, DimensionScores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("judge_reliability")

OUTPUT_DIR = PROJECT_ROOT / "runs" / "judge_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation",
]

PARTIES = ["CDU/CSU", "SPD", "Bündnis 90/Die Grünen", "FDP", "Die Linke", "AfD"]
RERUNS  = 3          # number of times each turn is re-scored
TURNS_PER_PARTY = 3  # 6 parties × 3 = 18 turns total
EXACT_AGREEMENT_THRESHOLD = 0.90   # flag if below this


# ---------------------------------------------------------------------------
# Sample turns from existing runs
# ---------------------------------------------------------------------------

def sample_turns(num_per_party: int = TURNS_PER_PARTY) -> list[dict]:
    """
    Load turns from the most recent run JSONs in runs/.
    Returns a flat list of raw turn dicts (with all required fields).
    """
    all_run_files = sorted(
        (PROJECT_ROOT / "runs").rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Collect turns per party, picking from different runs/rounds for diversity
    party_turns: dict[str, list[dict]] = {p: [] for p in PARTIES}

    for run_file in all_run_files:
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        turns = data.get("turns", [])
        topic = data.get("config", {}).get("topic_id", "unknown")

        for turn in turns:
            party = turn.get("agent_name", "")
            if party not in party_turns:
                continue
            if len(party_turns[party]) >= num_per_party:
                continue
            # Must have text and preceding context
            if not turn.get("text"):
                continue
            # Attach topic for judge context
            turn["_topic"] = topic
            # Collect preceding turns from this run (up to 6 prior turns)
            idx = turns.index(turn)
            turn["_preceding"] = turns[max(0, idx - 6):idx]
            party_turns[party].append(turn)

        if all(len(v) >= num_per_party for v in party_turns.values()):
            break

    sampled = []
    for party, turns in party_turns.items():
        sampled.extend(turns[:num_per_party])

    logger.info(f"Sampled {len(sampled)} turns across {len(PARTIES)} parties")
    return sampled


# ---------------------------------------------------------------------------
# Reconstruct Turn object from raw dict
# ---------------------------------------------------------------------------

def dict_to_turn(raw: dict) -> Turn:
    """Reconstruct a minimal Turn from a raw JSON turn dict."""
    scores_raw = raw.get("scores") or {}
    scores = DimensionScores(
        **{d: scores_raw.get(d, 3) for d in DIMENSIONS}
    ) if scores_raw else None

    return Turn(
        turn_id=raw.get("turn_id", "unknown"),
        round_number=raw.get("round_number", 1),
        turn_in_round=raw.get("turn_in_round", 1),
        agent_name=raw.get("agent_name", "Unknown"),
        text=raw.get("text", ""),
        scores=scores,
    )


def dicts_to_turns(raw_list: list[dict]) -> list[Turn]:
    return [dict_to_turn(r) for r in raw_list]


# ---------------------------------------------------------------------------
# Re-score a turn N times and collect scores
# ---------------------------------------------------------------------------

def rescore_turn(
    judge: EvaluationJudge,
    raw_turn: dict,
    n: int = RERUNS,
) -> list[dict[str, int]]:
    """Score one turn n times. Returns list of score dicts."""
    turn = dict_to_turn(raw_turn)
    preceding = dicts_to_turns(raw_turn.get("_preceding", []))
    topic = raw_turn.get("_topic", "mindestlohn")

    all_scores = []
    for _ in range(n):
        scores, _, _, _, _ = judge.score_turn(
            turn=turn,
            topic=topic,
            preceding_turns=preceding,
        )
        all_scores.append(scores.to_dict())
    return all_scores


# ---------------------------------------------------------------------------
# Compute per-dimension reliability stats
# ---------------------------------------------------------------------------

def compute_reliability(
    all_turn_scores: list[list[dict[str, int]]],
) -> dict[str, dict[str, float]]:
    """
    all_turn_scores: list of N turns, each a list of RERUNS score dicts.
    Returns per-dimension stats dict.
    """
    stats: dict[str, dict[str, float]] = {}

    for dim in DIMENSIONS:
        exact_matches = 0
        within_one    = 0
        std_vals      = []

        for reruns in all_turn_scores:
            scores_for_dim = [r[dim] for r in reruns]
            span = max(scores_for_dim) - min(scores_for_dim)
            if span == 0:
                exact_matches += 1
            if span <= 1:
                within_one += 1
            if len(scores_for_dim) > 1:
                std_vals.append(stdev(scores_for_dim))

        n = len(all_turn_scores)
        stats[dim] = {
            "exact_pct":    exact_matches / n if n else 0,
            "within1_pct":  within_one / n    if n else 0,
            "mean_std":     mean(std_vals)     if std_vals else 0.0,
        }

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"run_{ts_str}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s", "%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    print("\n" + "=" * 65)
    print("  JUDGE RELIABILITY PILOT")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  {TURNS_PER_PARTY} turns × {len(PARTIES)} parties = "
          f"{TURNS_PER_PARTY * len(PARTIES)} turns × {RERUNS} reruns each")
    print(f"  temperature=0 (deterministic judge)")
    print(f"  Flag threshold: exact agreement < {EXACT_AGREEMENT_THRESHOLD:.0%}")
    print("=" * 65)

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Set GROQ_API_KEY in .env: {e}")
        sys.exit(1)

    judge = EvaluationJudge(llm_client=client, temperature=0.0)

    # 1. Sample turns
    sampled_turns = sample_turns()
    if len(sampled_turns) < TURNS_PER_PARTY * len(PARTIES):
        logger.warning(
            f"Only {len(sampled_turns)} turns sampled "
            f"(wanted {TURNS_PER_PARTY * len(PARTIES)}). "
            f"Run more debates first if needed."
        )

    # 2. Re-score each turn RERUNS times
    print(f"\n  Scoring {len(sampled_turns)} turns × {RERUNS} reruns …")
    all_turn_scores: list[list[dict[str, int]]] = []
    raw_records = []

    for i, raw_turn in enumerate(sampled_turns, 1):
        party = raw_turn.get("agent_name", "?")
        topic = raw_turn.get("_topic", "?")
        rnd   = raw_turn.get("round_number", "?")
        logger.info(f"  Turn {i:>2}/{len(sampled_turns)} — {party} | {topic} R{rnd}")

        reruns = rescore_turn(judge, raw_turn, n=RERUNS)
        all_turn_scores.append(reruns)

        raw_records.append({
            "turn_id":    raw_turn.get("turn_id"),
            "agent_name": party,
            "topic":      topic,
            "round":      rnd,
            "reruns":     reruns,
        })

    # 3. Compute stats
    stats = compute_reliability(all_turn_scores)

    # 4. Print results table
    print(f"\n  {'Dimension':<24} {'Exact %':>8} {'Within±1%':>10} {'Mean std':>9}  Flag")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*9}  ----")

    flagged_dims = []
    for dim in DIMENSIONS:
        s = stats[dim]
        flag = ""
        is_flagged = s["exact_pct"] < EXACT_AGREEMENT_THRESHOLD
        if is_flagged:
            flag = "⚠  LOW"
            flagged_dims.append(dim)
        elif dim in ("document_grounding", "stance_differentiation"):
            flag = "(watch)"
        print(
            f"  {dim:<24} {s['exact_pct']:>7.1%} {s['within1_pct']:>9.1%} "
            f"{s['mean_std']:>9.3f}  {flag}"
        )

    # 5. Verdict
    print()
    if not flagged_dims:
        print("  ✓ All dimensions meet the 90% exact-agreement threshold.")
        print("  Judge is reliable at temperature=0 for all 7 dimensions.")
    else:
        print(f"  ⚠  Low reliability in: {', '.join(flagged_dims)}")
        print("     Consider expanding human validation for these dimensions.")

    doc_g = stats["document_grounding"]
    sd    = stats["stance_differentiation"]
    print(f"\n  Watchlist:")
    print(f"    document_grounding    — exact={doc_g['exact_pct']:.1%}  "
          f"within±1={doc_g['within1_pct']:.1%}  std={doc_g['mean_std']:.3f}")
    print(f"    stance_differentiation — exact={sd['exact_pct']:.1%}  "
          f"within±1={sd['within1_pct']:.1%}  std={sd['mean_std']:.3f}")

    # 6. Save raw records
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"reliability_scores_{ts}.json"
    out_file.write_text(
        json.dumps({
            "meta": {
                "turns_sampled": len(sampled_turns),
                "reruns_per_turn": RERUNS,
                "judge_temperature": 0.0,
                "exact_agreement_threshold": EXACT_AGREEMENT_THRESHOLD,
            },
            "per_dimension_stats": stats,
            "raw_records": raw_records,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved: {out_file}")
    print(f"\n  Full results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
