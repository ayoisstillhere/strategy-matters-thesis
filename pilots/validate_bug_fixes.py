"""
Bug Fix Validation — 3-Round Debate
=====================================
Validates 4 bug fixes from 20260629:

  Fix 1: Truncation  — max_tokens raised to 800, turns should complete
  Fix 2: Round 1 skip — no trigger should fire in round 1
  Fix 3: 1/round cap  — at most 1 trigger per round
  Fix 4: RAG enabled  — rag_passages_used non-empty in agent turns

Runs 1 debate: migrationspolitik, Strategy D (common-ground), 3 rounds.
Strategy D chosen because its responsiveness trigger fires most readily,
so all trigger-related fixes are exercised.

Usage:
    cd strategy-matters-thesis
    python pilots/validate_bug_fixes.py

Requires:
    - GROQ_API_KEY in .env or environment
    - FAISS indices at data/embeddings/ (built by data/build_faiss_indices.py)
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.rag_pipeline import RAGPipeline
from src.experiment_config import FRAMING_PROMPTS
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_bugs")

NUM_ROUNDS = 3
TOPIC = "sozialpolitik"
CONDITION = "strategy_d"


def check_truncation(turns) -> list[str]:
    """Returns list of agent names with truncated turns."""
    truncated = []
    for t in turns:
        if t.text and t.text.rstrip()[-1] not in ".!?\"')":
            truncated.append(f"R{t.round_number}T{t.turn_in_round} {t.agent_name}")
    return truncated


def check_round1_triggers(interventions) -> list:
    """Returns any triggers that fired in round 1."""
    return [i for i in interventions if i.round_number == 1]


def check_multi_triggers_per_round(interventions) -> list[str]:
    """Returns rounds where more than 1 trigger fired."""
    from collections import Counter
    counts = Counter(i.round_number for i in interventions)
    return [f"Round {r}: {n} triggers" for r, n in counts.items() if n > 1]


def check_rag(turns) -> tuple[int, int]:
    """Returns (turns_with_rag, total_turns)."""
    with_rag = sum(1 for t in turns if t.rag_passages_used)
    return with_rag, len(turns)


def main():
    print("\n" + "=" * 65)
    print("  BUG FIX VALIDATION")
    print(f"  {TOPIC} | {CONDITION} | {NUM_ROUNDS} rounds | RAG enabled")
    print("=" * 65 + "\n")

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Set GROQ_API_KEY in .env: {e}")
        sys.exit(1)

    rag = RAGPipeline()

    engine = DebateEngine(
        topic_id=TOPIC,
        framing_prompt=FRAMING_PROMPTS[TOPIC],
        topic_type="values-driven",
        condition_id=CONDITION,
        run_number=1,
        llm_client=client,
        rag_pipeline=rag,
        num_rounds=NUM_ROUNDS,
    )

    logger.info("Running debate…")
    result = engine.run()
    path = save_run_json(result, PROJECT_ROOT / "runs" / "validation")
    logger.info(f"Saved to {path}")

    turns = result.turns
    interventions = result.interventions

    print(f"\n{'='*65}")
    print("  RESULTS")
    print(f"{'='*65}")
    print(f"  Turns: {len(turns)} | Interventions: {len(interventions)}")
    print()

    # ── Fix 1: Truncation ──────────────────────────────────────────────
    truncated = check_truncation(turns)
    if truncated:
        print(f"  FIX 1 TRUNCATION  ✗  {len(truncated)}/{len(turns)} turns still truncated:")
        for t in truncated[:5]:
            print(f"         {t}")
        fix1_ok = False
    else:
        print(f"  FIX 1 TRUNCATION  ✓  0/{len(turns)} turns truncated")
        fix1_ok = True

    # ── Fix 2: No round-1 triggers ────────────────────────────────────
    r1_triggers = check_round1_triggers(interventions)
    if r1_triggers:
        print(f"  FIX 2 ROUND-1 SKIP ✗  {len(r1_triggers)} trigger(s) still fired in round 1")
        fix2_ok = False
    else:
        print(f"  FIX 2 ROUND-1 SKIP ✓  No triggers in round 1")
        fix2_ok = True

    # ── Fix 3: Max 1 trigger per round ───────────────────────────────
    multi = check_multi_triggers_per_round(interventions)
    if multi:
        print(f"  FIX 3 1/ROUND CAP  ✗  Multiple triggers in: {multi}")
        fix3_ok = False
    else:
        print(f"  FIX 3 1/ROUND CAP  ✓  At most 1 trigger per round")
        fix3_ok = True

    # ── Fix 4: RAG passages ───────────────────────────────────────────
    with_rag, total = check_rag(turns)
    if with_rag == 0:
        print(f"  FIX 4 RAG ENABLED  ✗  0/{total} turns have RAG passages")
        fix4_ok = False
    else:
        print(f"  FIX 4 RAG ENABLED  ✓  {with_rag}/{total} turns have RAG passages")
        fix4_ok = True

    # ── Intervention timeline ─────────────────────────────────────────
    if interventions:
        print(f"\n  Intervention timeline:")
        for iv in interventions:
            tag = "[SILENT]" if iv.silent_control else "[ACTIVE]"
            print(f"    R{iv.round_number} {tag} dim={iv.trigger_dimension} "
                  f"score={iv.trigger_score}")

    # ── Overall ───────────────────────────────────────────────────────
    all_ok = fix1_ok and fix2_ok and fix3_ok and fix4_ok
    print(f"\n{'='*65}")
    if all_ok:
        print("  ALL 4 BUG FIXES VALIDATED ✓")
    else:
        failed = [f"Fix {i+1}" for i, ok in
                  enumerate([fix1_ok, fix2_ok, fix3_ok, fix4_ok]) if not ok]
        print(f"  ISSUES REMAIN: {', '.join(failed)}")
    print(f"{'='*65}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
