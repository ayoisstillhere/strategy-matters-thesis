"""
Smoke Test — Debate Engine
============================
Runs two short debates (2 rounds each) to verify the full pipeline:

  Test 1: Baseline 1 (no moderator) — simplest path
  Test 2: Strategy A (de-escalation) — trigger + moderator path

Both skip RAG for speed (no FAISS indices needed).
Saves results to runs/ and prints a summary.

Usage:
    cd strategy-matters-thesis
    python pilots/smoke_test_debate.py

Requires:
    - GROQ_API_KEY in .env or environment
    - pip install pydantic groq httpx python-dotenv pandas
"""

import logging
import sys
from pathlib import Path

# Add project root to path so 'src' package resolves
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.experiment_config import FRAMING_PROMPTS
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")

OUTPUT_DIR = PROJECT_ROOT / "runs"
NUM_ROUNDS = 2  # short runs for smoke test


def print_summary(result, label: str) -> None:
    """Print a human-readable summary of a debate run."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Run ID:       {result.run_id[:8]}...")
    print(f"  Topic:        {result.config.topic_id}")
    print(f"  Condition:    {result.config.condition_id} ({result.config.condition_label})")
    print(f"  Turns:        {len(result.turns)}")
    print(f"  Interventions:{result.active_intervention_count()} active, "
          f"{sum(1 for i in result.interventions if i.silent_control)} silent")
    print(f"  Tokens:       {result.total_tokens_input + result.total_tokens_output:,}")
    print(f"  Latency:      {result.total_latency_s:.1f}s")

    # Per-round score summary
    print(f"\n  Round Scores (composite):")
    for rs in result.round_summaries:
        plateau_tag = " [PLATEAU]" if rs.plateau else ""
        print(f"    Round {rs.round_number}: {rs.composite:.2f}{plateau_tag}")

    # Sample turn
    if result.turns:
        first = result.turns[0]
        print(f"\n  First turn ({first.agent_name}):")
        print(f"    \"{first.text[:200]}...\"")
        if first.scores:
            print(f"    Scores: {first.scores.to_dict()}")

    # Sample intervention
    active_interventions = [i for i in result.interventions if i.intervention_text]
    if active_interventions:
        first_int = active_interventions[0]
        print(f"\n  First intervention ({first_int.source.value}, round {first_int.round_number}):")
        print(f"    \"{first_int.intervention_text[:200]}...\"")

    print()


def test_baseline_1(client: LLMClient) -> None:
    """Test 1: Baseline 1 (None) — no moderator, simplest path."""
    logger.info("TEST 1: Baseline 1 (No Moderator)")

    engine = DebateEngine(
        topic_id="mindestlohn",
        framing_prompt=FRAMING_PROMPTS["mindestlohn"],
        topic_type="empirical",
        condition_id="baseline_1",
        run_number=1,
        llm_client=client,
        rag_pipeline=None,
        num_rounds=NUM_ROUNDS,
    )

    result = engine.run()
    path = save_run_json(result, OUTPUT_DIR)
    print_summary(result, "TEST 1: Baseline 1 (No Moderator)")
    logger.info(f"Saved to {path}")

    # Assertions
    assert len(result.turns) == NUM_ROUNDS * 6, f"Expected {NUM_ROUNDS * 6} turns"
    assert len(result.interventions) == 0, "Baseline 1 should have no interventions"
    assert len(result.round_summaries) == NUM_ROUNDS
    assert all(t.scores is not None for t in result.turns), "All turns should be scored"
    logger.info("TEST 1 PASSED ✓")


def test_strategy_a(client: LLMClient) -> None:
    """Test 2: Strategy A (De-escalation) — trigger + moderator path."""
    logger.info("TEST 2: Strategy A (De-escalation)")

    engine = DebateEngine(
        topic_id="mindestlohn",
        framing_prompt=FRAMING_PROMPTS["mindestlohn"],
        topic_type="empirical",
        condition_id="strategy_a",
        run_number=1,
        llm_client=client,
        rag_pipeline=None,
        num_rounds=NUM_ROUNDS,
    )

    result = engine.run()
    path = save_run_json(result, OUTPUT_DIR)
    print_summary(result, "TEST 2: Strategy A (De-escalation)")
    logger.info(f"Saved to {path}")

    # Assertions
    assert len(result.turns) == NUM_ROUNDS * 6, f"Expected {NUM_ROUNDS * 6} turns"
    assert len(result.round_summaries) == NUM_ROUNDS
    assert all(t.scores is not None for t in result.turns), "All turns should be scored"
    # Interventions may or may not fire depending on scores — just check structure
    for i in result.interventions:
        assert i.strategy == "de-escalation"
        assert i.round_number >= 1
    logger.info(f"TEST 2 PASSED ✓ ({len(result.interventions)} trigger events)")


def main():
    print("\n" + "="*60)
    print("  DEBATE ENGINE SMOKE TEST")
    print("  2 rounds × 6 agents = 12 turns per test")
    print("  RAG disabled (no FAISS indices needed)")
    print("="*60 + "\n")

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Failed to create LLM client: {e}")
        logger.error("Set GROQ_API_KEY in .env or environment")
        sys.exit(1)

    test_baseline_1(client)
    test_strategy_a(client)

    print("\n" + "="*60)
    print("  ALL SMOKE TESTS PASSED ✓")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
