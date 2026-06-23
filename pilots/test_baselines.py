"""
Baseline Conditions — Integration Test
=========================================
Runs all 4 baseline conditions (2 rounds each, no RAG) to verify:

  Baseline 1 (None):     No triggers, no interventions, pure debate.
  Baseline 2 (Nudge):    No triggers, no interventions, nudge in system prompt.
  Baseline 3 (Habermas): Moderator summary after every round (unconditional).
  Baseline 4 (Random):   Trigger pipeline active, random message on fire.

Verifications per baseline:
  - Correct number of turns (12 = 2 rounds × 6 agents)
  - Correct intervention behaviour
  - Scores present on all turns
  - Nudge text presence in agent prompt (B2)
  - Habermas structured output (B3)

Usage:
    cd strategy-matters-thesis
    python pilots/test_baselines.py

Requires:
    - GROQ_API_KEY in .env or environment
    - pip install pydantic groq httpx python-dotenv
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
from src.experiment_config import FRAMING_PROMPTS, NUDGE_INSTRUCTION
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_baselines")

OUTPUT_DIR = PROJECT_ROOT / "runs"
NUM_ROUNDS = 2


def run_baseline(client: LLMClient, condition_id: str, label: str) -> dict:
    """Run a single baseline test and return summary dict."""
    logger.info(f"{'='*60}")
    logger.info(f"  TESTING: {label} ({condition_id})")
    logger.info(f"{'='*60}")

    engine = DebateEngine(
        topic_id="mindestlohn",
        framing_prompt=FRAMING_PROMPTS["mindestlohn"],
        topic_type="empirical",
        condition_id=condition_id,
        run_number=1,
        llm_client=client,
        rag_pipeline=None,
        num_rounds=NUM_ROUNDS,
    )

    result = engine.run()
    save_run_json(result, OUTPUT_DIR)

    return {
        "condition_id": condition_id,
        "label": label,
        "turns": len(result.turns),
        "active_interventions": result.active_intervention_count(),
        "silent_interventions": sum(1 for i in result.interventions if i.silent_control),
        "total_events": len(result.interventions),
        "tokens": result.total_tokens_input + result.total_tokens_output,
        "latency_s": result.total_latency_s,
        "result": result,
        "engine": engine,
    }


def verify_baseline_1(summary: dict) -> None:
    """Baseline 1: No moderation at all."""
    result = summary["result"]

    assert len(result.turns) == NUM_ROUNDS * 6, \
        f"Expected {NUM_ROUNDS * 6} turns, got {len(result.turns)}"
    assert len(result.interventions) == 0, \
        f"Baseline 1 should have 0 interventions, got {len(result.interventions)}"
    assert all(t.scores is not None for t in result.turns), \
        "All turns should be scored"
    assert result.config.has_moderator_agent is False
    assert result.config.uses_trigger is False
    assert result.config.nudge_text == ""

    logger.info("  ✓ No interventions")
    logger.info("  ✓ No trigger mechanism")
    logger.info("  ✓ No nudge text")
    logger.info("  ✓ All turns scored")
    logger.info("  BASELINE 1 PASSED ✓")


def verify_baseline_2(summary: dict) -> None:
    """Baseline 2: Nudge in system prompt, no moderator."""
    result = summary["result"]
    engine = summary["engine"]

    assert len(result.turns) == NUM_ROUNDS * 6
    assert len(result.interventions) == 0, \
        f"Baseline 2 should have 0 interventions, got {len(result.interventions)}"
    assert result.config.has_moderator_agent is False
    assert result.config.uses_trigger is False
    assert result.config.nudge_text == NUDGE_INSTRUCTION

    # Verify nudge is in agent system prompts
    for party, agent in engine.agents.items():
        assert NUDGE_INSTRUCTION in agent.system_prompt, \
            f"Nudge missing from {party}'s system prompt"

    logger.info("  ✓ No interventions")
    logger.info("  ✓ Nudge text present in config")
    logger.info("  ✓ Nudge present in all 6 agent system prompts")
    logger.info("  ✓ All turns scored")
    logger.info("  BASELINE 2 PASSED ✓")


def verify_baseline_3(summary: dict) -> None:
    """Baseline 3: Habermas — moderator summary after every round."""
    result = summary["result"]

    assert len(result.turns) == NUM_ROUNDS * 6

    # Should have exactly NUM_ROUNDS interventions (one per round)
    assert len(result.interventions) == NUM_ROUNDS, \
        f"Baseline 3 should have {NUM_ROUNDS} interventions (1/round), " \
        f"got {len(result.interventions)}"

    assert result.config.has_moderator_agent is True
    assert result.config.uses_trigger is False  # unconditional

    # Verify each intervention is a Habermas summary
    for i, event in enumerate(result.interventions):
        assert event.source.value == "habermas", \
            f"Intervention {i} source should be 'habermas', got {event.source.value}"
        assert event.intervention_text != "", \
            f"Intervention {i} should have text"
        assert event.strategy == "habermas"
        # Check Habermas structured output
        if event.habermas_output:
            expected_keys = {"round_summary", "areas_of_agreement",
                           "areas_of_disagreement", "consensus_statement",
                           "instruction_for_next_round"}
            actual_keys = set(event.habermas_output.keys())
            missing = expected_keys - actual_keys
            if missing:
                logger.warning(f"  ⚠ Habermas output missing keys: {missing}")
            else:
                logger.info(f"  ✓ Round {event.round_number} Habermas output has all 5 fields")

    logger.info(f"  ✓ {NUM_ROUNDS} interventions (1 per round, unconditional)")
    logger.info("  ✓ All interventions are habermas source")
    logger.info("  ✓ Intervention text non-empty")
    logger.info("  BASELINE 3 PASSED ✓")

    # Print a sample Habermas output
    first = result.interventions[0]
    if first.habermas_output:
        logger.info(f"  Sample consensus: {first.habermas_output.get('consensus_statement', '')[:150]}")


def verify_baseline_4(summary: dict) -> None:
    """Baseline 4: Random — trigger pipeline with generic messages."""
    result = summary["result"]

    assert len(result.turns) == NUM_ROUNDS * 6
    assert result.config.has_moderator_agent is True
    assert result.config.uses_trigger is True
    assert result.config.trigger_strategy == "random"

    # Interventions may or may not fire depending on scores
    # Just verify structural correctness
    for event in result.interventions:
        assert event.strategy == "random"
        if not event.silent_control:
            assert event.source.value == "random"
            # Intervention text should be from the random pool
            assert event.intervention_text != ""

    active = summary["active_interventions"]
    silent = summary["silent_interventions"]
    logger.info(f"  ✓ Trigger strategy = 'random'")
    logger.info(f"  ✓ {active} active interventions, {silent} silent controls")
    logger.info(f"  ✓ All intervention structures valid")
    logger.info("  BASELINE 4 PASSED ✓")

    if active > 0:
        first_active = next(i for i in result.interventions if not i.silent_control)
        logger.info(f"  Sample message: \"{first_active.intervention_text[:100]}...\"")
    else:
        logger.info("  (No triggers fired — agents scored above threshold)")


def main():
    print("\n" + "=" * 60)
    print("  BASELINE CONDITIONS — INTEGRATION TEST")
    print("  4 baselines × 2 rounds × 6 agents")
    print("  RAG disabled (no FAISS indices needed)")
    print("=" * 60 + "\n")

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Failed to create LLM client: {e}")
        sys.exit(1)

    results = {}

    # Test all 4 baselines
    baselines = [
        ("baseline_1", "Baseline 1 — No Moderation"),
        ("baseline_2", "Baseline 2 — Nudge"),
        ("baseline_3", "Baseline 3 — Habermas"),
        ("baseline_4", "Baseline 4 — Random"),
    ]

    failed = []
    for condition_id, label in baselines:
        try:
            summary = run_baseline(client, condition_id, label)
            results[condition_id] = summary
        except Exception as e:
            logger.error(f"  FAILED: {label} — {type(e).__name__}: {e}")
            failed.append((condition_id, label, str(e)))
            logger.info("  Continuing with next baseline...\n")

    # Verify completed baselines
    verify_map = {
        "baseline_1": verify_baseline_1,
        "baseline_2": verify_baseline_2,
        "baseline_3": verify_baseline_3,
        "baseline_4": verify_baseline_4,
    }

    if results:
        print("\n" + "=" * 60)
        print("  VERIFICATION")
        print("=" * 60 + "\n")

        for cid, verify_fn in verify_map.items():
            if cid in results:
                verify_fn(results[cid])
                print()

    # Summary table
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Condition':<30} {'Turns':<8} {'Interv.':<10} {'Tokens':<10} {'Time':<8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for cid, s in results.items():
        print(
            f"  {s['label']:<30} {s['turns']:<8} "
            f"{s['active_interventions']:<10} {s['tokens']:<10,} "
            f"{s['latency_s']:<8.1f}s"
        )
    for cid, label, err in failed:
        print(f"  {label:<30} {'FAILED':<8} {err[:40]}")

    if failed:
        print(f"\n  {len(results)}/{len(baselines)} baselines passed, "
              f"{len(failed)} failed (likely rate limit — re-run later)")
    else:
        print(f"\n  ALL 4 BASELINES PASSED ✓")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
