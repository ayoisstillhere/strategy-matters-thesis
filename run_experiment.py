"""
Full Experiment Runner
========================
Runs all 160 debate runs (4 topics × 8 conditions × 5 runs each).

Features:
  - Checkpoint-based resume: completed runs are logged in a JSON file;
    re-running the script skips already-finished debates.
  - Per-run retry: up to MAX_RETRIES attempts before marking a run as failed.
  - Rate-limit delay between runs (configurable).
  - File + console logging.
  - Progress tracking with ETA.

Usage:
    cd strategy-matters-thesis
    python run_experiment.py                  # full experiment
    python run_experiment.py --dry-run        # show matrix, don't execute
    python run_experiment.py --max-runs 5     # cap at N runs (for testing)
    python run_experiment.py --condition strategy_a  # filter by condition
    python run_experiment.py --topic mindestlohn     # filter by topic

Requires:
    - GROQ_API_KEY in .env or environment
    - FAISS indices built (data/embeddings/*.faiss)
    - pip install pydantic groq httpx python-dotenv sentence-transformers faiss-cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.experiment_config import (
    CONDITIONS,
    FRAMING_PROMPTS,
    TOPIC_TYPES,
    TOPICS,
    RUNS_PER_CELL,
    get_experiment_matrix,
)
from src.export import save_run_json
from src.prompts.agent_prompts import TURN_ORDER
from src.rag_pipeline import RAGPipeline

REVERSED_TURN_ORDER = list(reversed(TURN_ORDER))


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = PROJECT_ROOT / "runs" / "experiment"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
NUM_ROUNDS = 10
LANGUAGE = "de"
MAX_RETRIES = 3
DELAY_BETWEEN_RUNS_S = 1.0  # paid tier — minimal delay, just avoids burst spikes


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint file. Returns dict with 'completed' and 'failed' sets."""
    if not checkpoint_file.exists():
        return {"completed": [], "failed": []}
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint: dict, checkpoint_file: Path) -> None:
    """Save checkpoint file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def run_key(entry: dict) -> str:
    """Unique key for a run: condition|topic|run_number."""
    return f"{entry['condition_id']}|{entry['topic']}|{entry['run_number']}"


# ═══════════════════════════════════════════════════════════════════════════
# Single run execution
# ═══════════════════════════════════════════════════════════════════════════

def execute_single_run(
    entry: dict,
    llm_client: LLMClient,
    logger: logging.Logger,
    output_dir: Path = OUTPUT_DIR,
    turn_order: list[str] | None = None,
) -> Path:
    """Run a single debate and save to disk. Returns path to saved JSON."""
    topic_id = entry["topic"]
    condition_id = entry["condition_id"]
    run_number = entry["run_number"]

    # Fresh RAG pipeline per run
    rag = RAGPipeline()

    engine = DebateEngine(
        topic_id=topic_id,
        framing_prompt=FRAMING_PROMPTS[topic_id],
        topic_type=TOPIC_TYPES[topic_id],
        condition_id=condition_id,
        run_number=run_number,
        llm_client=llm_client,
        rag_pipeline=rag,
        num_rounds=NUM_ROUNDS,
        language=LANGUAGE,
        turn_order=turn_order,
    )

    result = engine.run()
    path = save_run_json(result, output_dir)

    logger.info(
        f"  ✓ Saved: {path.relative_to(PROJECT_ROOT)} | "
        f"turns={len(result.turns)} | "
        f"interventions={result.active_intervention_count()} | "
        f"tokens={result.total_tokens_input + result.total_tokens_output:,} | "
        f"latency={result.total_latency_s:.1f}s"
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment (160 debates)")
    parser.add_argument("--dry-run", action="store_true", help="Print matrix without executing")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap number of runs (for testing)")
    parser.add_argument("--condition", type=str, default=None, help="Filter by condition_id")
    parser.add_argument("--topic", type=str, default=None, help="Filter by topic_id")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_RUNS_S, help="Seconds between runs")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG (for quick testing)")
    parser.add_argument(
        "--reverse-order", action="store_true",
        help="Speaking-order ablation: 5 runs with reversed turn order "
             "(1 topic, 1 condition). Uses --topic/--condition or defaults "
             "to baseline_1 + mindestlohn.",
    )
    args = parser.parse_args()

    # ── Resolve output dir (must happen before logging setup) ──
    active_turn_order = None  # None = default TURN_ORDER
    if args.reverse_order:
        active_turn_order = REVERSED_TURN_ORDER
        abl_topic = args.topic or "mindestlohn"
        abl_cond = args.condition or "baseline_1"
        cond = CONDITIONS[abl_cond]
        matrix = [
            {
                "topic": abl_topic,
                "condition_id": abl_cond,
                "condition_label": cond.label,
                "condition_type": cond.condition_type.value,
                "run_number": r,
                "framing_prompt": FRAMING_PROMPTS[abl_topic],
                "topic_type": TOPIC_TYPES[abl_topic],
                "has_moderator_agent": cond.has_moderator_agent,
                "uses_trigger": cond.uses_trigger,
                "trigger_strategy": cond.trigger_strategy,
            }
            for r in range(1, RUNS_PER_CELL + 1)
        ]
        # Separate output dir + checkpoint for ablation
        output_dir = PROJECT_ROOT / "runs" / "ablation_turn_order"
        checkpoint_file = output_dir / "checkpoint.json"
    else:
        output_dir = OUTPUT_DIR
        checkpoint_file = CHECKPOINT_FILE
        # ── Build run matrix ──
        matrix = get_experiment_matrix()

    # ── Setup output & logging ──
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"experiment_{ts_str}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("experiment")

    if args.reverse_order:
        logger.info(
            f"ABLATION MODE: reversed turn order "
            f"({' → '.join(REVERSED_TURN_ORDER)})"
        )

    # Apply filters (no-op for ablation since matrix is pre-built)
    if not args.reverse_order:
        if args.condition:
            matrix = [e for e in matrix if e["condition_id"] == args.condition]
        if args.topic:
            matrix = [e for e in matrix if e["topic"] == args.topic]

    # Load checkpoint — skip completed runs
    checkpoint = load_checkpoint(checkpoint_file)
    completed_keys = set(checkpoint["completed"])
    failed_keys = set(checkpoint["failed"])

    pending = [e for e in matrix if run_key(e) not in completed_keys]

    if args.max_runs:
        pending = pending[:args.max_runs]

    # ── Summary ──
    total_in_matrix = len(matrix)
    already_done = total_in_matrix - len([e for e in matrix if run_key(e) not in completed_keys])

    title = "SPEAKING-ORDER ABLATION" if args.reverse_order else "FULL EXPERIMENT RUNNER"
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    if args.reverse_order:
        print(f"  Turn order:       {' → '.join(REVERSED_TURN_ORDER)}")
    print(f"  Total matrix:     {total_in_matrix} runs")
    print(f"  Already complete: {already_done}")
    print(f"  Pending:          {len(pending)}")
    print(f"  Previously failed:{len(failed_keys)}")
    print(f"  Rounds/debate:    {NUM_ROUNDS}")
    print(f"  Language:          {LANGUAGE}")
    print(f"  Delay between:    {args.delay}s")
    print(f"  Output:           {output_dir}")
    print(f"  Checkpoint:       {checkpoint_file}")
    print(f"  Log file:         {log_file}")
    if args.condition:
        print(f"  Filter (cond):    {args.condition}")
    if args.topic:
        print(f"  Filter (topic):   {args.topic}")
    if args.max_runs:
        print(f"  Max runs cap:     {args.max_runs}")
    if args.no_rag:
        print(f"  RAG:              DISABLED (testing mode)")
    print("=" * 70)

    if args.dry_run:
        print("\n  [DRY RUN] Pending runs:")
        for i, e in enumerate(pending, 1):
            print(f"    {i:3d}. {e['condition_id']:12s} | {e['topic']:18s} | run {e['run_number']}")
        print(f"\n  Total: {len(pending)} runs would be executed.")
        return

    if not pending:
        print("\n  All runs complete! Nothing to do.")
        return

    # ── Execute ──
    llm_client = LLMClient()
    run_times: list[float] = []
    start_time = time.time()

    for idx, entry in enumerate(pending, 1):
        key = run_key(entry)
        logger.info(
            f"[{idx}/{len(pending)}] "
            f"{entry['condition_id']} | {entry['topic']} | run {entry['run_number']}"
        )

        # ETA
        if run_times:
            avg_time = sum(run_times) / len(run_times)
            remaining = (len(pending) - idx + 1) * avg_time
            eta_min = remaining / 60
            logger.info(f"  ETA: ~{eta_min:.0f} min ({avg_time:.0f}s/run avg)")

        # Retry loop
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                run_start = time.time()
                execute_single_run(entry, llm_client, logger, output_dir=output_dir, turn_order=active_turn_order)
                run_duration = time.time() - run_start
                run_times.append(run_duration)
                success = True
                break
            except KeyboardInterrupt:
                logger.warning("\n  ⚠ Interrupted by user. Saving checkpoint...")
                save_checkpoint(checkpoint, checkpoint_file)
                print(f"\n  Checkpoint saved. {len(checkpoint['completed'])} runs complete.")
                print(f"  Resume by re-running: python run_experiment.py")
                sys.exit(0)
            except Exception as e:
                logger.error(
                    f"  ✗ Attempt {attempt}/{MAX_RETRIES} failed: {e}"
                )
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt * 5  # exponential backoff: 10s, 20s, 40s
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"  ✗ PERMANENTLY FAILED after {MAX_RETRIES} attempts")
                    logger.error(traceback.format_exc())

        # Update checkpoint
        if success:
            checkpoint["completed"].append(key)
        else:
            if key not in checkpoint["failed"]:
                checkpoint["failed"].append(key)

        save_checkpoint(checkpoint, checkpoint_file)

        # Delay between runs (skip after last)
        if idx < len(pending) and args.delay > 0:
            time.sleep(args.delay)

    # ── Final summary ──
    elapsed = time.time() - start_time
    n_done = len([e for e in pending if run_key(e) in set(checkpoint["completed"])])
    n_failed = len(pending) - n_done

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Runs completed:  {n_done}")
    print(f"  Runs failed:     {n_failed}")
    print(f"  Total time:      {elapsed/60:.1f} min")
    if run_times:
        print(f"  Avg time/run:    {sum(run_times)/len(run_times):.1f}s")
    print(f"  Output:          {output_dir}")
    print(f"  Checkpoint:      {checkpoint_file}")
    print("=" * 70)

    if n_failed:
        print(f"\n  ⚠ {n_failed} runs failed. Re-run the script to retry them,")
        print(f"    or check the log: {log_file}")


if __name__ == "__main__":
    main()
