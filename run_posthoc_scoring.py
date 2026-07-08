"""
Post-hoc evaluation scorer
============================
Re-scores ALL turns from all completed debate runs using the EvaluationJudge.
The judge receives only: topic, round, phase, agent name, turn text,
and preceding turns — NO condition info. This is the primary data source
for all RQ analyses.

Output:
  runs/posthoc_scores/
    posthoc_scores.csv          — flat CSV, one row per turn (for analysis)
    checkpoint.json             — tracks completed run files
    posthoc_scoring_<ts>.log    — log file

Usage:
  .venv\\Scripts\\python.exe run_posthoc_scoring.py
  .venv\\Scripts\\python.exe run_posthoc_scoring.py --dry-run
  .venv\\Scripts\\python.exe run_posthoc_scoring.py --max-runs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Path setup ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.judge import EvaluationJudge
from src.models import Turn, DimensionScores
from src.prompts.judge_prompts import DIMENSIONS

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = PROJECT_ROOT / "runs" / "posthoc_scores"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
CSV_FILE = OUTPUT_DIR / "posthoc_scores.csv"
MAX_RETRIES = 3
DELAY_BETWEEN_RUNS_S = 0.5  # short delay between runs (turns have built-in API latency)

# Source directories
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "experiment"
ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation_turn_order"

CSV_HEADER = [
    "run_file", "run_id", "condition_id", "topic_id", "run_number",
    "round_number", "turn_in_round", "agent_name", "turn_id",
] + DIMENSIONS + ["composite"]


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if not CHECKPOINT_FILE.exists():
        return {"completed": [], "failed": []}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Discover run files
# ═══════════════════════════════════════════════════════════════════════════

def discover_run_files() -> list[Path]:
    """Find all run JSON files from experiment and ablation directories."""
    files = []
    for src_dir in [EXPERIMENT_DIR, ABLATION_DIR]:
        if src_dir.exists():
            files.extend(sorted(src_dir.rglob("run_*.json")))
    return files


# ═══════════════════════════════════════════════════════════════════════════
# Score one run
# ═══════════════════════════════════════════════════════════════════════════

def score_run(
    run_path: Path,
    judge: EvaluationJudge,
    logger: logging.Logger,
) -> list[dict]:
    """Load a run JSON and re-score every turn. Returns list of row dicts."""
    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    topic_id = config["topic_id"]
    condition_id = config["condition_id"]
    run_number = config["run_number"]
    run_id = data.get("run_id", "")
    raw_turns = data["turns"]

    # Reconstruct Turn objects for preceding context
    turns: list[Turn] = []
    for t in raw_turns:
        turns.append(Turn(
            turn_id=t["turn_id"],
            round_number=t["round_number"],
            turn_in_round=t["turn_in_round"],
            agent_name=t["agent_name"],
            text=t["text"],
        ))

    rows = []
    for idx, turn in enumerate(turns):
        preceding = turns[max(0, idx - 12):idx]  # last 12 turns as context

        scores, justifications, tok_in, tok_out, latency = judge.score_turn(
            turn=turn,
            topic=topic_id,
            preceding_turns=preceding,
        )

        row = {
            "run_file": str(run_path.relative_to(PROJECT_ROOT)),
            "run_id": run_id,
            "condition_id": condition_id,
            "topic_id": topic_id,
            "run_number": run_number,
            "round_number": turn.round_number,
            "turn_in_round": turn.turn_in_round,
            "agent_name": turn.agent_name,
            "turn_id": turn.turn_id,
        }
        score_dict = scores.to_dict()
        for dim in DIMENSIONS:
            row[dim] = score_dict[dim]
        row["composite"] = round(scores.composite, 3)
        rows.append(row)

    logger.info(
        f"  ✓ Scored {len(rows)} turns | "
        f"topic={topic_id} | cond={condition_id} | run={run_number}"
    )
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# CSV I/O
# ═══════════════════════════════════════════════════════════════════════════

def init_csv():
    """Create CSV with header if it doesn't exist."""
    if not CSV_FILE.exists():
        CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()


def append_rows_to_csv(rows: list[dict]):
    """Append scored rows to the CSV file."""
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation: re-score all turns with the evaluation judge"
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs without scoring")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap number of runs")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_RUNS_S,
                        help="Seconds between runs")
    args = parser.parse_args()

    # ── Setup output & logging ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"posthoc_scoring_{ts_str}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("posthoc")

    # ── Discover run files ──
    all_files = discover_run_files()
    checkpoint = load_checkpoint()
    completed_keys = set(checkpoint["completed"])
    failed_keys = set(checkpoint["failed"])

    pending = [f for f in all_files
               if str(f.relative_to(PROJECT_ROOT)) not in completed_keys]

    if args.max_runs:
        pending = pending[:args.max_runs]

    # ── Summary ──
    total_turns_est = len(pending) * 60  # 60 turns per run
    time_est_h = total_turns_est * 1.3 / 3600

    print("\n" + "=" * 70)
    print("  POST-HOC EVALUATION SCORER")
    print("=" * 70)
    print(f"  Total run files:  {len(all_files)}")
    print(f"  Already scored:   {len(completed_keys)}")
    print(f"  Pending:          {len(pending)}")
    print(f"  Previously failed:{len(failed_keys)}")
    print(f"  Est. turns:       ~{total_turns_est}")
    print(f"  Est. time:        ~{time_est_h:.1f}h (at ~1.3s/turn)")
    print(f"  Output CSV:       {CSV_FILE}")
    print(f"  Checkpoint:       {CHECKPOINT_FILE}")
    print(f"  Log file:         {log_file}")
    if args.max_runs:
        print(f"  Max runs cap:     {args.max_runs}")
    print("=" * 70)

    if args.dry_run:
        print("\n  [DRY RUN] Pending run files:")
        for i, f in enumerate(pending, 1):
            print(f"    {i:3d}. {f.relative_to(PROJECT_ROOT)}")
        print(f"\n  Total: {len(pending)} runs would be scored.")
        return

    if not pending:
        print("\n  All runs already scored! Nothing to do.")
        return

    # ── Initialise ──
    init_csv()
    llm_client = LLMClient()
    judge = EvaluationJudge(llm_client)

    run_times: list[float] = []
    start_time = time.time()

    for idx, run_path in enumerate(pending, 1):
        rel_path = str(run_path.relative_to(PROJECT_ROOT))
        logger.info(f"[{idx}/{len(pending)}] Scoring: {rel_path}")

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
                rows = score_run(run_path, judge, logger)
                append_rows_to_csv(rows)
                run_duration = time.time() - run_start
                run_times.append(run_duration)
                success = True
                break
            except KeyboardInterrupt:
                logger.warning("\n  ⚠ Interrupted by user. Saving checkpoint...")
                save_checkpoint(checkpoint)
                print(f"\n  Checkpoint saved. {len(checkpoint['completed'])} runs scored.")
                print(f"  Resume by re-running: python run_posthoc_scoring.py")
                sys.exit(0)
            except Exception as e:
                logger.error(f"  ✗ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt * 5
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"  ✗ PERMANENTLY FAILED after {MAX_RETRIES} attempts")
                    logger.error(traceback.format_exc())

        # Update checkpoint
        if success:
            checkpoint["completed"].append(rel_path)
        else:
            if rel_path not in checkpoint["failed"]:
                checkpoint["failed"].append(rel_path)

        save_checkpoint(checkpoint)

        # Delay between runs
        if idx < len(pending) and args.delay > 0:
            time.sleep(args.delay)

    # ── Final summary ──
    elapsed = time.time() - start_time
    n_done = sum(1 for f in pending
                 if str(f.relative_to(PROJECT_ROOT)) in set(checkpoint["completed"]))
    n_failed = len(pending) - n_done

    print("\n" + "=" * 70)
    print("  POST-HOC SCORING COMPLETE")
    print("=" * 70)
    print(f"  Runs scored:     {n_done}")
    print(f"  Runs failed:     {n_failed}")
    print(f"  Total time:      {elapsed/60:.1f} min")
    if run_times:
        print(f"  Avg time/run:    {sum(run_times)/len(run_times):.1f}s")
    print(f"  Output CSV:      {CSV_FILE}")
    print(f"  Checkpoint:      {CHECKPOINT_FILE}")
    print("=" * 70)

    if n_failed:
        print(f"\n  ⚠ {n_failed} runs failed. Re-run the script to retry them,")
        print(f"    or check the log: {log_file}")


if __name__ == "__main__":
    main()
