"""
Model Selection Pilots — Comparison Script
============================================
Addresses the "Model Selection Pilots" step in action_plan.html.

AGENT MODEL COMPARISON
  Runs the same debate (rentenpolitik, Strategy B, 3 rounds) with 3 agent
  models available on Groq free tier, using the same judge throughout.
  Evaluates: instruction following, persona consistency, truncation rate,
  argument quality (judge scores), latency, token cost.

  Candidates:
    A) llama-3.1-8b-instant   — current selection (fast, free)
    B) llama-3.3-70b-versatile — stronger, same provider (current judge)
    C) gemma2-9b-it            — alternative architecture, free on Groq

JUDGE MODEL COMPARISON
  Scores 10 sample turns from today's runs using 2 judge models.
  Evaluates: score distribution, JSON validity, consistency.

  Candidates:
    J1) llama-3.3-70b-versatile — current selection
    J2) mixtral-8x7b-32768      — Groq alternative

Usage:
    cd strategy-matters-thesis
    python pilots/model_comparison.py

    # Agent comparison only:
    python pilots/model_comparison.py --agents-only

    # Judge comparison only:
    python pilots/model_comparison.py --judges-only

Requires:
    - GROQ_API_KEY in .env
    - FAISS indices at data/embeddings/
    - Existing run JSON in runs/demo/ for judge comparison sample turns
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.rag_pipeline import RAGPipeline
from src.experiment_config import FRAMING_PROMPTS
from src.judge import EvaluationJudge
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_comparison")

# ── Configuration ──────────────────────────────────────────────────────────

AGENT_MODELS = [
    ("llama-3.1-8b-instant",    "Llama-3.1-8B  (current)"),
    ("llama-3.3-70b-versatile", "Llama-3.3-70B (larger)"),
    ("gemma2-9b-it",            "Gemma-2-9B    (alternative)"),
]

JUDGE_MODELS = [
    ("llama-3.3-70b-versatile", "Llama-3.3-70B (current)"),
    ("mixtral-8x7b-32768",      "Mixtral-8x7B  (alternative)"),
]

DEBATE_TOPIC    = "rentenpolitik"
DEBATE_CONDITION = "strategy_b"
NUM_ROUNDS      = 3
JUDGE_MODEL     = "llama-3.3-70b-versatile"  # fixed for agent comparison

OUTPUT_DIR = PROJECT_ROOT / "runs" / "model_comparison"


# ── Agent comparison ───────────────────────────────────────────────────────

def run_agent_comparison(args) -> list[dict]:
    """Run the same debate with each agent model; return per-model stats."""
    print("\n" + "=" * 65)
    print("  AGENT MODEL COMPARISON")
    print(f"  {DEBATE_TOPIC} | {DEBATE_CONDITION} | {NUM_ROUNDS} rounds")
    print(f"  Judge (fixed): {JUDGE_MODEL}")
    print("=" * 65)

    results = []

    for model_id, model_label in AGENT_MODELS:
        print(f"\n  ── {model_label} ──")
        logger.info(f"Running debate with agent model: {model_id}")

        try:
            client = LLMClient()
            rag = RAGPipeline()

            engine = DebateEngine(
                topic_id=DEBATE_TOPIC,
                framing_prompt=FRAMING_PROMPTS[DEBATE_TOPIC],
                topic_type="empirical",
                condition_id=DEBATE_CONDITION,
                run_number=1,
                llm_client=client,
                rag_pipeline=rag,
                agent_model=model_id,
                judge_model=JUDGE_MODEL,
                num_rounds=NUM_ROUNDS,
            )

            t0 = time.time()
            result = engine.run()
            wall_time = time.time() - t0

            save_run_json(result, OUTPUT_DIR / model_id.replace("-", "_").replace(".", "_"))

            turns = result.turns
            all_scores = {}
            for t in turns:
                if t.scores:
                    for dim, val in t.scores.to_dict().items():
                        all_scores.setdefault(dim, []).append(val)

            truncated = sum(
                1 for t in turns
                if t.text and t.text.rstrip()[-1] not in ".!?\"')"
            )
            avg_latency = mean(t.latency_s for t in turns) if turns else 0
            total_tokens = result.total_tokens_input + result.total_tokens_output
            rag_used = sum(1 for t in turns if t.rag_passages_used)

            stats = {
                "model_id":    model_id,
                "label":       model_label,
                "turns":       len(turns),
                "interventions": len(result.interventions),
                "truncated":   truncated,
                "avg_scores":  {d: round(mean(v), 2) for d, v in all_scores.items()},
                "composite":   round(mean(
                    (t.scores.to_dict().get("argument_strength", 0) +
                     t.scores.to_dict().get("responsiveness", 0) +
                     t.scores.to_dict().get("stance_differentiation", 0)) / 3
                    for t in turns if t.scores
                ), 2) if turns else 0,
                "avg_latency_s": round(avg_latency, 2),
                "total_tokens":  total_tokens,
                "wall_time_s":   round(wall_time, 1),
                "rag_used":      rag_used,
                "error":         None,
            }
            results.append(stats)
            logger.info(f"  Done: {len(turns)} turns, {truncated} truncated, "
                        f"composite={stats['composite']}, {wall_time:.0f}s")

        except Exception as e:
            logger.error(f"  FAILED: {model_label} — {e}")
            results.append({
                "model_id": model_id, "label": model_label, "error": str(e),
                "turns": 0, "truncated": 0, "avg_scores": {}, "composite": 0,
                "avg_latency_s": 0, "total_tokens": 0, "wall_time_s": 0, "rag_used": 0,
            })

    _print_agent_table(results)
    return results


def _print_agent_table(results: list[dict]) -> None:
    dims = ["civility", "responsiveness", "stance_differentiation",
            "argument_strength", "document_grounding"]

    print("\n" + "=" * 65)
    print("  AGENT COMPARISON TABLE")
    print("=" * 65)
    print(f"  {'Model':<25} {'Trunc':<7} {'RAG':<6} {'Comp.':<7} {'Lat(s)':<8} {'Tokens':<10}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")
    for r in results:
        if r["error"]:
            print(f"  {r['label']:<25} FAILED: {r['error'][:30]}")
        else:
            print(
                f"  {r['label']:<25} {r['truncated']}/{r['turns']:<4} "
                f"{r['rag_used']}/{r['turns']:<3} "
                f"{r['composite']:<7} {r['avg_latency_s']:<8} {r['total_tokens']:<10,}"
            )

    print()
    print(f"  {'Model':<25}", end="")
    for d in dims:
        print(f" {d[:6]:<7}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in dims:
        print(f" {'-'*7}", end="")
    print()
    for r in results:
        if not r["error"]:
            print(f"  {r['label']:<25}", end="")
            for d in dims:
                v = r["avg_scores"].get(d, "-")
                print(f" {v:<7}", end="")
            print()
    print()


# ── Judge comparison ───────────────────────────────────────────────────────

def load_sample_turns(n: int = 10) -> list:
    """Load n sample turns from the most recent demo run JSON."""
    run_files = sorted((PROJECT_ROOT / "runs" / "demo").rglob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        logger.error("No run files found in runs/demo/. Run a debate first.")
        return []

    turns = []
    for f in run_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for t in data.get("turns", []):
            if t.get("text") and t.get("round_number", 0) > 1:
                turns.append(t)
            if len(turns) >= n:
                break
        if len(turns) >= n:
            break

    logger.info(f"Loaded {len(turns)} sample turns from {run_files[0].name}")
    return turns[:n]


def run_judge_comparison() -> None:
    """Score the same 10 turns with each judge model; compare distributions."""
    print("\n" + "=" * 65)
    print("  JUDGE MODEL COMPARISON")
    print(f"  10 sample turns scored by each candidate judge")
    print("=" * 65)

    sample_turns = load_sample_turns(10)
    if not sample_turns:
        print("  SKIPPED — no sample turns available")
        return

    from src.models import Turn, DimensionScores
    import uuid
    from datetime import datetime, timezone

    judge_results = {}

    for judge_id, judge_label in JUDGE_MODELS:
        print(f"\n  ── {judge_label} ──")
        client = LLMClient()
        judge = EvaluationJudge(client, model=judge_id)

        scores_per_dim: dict[str, list[int]] = {}
        json_failures = 0
        latencies = []

        for raw in sample_turns:
            # Reconstruct a minimal Turn object
            turn = Turn(
                turn_id=str(uuid.uuid4()),
                round_number=raw["round_number"],
                turn_in_round=raw["turn_in_round"],
                agent_name=raw["agent_name"],
                text=raw["text"],
            )
            try:
                t0 = time.time()
                scores, _, _, _, lat = judge.score_turn(
                    turn=turn,
                    topic="mindestlohn",
                    preceding_turns=[],
                )
                latencies.append(time.time() - t0)

                for dim, val in scores.to_dict().items():
                    scores_per_dim.setdefault(dim, []).append(val)

            except Exception as e:
                logger.warning(f"  Judge call failed: {e}")
                json_failures += 1

        judge_results[judge_id] = {
            "label":          judge_label,
            "json_failures":  json_failures,
            "avg_scores":     {d: round(mean(v), 2) for d, v in scores_per_dim.items()},
            "score_stdev":    {d: round(stdev(v), 2) if len(v) > 1 else 0.0
                               for d, v in scores_per_dim.items()},
            "avg_latency_s":  round(mean(latencies), 2) if latencies else 0,
        }
        logger.info(f"  Done: {json_failures} failures, avg lat {judge_results[judge_id]['avg_latency_s']}s")

    _print_judge_table(judge_results)


def _print_judge_table(results: dict) -> None:
    dims = ["civility", "relevance", "logical_consistency",
            "argument_strength", "document_grounding",
            "responsiveness", "stance_differentiation"]

    print("\n" + "=" * 65)
    print("  JUDGE COMPARISON TABLE  (avg score ± stdev over 10 turns)")
    print("=" * 65)
    print(f"  {'Model':<28} {'Fail':<6} {'Lat(s)'}")
    for r in results.values():
        print(f"  {r['label']:<28} {r['json_failures']:<6} {r['avg_latency_s']}")

    print()
    print(f"  {'Dimension':<28}", end="")
    for r in results.values():
        print(f" {r['label'][:16]:<18}", end="")
    print()
    print(f"  {'-'*28}", end="")
    for _ in results:
        print(f" {'-'*18}", end="")
    print()
    for dim in dims:
        print(f"  {dim:<28}", end="")
        for r in results.values():
            avg = r["avg_scores"].get(dim, "-")
            std = r["score_stdev"].get(dim, "-")
            print(f" {str(avg)+'±'+str(std):<18}", end="")
        print()
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model Selection Pilots")
    parser.add_argument("--agents-only", action="store_true")
    parser.add_argument("--judges-only", action="store_true")
    args = parser.parse_args()

    run_agents = not args.judges_only
    run_judges = not args.agents_only

    if run_agents:
        run_agent_comparison(args)
    if run_judges:
        run_judge_comparison()

    print("\n  Results saved to:", OUTPUT_DIR)
    print("  Use these tables for the model comparison table in Chapter 3.\n")


if __name__ == "__main__":
    main()
