"""
Judge Replacement Pilot
========================
llama-3.3-70b-versatile is being deprecated by Groq (decommission: Aug 16 2026).
This script compares candidate replacement models for the judge role by scoring
the same set of turns with each model and reporting:

  1. JSON validity rate  — must be 100% to be usable
  2. Per-dimension mean scores — should be comparable to baseline
  3. Intra-model exact agreement — rescore same turns 2× at temperature=0
  4. Latency per turn

Candidates:
  - llama-3.3-70b-versatile  (current baseline, being deprecated)
  - openai/gpt-oss-120b      (Groq recommended replacement A)
  - qwen/qwen3.6-27b         (Groq recommended replacement B)
  - qwen/qwen3-32b           (bonus candidate — larger Qwen3)

Turn sampling:
  18 turns from existing runs (3 per party). Same turns scored by all models.

Decision rule:
  - Eliminate any model with JSON validity < 100%
  - From remaining: prefer model with scores closest to baseline
    (smallest mean absolute deviation across all dimensions)
  - Tie-break: higher intra-model agreement, then lower latency

Usage:
    cd strategy-matters-thesis
    .venv\\Scripts\\python.exe pilots/judge_replacement.py

Output:
  runs/judge_replacement/comparison_<timestamp>.json
  Console: per-model comparison table + recommended replacement
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.judge import EvaluationJudge
from src.models import Turn, DimensionScores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("judge_replacement")

OUTPUT_DIR = PROJECT_ROOT / "runs" / "judge_replacement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Round 1 findings (comparison_20260630_100909.json):
#   openai/gpt-oss-120b : 97.2% valid but 23.5% intra-agreement at temp=0
#     → ELIMINATED: non-deterministic at temperature=0, unusable as judge
#   qwen/qwen3.6-27b    : 0% valid — <think> block exhausts 1800-token budget
#     → thinking must be disabled; retry with /no_think token
#   qwen/qwen3-32b      : 100% valid but 5.5% intra-agreement
#     → thinking non-determinism; retry with /no_think token
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",                    # baseline (deprecated Aug 16 2026)
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama4 MoE, non-thinking
    "qwen/qwen3-32b",                              # Qwen3 32B with /no_think
    "qwen/qwen3.6-27b",                            # Qwen3.6 27B with /no_think
]

# Qwen3 models are reasoning/thinking models that produce <think>...</think>
# chain-of-thought blocks by default, making them non-deterministic at temp=0
# and exhausting the token budget before JSON is produced.
# Fix: prepend /no_think to the user message → direct JSON output, no CoT.
THINKING_MODELS = {"qwen/qwen3-32b", "qwen/qwen3.6-27b"}

# Per-model token budget:
#   qwen/qwen3.6-27b: /no_think is ignored by this model — it still produces
#     <think>...</think> blocks. For the full 7-dim judge prompt the thinking
#     runs ~1500-2000 tokens before JSON. Give it 3000 so </think> completes
#     and _parse_json can strip it. Other models need only 1200.
PILOT_MAX_TOKENS: dict[str, int] = {
    "llama-3.3-70b-versatile":                   1200,
    "meta-llama/llama-4-scout-17b-16e-instruct": 1200,
    "qwen/qwen3-32b":                            1200,  # /no_think works, no CoT
    "qwen/qwen3.6-27b":                          3000,  # /no_think ignored, needs CoT budget
}
PILOT_JUDGE_MAX_TOKENS = 1200  # fallback

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation",
]

PARTIES = ["CDU/CSU", "SPD", "Bündnis 90/Die Grünen", "FDP", "Die Linke", "AfD"]
TURNS_PER_PARTY = 3   # 3 × 6 = 18 turns total
RERUNS = 2            # rescore each turn 2× for intra-model agreement


# ---------------------------------------------------------------------------
# Sampling (reuses same logic as judge_reliability.py)
# ---------------------------------------------------------------------------

def sample_turns(num_per_party: int = TURNS_PER_PARTY) -> list[dict]:
    """Sample turns from existing run JSONs, 3 per party."""
    all_run_files = sorted(
        (PROJECT_ROOT / "runs").rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

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
            if not turn.get("text"):
                continue
            turn["_topic"] = topic
            idx = turns.index(turn)
            turn["_preceding"] = turns[max(0, idx - 6):idx]
            party_turns[party].append(turn)
        if all(len(v) >= num_per_party for v in party_turns.values()):
            break

    sampled = []
    for turns in party_turns.values():
        sampled.extend(turns[:num_per_party])
    logger.info(f"Sampled {len(sampled)} turns across {len(PARTIES)} parties")
    return sampled


def dict_to_turn(raw: dict) -> Turn:
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
# Score one turn with a given judge model, return (scores_dict, latency, valid)
# ---------------------------------------------------------------------------

def score_one(
    judge: EvaluationJudge,
    raw_turn: dict,
) -> tuple[dict[str, int] | None, float]:
    """Returns (score_dict_or_None, latency_seconds).
    Returns None on hard API failure OR on JSON parse failure
    (detected by checking whether the llm_client warning was emitted,
    i.e. parsed_json is None in the underlying response).
    """
    turn = dict_to_turn(raw_turn)
    preceding = dicts_to_turns(raw_turn.get("_preceding", []))
    topic = raw_turn.get("_topic", "mindestlohn")

    # Patch: call the llm_client directly to get parsed_json visibility
    import warnings as _w
    from src.prompts.judge_prompts import format_judge_user_prompt, get_judge_system_prompt

    phase = (
        "opening (rounds 1-3)" if turn.round_number <= 3
        else "mid-debate (rounds 4-7)" if turn.round_number <= 7
        else "closing (rounds 8-10)"
    )
    preceding_text = judge._format_preceding(preceding)
    user_prompt = format_judge_user_prompt(
        topic=topic,
        round_number=turn.round_number,
        phase=phase,
        agent_name=turn.agent_name,
        turn_text=turn.text,
        preceding_turns=preceding_text,
    )

    # Disable thinking for Qwen3 models — /no_think must appear in the user
    # message to switch the model out of chain-of-thought mode.  Without this,
    # <think> blocks exhaust the token budget before JSON is produced.
    effective_user_prompt = user_prompt
    if judge.model in THINKING_MODELS:
        effective_user_prompt = "/no_think\n" + user_prompt

    t0 = time.time()
    try:
        resp = judge.llm_client.complete(
            model=judge.model,
            system_prompt=get_judge_system_prompt(),
            user_prompt=effective_user_prompt,
            max_tokens=judge.max_tokens,
            temperature=judge.temperature,
            parse_json=True,
        )
        latency = time.time() - t0
        if resp.parsed_json is None:
            # JSON parse failed — return None so validity is counted correctly
            return None, latency
        scores, _ = judge._parse_response(resp.parsed_json, turn)
        return scores.to_dict(), latency
    except Exception as exc:
        latency = time.time() - t0
        logger.warning(f"Judge call failed: {exc}")
        return None, latency


# ---------------------------------------------------------------------------
# Evaluate one model across all sampled turns (RERUNS times each)
# ---------------------------------------------------------------------------

def evaluate_model(
    model_id: str,
    client: LLMClient,
    sampled_turns: list[dict],
) -> dict:
    """
    Returns a result dict with:
      - model: str
      - json_valid_rate: float  (actual parse successes, not fallback-3 count)
      - per_dim_means: dict[str, float]  (only from successful parses)
      - intra_agreement_exact: float   (rerun 1 == rerun 2)
      - mean_latency: float
      - raw: list of per-turn rerun results
    """
    max_tok = PILOT_MAX_TOKENS.get(model_id, PILOT_JUDGE_MAX_TOKENS)
    judge = EvaluationJudge(
        llm_client=client, model=model_id, temperature=0.0,
        max_tokens=max_tok,
    )
    logger.info(f"  max_tokens={max_tok} for {model_id}")
    logger.info(f"  Evaluating model: {model_id}")

    valid_count = 0
    total = 0
    latencies: list[float] = []
    dim_scores: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    agreement_checks: list[bool] = []
    raw_records = []

    for i, raw_turn in enumerate(sampled_turns, 1):
        party = raw_turn.get("agent_name", "?")
        rnd   = raw_turn.get("round_number", "?")
        logger.info(f"    Turn {i:>2}/{len(sampled_turns)} — {party} R{rnd} | {model_id}")

        reruns_scores = []
        reruns_latencies = []

        for _ in range(RERUNS):
            scores_dict, latency = score_one(judge, raw_turn)
            reruns_scores.append(scores_dict)
            reruns_latencies.append(latency)
            total += 1
            latencies.append(latency)

        # score_one returns None if parsed_json was None (JSON parse failed)
        # or if the API call raised an exception. This gives accurate validity.
        valid_this_turn = [s for s in reruns_scores if s is not None]
        valid_count += len(valid_this_turn)

        # Collect valid scores
        valid_reruns = valid_this_turn
        if valid_reruns:
            for dim in DIMENSIONS:
                dim_scores[dim].extend(s[dim] for s in valid_reruns)

        # Intra-model agreement: all reruns identical?
        if len(valid_reruns) == RERUNS:
            all_same = all(
                valid_reruns[0][dim] == valid_reruns[j][dim]
                for dim in DIMENSIONS
                for j in range(1, RERUNS)
            )
            agreement_checks.append(all_same)

        raw_records.append({
            "turn_id": raw_turn.get("turn_id"),
            "agent_name": party,
            "round": rnd,
            "reruns": reruns_scores,
            "latencies": reruns_latencies,
        })

    per_dim_means = {
        d: round(mean(dim_scores[d]), 3) if dim_scores[d] else 0.0
        for d in DIMENSIONS
    }
    composite = round(mean(per_dim_means.values()), 3)

    return {
        "model": model_id,
        "json_valid_rate": valid_count / total if total else 0.0,
        "per_dim_means": per_dim_means,
        "composite_mean": composite,
        "intra_agreement_exact": mean(agreement_checks) if agreement_checks else 0.0,
        "mean_latency_s": round(mean(latencies), 2),
        "raw": raw_records,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # File logging
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"run_{ts_str}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s", "%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    print("\n" + "=" * 70)
    print("  JUDGE REPLACEMENT PILOT")
    print(f"  Baseline: llama-3.3-70b-versatile (deprecated Aug 16 2026)")
    print(f"  Candidates: {len(CANDIDATE_MODELS)} models")
    print(f"  Turns: {TURNS_PER_PARTY * len(PARTIES)} ({TURNS_PER_PARTY}/party × {len(PARTIES)} parties)")
    print(f"  Reruns per turn: {RERUNS} (intra-model agreement)")
    print(f"  Pilot max_tokens: {PILOT_JUDGE_MAX_TOKENS}")
    print(f"  Round 1 eliminated: openai/gpt-oss-120b (non-deterministic at temp=0)")
    print(f"  Fix: /no_think token for Qwen3 models, Llama4-Scout as new candidate")
    print("=" * 70)

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Set GROQ_API_KEY in .env: {e}")
        sys.exit(1)

    sampled_turns = sample_turns()
    if len(sampled_turns) < TURNS_PER_PARTY * len(PARTIES):
        logger.warning(f"Only {len(sampled_turns)} turns sampled. Results may be less reliable.")

    model_results: list[dict] = []
    for model_id in CANDIDATE_MODELS:
        result = evaluate_model(model_id, client, sampled_turns)
        model_results.append(result)

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    baseline = model_results[0]  # llama-3.3-70b-versatile

    header = f"  {'Model':<30} {'Valid%':>7} {'Composite':>10} {'IntraAgr%':>10} {'Lat(s)':>7}"
    print(header)
    print("  " + "-" * 67)

    for r in model_results:
        tag = " ← baseline" if r["model"] == baseline["model"] else ""
        print(
            f"  {r['model']:<30} {r['json_valid_rate']:>6.1%} "
            f"{r['composite_mean']:>10.3f} {r['intra_agreement_exact']:>9.1%} "
            f"{r['mean_latency_s']:>7.2f}{tag}"
        )

    # Per-dimension table
    print(f"\n  Per-dimension means (Δ = model − baseline):")
    print(f"  {'Dimension':<24}" +
          "".join(f" {m['model'].split('/')[-1][:12]:>14}" for m in model_results))
    print("  " + "-" * (24 + 14 * len(model_results)))

    for dim in DIMENSIONS:
        base_val = baseline["per_dim_means"][dim]
        row = f"  {dim:<24}"
        for r in model_results:
            val = r["per_dim_means"][dim]
            if r["model"] == baseline["model"]:
                row += f" {val:>14.2f}"
            else:
                delta = val - base_val
                sign = "+" if delta >= 0 else ""
                row += f" {val:>8.2f}({sign}{delta:.2f})"
        print(row)

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Eliminate invalid models
    valid_models = [r for r in model_results if r["json_valid_rate"] >= 1.0]
    invalid_models = [r for r in model_results if r["json_valid_rate"] < 1.0]

    if invalid_models:
        print(f"  ✗ Eliminated (JSON failures): {[r['model'] for r in invalid_models]}")

    # Exclude baseline from selection (it's being deprecated)
    candidates = [r for r in valid_models if r["model"] != baseline["model"]]

    if not candidates:
        print("  ⚠  No valid replacement found. Manual review required.")
    else:
        # Rank by MAD from baseline per-dimension means
        def mad_from_baseline(r: dict) -> float:
            return mean(
                abs(r["per_dim_means"][d] - baseline["per_dim_means"][d])
                for d in DIMENSIONS
            )

        candidates.sort(key=lambda r: (
            -r["json_valid_rate"],
            mad_from_baseline(r),
            -r["intra_agreement_exact"],
            r["mean_latency_s"],
        ))
        winner = candidates[0]

        print(f"\n  Recommended replacement: {winner['model']}")
        print(f"    JSON valid rate:    {winner['json_valid_rate']:.1%}")
        print(f"    Composite mean:     {winner['composite_mean']:.3f} "
              f"(baseline: {baseline['composite_mean']:.3f}, "
              f"Δ={winner['composite_mean']-baseline['composite_mean']:+.3f})")
        print(f"    Intra-model agree: {winner['intra_agreement_exact']:.1%}")
        print(f"    Mean latency:       {winner['mean_latency_s']:.2f}s")
        print(f"\n  To apply: update JUDGE_MODEL in src/llm_client.py:")
        print(f'    JUDGE_MODEL = "{winner["model"]}"')

    # ── Save ───────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"comparison_{ts}.json"
    out_file.write_text(
        json.dumps(
            {
                "meta": {
                    "turns_sampled": len(sampled_turns),
                    "reruns_per_turn": RERUNS,
                    "models_tested": CANDIDATE_MODELS,
                    "baseline_model": baseline["model"],
                    "deprecation_date": "2026-08-16",
                },
                "results": [
                    {k: v for k, v in r.items() if k != "raw"}
                    for r in model_results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved: {out_file}")
    print(f"\n  Full results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
