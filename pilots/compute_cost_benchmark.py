"""
Compute Cost Benchmark
=======================
Simulates the API call pattern of a single full debate run to measure
real wall-clock time including rate-limit waits. Uses realistic token
counts but dummy content.

Usage:
    python pilots/compute_cost_benchmark.py

Requires: GROQ_API_KEY in environment or .env file.
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"

NUM_ROUNDS = 10
NUM_AGENTS = 6
TURNS_PER_RUN = NUM_ROUNDS * NUM_AGENTS  # 60

# Estimated trigger fires and moderator calls per run
EST_TRIGGER_FIRES = 10
EST_MODERATOR_CALLS = 2

# Realistic dummy prompts sized to match actual token budgets
AGENT_SYSTEM_DUMMY = (
    "You are a political debate agent representing a German party. "
    "You argue based on your party's documented positions. "
) * 25  # ~700 tokens

AGENT_RULES_DUMMY = (
    "Produce 3-4 sentences per turn. Be concise. "
    "Argue with policy reasoning. No personal attacks. "
) * 8  # ~230 tokens

RAG_DUMMY = (
    "Grounding passage: The party programme states that minimum wage "
    "policy should be determined by economic evidence and social needs. "
) * 5  # ~300 tokens

TRANSCRIPT_DUMMY = (
    "CDU/CSU: We support market-based solutions. "
    "SPD: Workers deserve a living wage of at least 15 euros. "
    "Grüne: Climate and social justice must go together. "
    "FDP: Deregulation drives growth. "
    "Linke: Wealth redistribution is a necessity. "
    "AfD: National sovereignty must come first. "
) * 2  # ~600 tokens (last round)

FRAMING_DUMMY = (
    "The German federal minimum wage currently stands at 12.82 euros "
    "per hour. Discuss whether it should be raised to 15 euros. "
    "Ground your arguments in your party's documented positions."
)  # ~110 tokens

JUDGE_SYSTEM_DUMMY = (
    "You are an evaluation judge scoring political debate turns on "
    "7 dimensions: civility, relevance, logical consistency, argument "
    "strength, document-grounding, responsiveness, stance differentiation. "
    "Score each on a 1-5 scale with calibration anchors. "
) * 8  # ~800 tokens

JUDGE_TURN_DUMMY = (
    "The agent said: We support raising the minimum wage to 15 euros "
    "because our programme explicitly commits to fair wages for all "
    "workers, particularly in eastern Germany where costs are lower. "
    "This is grounded in our Wahlprogramm chapter on social policy. "
) * 3  # ~700 tokens

TRIGGER_SYSTEM_DUMMY = (
    "You are a trigger-check judge. Evaluate whether the civility "
    "dimension score is justified. Return a JSON with score and "
    "confirmation boolean. "
) * 4  # ~400 tokens

MODERATOR_SYSTEM_DUMMY = (
    "You are a neutral moderator in a structured multi-party debate. "
    "Your role is to improve discourse quality without taking sides. "
    "Produce a JSON intervention with diagnosis, target parties, "
    "intervention text, and expected behaviour. "
) * 10  # ~1000 tokens


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env")
        sys.exit(1)

    client = Groq(api_key=api_key)

    results = {
        "agent_calls": [],
        "judge_calls": [],
        "trigger_calls": [],
        "moderator_calls": [],
    }

    agent_input = AGENT_SYSTEM_DUMMY + AGENT_RULES_DUMMY + FRAMING_DUMMY + RAG_DUMMY + TRANSCRIPT_DUMMY
    judge_input = JUDGE_SYSTEM_DUMMY + JUDGE_TURN_DUMMY
    trigger_input = TRIGGER_SYSTEM_DUMMY + JUDGE_TURN_DUMMY
    moderator_input = MODERATOR_SYSTEM_DUMMY + TRANSCRIPT_DUMMY

    total_input_tokens = 0
    total_output_tokens = 0
    total_start = time.time()

    # --- Agent calls (60 turns, 8b model) ---
    print(f"Running {TURNS_PER_RUN} agent calls ({AGENT_MODEL})...")
    agent_start = time.time()
    for i in range(TURNS_PER_RUN):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": agent_input},
                {"role": "user", "content": "Produce your next debate turn."},
            ],
            max_tokens=150,
        )
        elapsed = time.time() - t0
        results["agent_calls"].append({
            "call": i + 1,
            "latency_s": round(elapsed, 3),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
        total_input_tokens += resp.usage.prompt_tokens
        total_output_tokens += resp.usage.completion_tokens
        if (i + 1) % 10 == 0:
            print(f"  Agent call {i+1}/{TURNS_PER_RUN} — {elapsed:.2f}s")
    agent_elapsed = time.time() - agent_start
    print(f"  Agent total: {agent_elapsed:.1f}s\n")

    # --- Eval judge calls (60 turns, 70b model) ---
    print(f"Running {TURNS_PER_RUN} eval judge calls ({JUDGE_MODEL})...")
    judge_start = time.time()
    for i in range(TURNS_PER_RUN):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": judge_input},
                {"role": "user", "content": "Score this turn on all 7 dimensions. Return JSON."},
            ],
            max_tokens=400,
        )
        elapsed = time.time() - t0
        results["judge_calls"].append({
            "call": i + 1,
            "latency_s": round(elapsed, 3),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
        total_input_tokens += resp.usage.prompt_tokens
        total_output_tokens += resp.usage.completion_tokens
        if (i + 1) % 10 == 0:
            print(f"  Judge call {i+1}/{TURNS_PER_RUN} — {elapsed:.2f}s")
    judge_elapsed = time.time() - judge_start
    print(f"  Judge total: {judge_elapsed:.1f}s\n")

    # --- Trigger-check calls (10 fires, 70b model) ---
    print(f"Running {EST_TRIGGER_FIRES} trigger-check calls ({JUDGE_MODEL})...")
    trigger_start = time.time()
    for i in range(EST_TRIGGER_FIRES):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": trigger_input},
                {"role": "user", "content": "Confirm or reject this trigger. Return JSON."},
            ],
            max_tokens=200,
        )
        elapsed = time.time() - t0
        results["trigger_calls"].append({
            "call": i + 1,
            "latency_s": round(elapsed, 3),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
        total_input_tokens += resp.usage.prompt_tokens
        total_output_tokens += resp.usage.completion_tokens
    trigger_elapsed = time.time() - trigger_start
    print(f"  Trigger total: {trigger_elapsed:.1f}s\n")

    # --- Moderator calls (2 interventions, 70b model) ---
    print(f"Running {EST_MODERATOR_CALLS} moderator calls ({JUDGE_MODEL})...")
    mod_start = time.time()
    for i in range(EST_MODERATOR_CALLS):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": moderator_input},
                {"role": "user", "content": "Produce your intervention as JSON."},
            ],
            max_tokens=300,
        )
        elapsed = time.time() - t0
        results["moderator_calls"].append({
            "call": i + 1,
            "latency_s": round(elapsed, 3),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
        total_input_tokens += resp.usage.prompt_tokens
        total_output_tokens += resp.usage.completion_tokens
    mod_elapsed = time.time() - mod_start
    print(f"  Moderator total: {mod_elapsed:.1f}s\n")

    # --- Summary ---
    total_elapsed = time.time() - total_start
    total_calls = TURNS_PER_RUN + TURNS_PER_RUN + EST_TRIGGER_FIRES + EST_MODERATOR_CALLS

    summary = {
        "single_run_wall_clock_s": round(total_elapsed, 1),
        "single_run_wall_clock_min": round(total_elapsed / 60, 1),
        "breakdown_s": {
            "agent_calls": round(agent_elapsed, 1),
            "judge_calls": round(judge_elapsed, 1),
            "trigger_calls": round(trigger_elapsed, 1),
            "moderator_calls": round(mod_elapsed, 1),
        },
        "total_api_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "extrapolation_160_runs": {
            "estimated_hours": round(total_elapsed * 160 / 3600, 1),
            "estimated_cost_groq_paid": {
                "8b_input": round(sum(c["input_tokens"] for c in results["agent_calls"]) * 160 / 1e6 * 0.05, 2),
                "8b_output": round(sum(c["output_tokens"] for c in results["agent_calls"]) * 160 / 1e6 * 0.08, 2),
                "70b_input": round((
                    sum(c["input_tokens"] for c in results["judge_calls"]) +
                    sum(c["input_tokens"] for c in results["trigger_calls"]) +
                    sum(c["input_tokens"] for c in results["moderator_calls"])
                ) * 160 / 1e6 * 0.59, 2),
                "70b_output": round((
                    sum(c["output_tokens"] for c in results["judge_calls"]) +
                    sum(c["output_tokens"] for c in results["trigger_calls"]) +
                    sum(c["output_tokens"] for c in results["moderator_calls"])
                ) * 160 / 1e6 * 0.79, 2),
            },
        },
    }

    # Calculate total cost
    cost = summary["extrapolation_160_runs"]["estimated_cost_groq_paid"]
    cost["total_usd"] = round(
        cost["8b_input"] + cost["8b_output"] + cost["70b_input"] + cost["70b_output"], 2
    )

    print("=" * 60)
    print("BENCHMARK RESULTS — SINGLE RUN SIMULATION")
    print("=" * 60)
    print(f"Wall clock time:  {summary['single_run_wall_clock_s']}s ({summary['single_run_wall_clock_min']} min)")
    print(f"  Agent ({AGENT_MODEL}):   {summary['breakdown_s']['agent_calls']}s")
    print(f"  Judge ({JUDGE_MODEL}):  {summary['breakdown_s']['judge_calls']}s")
    print(f"  Trigger checks:       {summary['breakdown_s']['trigger_calls']}s")
    print(f"  Moderator:            {summary['breakdown_s']['moderator_calls']}s")
    print(f"Total API calls:  {summary['total_api_calls']}")
    print(f"Total tokens:     {summary['total_tokens']:,} (in: {summary['total_input_tokens']:,}, out: {summary['total_output_tokens']:,})")
    print()
    print("EXTRAPOLATION TO 160 RUNS:")
    print(f"  Estimated time:  {summary['extrapolation_160_runs']['estimated_hours']} hours")
    print(f"  Estimated cost (Groq paid): ${cost['total_usd']}")
    print(f"    8b:  ${cost['8b_input'] + cost['8b_output']:.2f}")
    print(f"    70b: ${cost['70b_input'] + cost['70b_output']:.2f}")

    # Save full results
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compute_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw_calls": results}, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
