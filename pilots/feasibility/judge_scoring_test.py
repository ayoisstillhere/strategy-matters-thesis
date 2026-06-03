"""
Feasibility Test: LLM-as-a-Judge Scoring
=========================================
Goal: Verify that an LLM judge can produce structured, per-dimension scores
for debate turns, and that scores are consistent and differentiated across
dimensions and agents.

Setup:
- Uses the moderator intervention transcript as input
- Judge LLM (llama-3.3-70b-versatile on Groq) scores each turn on 7 dimensions
- Scores are 1-5 scale per dimension
- Tests whether the judge produces valid JSON, differentiates between dimensions,
  and whether scores change meaningfully after the moderator intervention

The 7 evaluation dimensions:
1. Civility — absence of personal attacks, insults, inflammatory language
2. Relevance — stays on topic and addresses the debate question
3. Logical consistency — arguments are internally coherent, no contradictions
4. Argument strength — claims are well-supported with reasoning/evidence
5. Document-grounding — faithfulness to party programme positions (N/A without RAG)
6. Responsiveness — directly engages with the other party's prior arguments
7. Stance differentiation — maintains a distinct party position, doesn't drift to center

Usage:
    python judge_scoring_test.py
"""

import os
import json
import httpx
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# --- Configuration ---
# Use a DIFFERENT (stronger) model for the judge than for the agents
# This avoids self-evaluation bias
JUDGE_MODEL = "llama-3.3-70b-versatile"  # Free on Groq, strong reasoning
API_BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")

# For OpenAI judge:
# JUDGE_MODEL = "gpt-4o"
# API_BASE = "https://api.openai.com/v1"
# API_KEY = os.getenv("OPENAI_API_KEY")

NUM_RERUNS = 3  # Score each turn this many times to test intra-judge consistency

# --- Judge System Prompt ---
JUDGE_SYSTEM_PROMPT = """You are an impartial discourse quality judge evaluating political debate exchanges.

You will be given a single debate turn (one agent's response) along with the conversation context that preceded it.

Score the turn on EACH of the following 7 dimensions using a 1-5 scale:

1. **Civility** (1=hostile/insulting, 5=fully respectful and professional)
   - Are there personal attacks, inflammatory language, or disrespectful framing?

2. **Relevance** (1=completely off-topic, 5=directly addresses the debate question)
   - Does the response stay focused on the topic and advance the discussion?

3. **Logical Consistency** (1=self-contradictory, 5=internally coherent argument)
   - Are the claims logically structured without internal contradictions?

4. **Argument Strength** (1=unsupported assertions, 5=well-evidenced claims)
   - Does the agent provide reasoning, evidence, or examples to support claims?

5. **Document-Grounding** (1=fabricates positions, 5=faithful to known party positions)
   - Does the agent's stated position align with the known positions of their party?
   - Note: In this pilot test without RAG, judge based on general knowledge of German party positions.

6. **Responsiveness** (1=ignores opponent, 5=directly engages opponent's specific claims)
   - Does the agent name and address what the other party actually said?
   - Does it respond to the SPECIFIC argument, not a strawman?

7. **Stance Differentiation** (1=indistinguishable from opponent, 5=clearly distinct position)
   - Does the agent maintain a unique party identity and perspective?
   - Would you be able to tell which party this is without the label?

IMPORTANT RULES:
- You MUST respond with ONLY valid JSON, no other text.
- Use this exact format:
{
  "scores": {
    "civility": <int 1-5>,
    "relevance": <int 1-5>,
    "logical_consistency": <int 1-5>,
    "argument_strength": <int 1-5>,
    "document_grounding": <int 1-5>,
    "responsiveness": <int 1-5>,
    "stance_differentiation": <int 1-5>
  },
  "reasoning": {
    "civility": "<1 sentence justification>",
    "relevance": "<1 sentence justification>",
    "logical_consistency": "<1 sentence justification>",
    "argument_strength": "<1 sentence justification>",
    "document_grounding": "<1 sentence justification>",
    "responsiveness": "<1 sentence justification>",
    "stance_differentiation": "<1 sentence justification>"
  }
}"""


def create_client():
    """Create the API client."""
    if not API_KEY:
        raise ValueError(
            f"API key not found. Please set GROQ_API_KEY (or OPENAI_API_KEY) in .env file.\n"
            f"Looking for .env at: {Path(__file__).resolve().parents[2] / '.env'}"
        )
    http_client = httpx.Client(verify=False)
    return OpenAI(base_url=API_BASE, api_key=API_KEY, http_client=http_client)


def build_judge_prompt(turn_text, agent_name, round_num, phase, context_turns):
    """Build the user prompt for the judge, including conversation context."""
    context_str = ""
    if context_turns:
        context_str = "PRIOR CONVERSATION CONTEXT:\n"
        for t in context_turns:
            context_str += f"[{t['agent']}]: {t['text']}\n\n"
        context_str += "---\n\n"

    prompt = f"""{context_str}TURN TO EVALUATE:
Agent: {agent_name}
Round: {round_num}
Phase: {phase}

"{turn_text}"

Score this turn on all 7 dimensions. Respond with ONLY valid JSON."""

    return prompt


def score_turn(client, turn_text, agent_name, round_num, phase, context_turns):
    """Have the judge score a single debate turn."""
    user_prompt = build_judge_prompt(
        turn_text, agent_name, round_num, phase, context_turns)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # Low temperature for consistent scoring
        max_tokens=500,
    )

    raw_output = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens if response.usage else 0

    # Try to parse JSON
    try:
        # Handle case where model wraps JSON in markdown code block
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        result = json.loads(raw_output)
        return result, tokens_used, None
    except json.JSONDecodeError as e:
        return None, tokens_used, f"JSON parse error: {e}\nRaw output: {raw_output[:200]}"


def load_transcript():
    """Load the most recent moderator intervention transcript."""
    output_dir = Path(__file__).parent / "outputs"
    transcripts = sorted(output_dir.glob(
        "moderator_test_*.json"), reverse=True)

    if not transcripts:
        raise FileNotFoundError(
            "No moderator test transcript found. Run moderator_injection_test.py first."
        )

    latest = transcripts[0]
    print(f"Loading transcript: {latest.name}")

    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def run_judge_test():
    """Score all turns in a debate transcript using the LLM judge, with reruns for consistency."""
    client = create_client()
    transcript = load_transcript()

    print("=" * 70)
    print("FEASIBILITY TEST: LLM-as-a-Judge Scoring")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Scoring transcript with {len(transcript['rounds'])} rounds")
    print(f"Reruns per turn: {NUM_RERUNS}")
    print("=" * 70)

    all_scores = []  # Each entry has a "reruns" list
    total_tokens = 0
    parse_errors = 0
    context_turns = []  # Accumulate turns for context

    for round_data in transcript["rounds"]:
        round_num = round_data["round"]
        phase = round_data["phase"]

        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num} ({phase})")
        print(f"{'─' * 70}")

        for turn in round_data["turns"]:
            agent = turn["agent"]
            text = turn["text"]

            print(f"\n  Scoring [{agent}] x{NUM_RERUNS}...", end=" ")

            rerun_results = []
            for run_idx in range(NUM_RERUNS):
                result, tokens, error = score_turn(
                    client, text, agent, round_num, phase, context_turns
                )
                total_tokens += tokens

                if error:
                    parse_errors += 1
                    rerun_results.append({"error": error})
                else:
                    rerun_results.append(result["scores"])

            # Report summary for this turn
            valid_runs = [r for r in rerun_results if isinstance(
                r, dict) and "error" not in r]
            if valid_runs:
                # Compute mean scores across reruns
                dimensions = list(valid_runs[0].keys())
                mean_scores = {}
                for dim in dimensions:
                    vals = [r[dim] for r in valid_runs]
                    mean_scores[dim] = sum(vals) / len(vals)

                avg = sum(mean_scores.values()) / len(mean_scores)
                print(f"✓ avg={avg:.1f}  ", end="")

                # Show per-run scores compactly
                for i, r in enumerate(valid_runs):
                    scores_str = "/".join(str(r[d]) for d in dimensions)
                    print(f"run{i+1}=[{scores_str}] ", end="")
                print()
            else:
                print(f"❌ All {NUM_RERUNS} runs failed to parse")

            score_entry = {
                "round": round_num,
                "phase": phase,
                "agent": agent,
                "reruns": rerun_results,
            }

            all_scores.append(score_entry)
            context_turns.append({"agent": agent, "text": text})

    # ── ANALYSIS ──
    print(f"\n{'=' * 70}")
    print("SCORING COMPLETE")
    print(f"Total judge tokens: {total_tokens}")
    print(f"Parse errors: {parse_errors}/{len(all_scores) * NUM_RERUNS}")
    print(f"{'=' * 70}")

    dimensions = ["civility", "relevance", "logical_consistency", "argument_strength",
                  "document_grounding", "responsiveness", "stance_differentiation"]

    # ── CONSISTENCY ANALYSIS ──
    print(f"\n{'─' * 70}")
    print(f"INTRA-JUDGE CONSISTENCY (across {NUM_RERUNS} reruns per turn)")
    print(f"{'─' * 70}")

    total_comparisons = 0
    exact_matches = 0
    within_one_matches = 0
    max_deviations = {dim: 0 for dim in dimensions}

    for entry in all_scores:
        valid_runs = [r for r in entry["reruns"]
                      if isinstance(r, dict) and "error" not in r]
        if len(valid_runs) < 2:
            continue

        for dim in dimensions:
            vals = [r[dim] for r in valid_runs]
            deviation = max(vals) - min(vals)
            max_deviations[dim] = max(max_deviations[dim], deviation)

            # Pairwise comparisons
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    total_comparisons += 1
                    if vals[i] == vals[j]:
                        exact_matches += 1
                        within_one_matches += 1
                    elif abs(vals[i] - vals[j]) <= 1:
                        within_one_matches += 1

    if total_comparisons > 0:
        print(
            f"\n  Exact agreement:  {exact_matches}/{total_comparisons} ({100*exact_matches/total_comparisons:.0f}%)")
        print(
            f"  Within ±1:       {within_one_matches}/{total_comparisons} ({100*within_one_matches/total_comparisons:.0f}%)")
        print(f"\n  Max deviation per dimension:")
        for dim in dimensions:
            status = "✓" if max_deviations[dim] <= 1 else "⚠"
            print(f"    {status} {dim:<25} max Δ = {max_deviations[dim]}")

    # ── PHASE COMPARISON (using mean of reruns) ──
    def get_mean_scores(entry):
        valid = [r for r in entry["reruns"]
                 if isinstance(r, dict) and "error" not in r]
        if not valid:
            return None
        return {dim: sum(r[dim] for r in valid) / len(valid) for dim in dimensions}

    pre_entries = [e for e in all_scores if e["phase"] == "pre-intervention"]
    post_entries = [e for e in all_scores if e["phase"] == "post-intervention"]
    pre_means = [get_mean_scores(e) for e in pre_entries if get_mean_scores(e)]
    post_means = [get_mean_scores(e)
                  for e in post_entries if get_mean_scores(e)]

    if pre_means and post_means:
        print(f"\n{'─' * 70}")
        print("DIMENSION AVERAGES: Pre-Intervention vs Post-Intervention (mean of reruns)")
        print(f"{'─' * 70}")
        print(f"{'Dimension':<25} {'Pre':>6} {'Post':>6} {'Delta':>7}")
        print(f"{'─' * 25} {'─' * 6} {'─' * 6} {'─' * 7}")

        for dim in dimensions:
            pre_avg = sum(s[dim] for s in pre_means) / len(pre_means)
            post_avg = sum(s[dim] for s in post_means) / len(post_means)
            delta = post_avg - pre_avg
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(
                f"{dim:<25} {pre_avg:>5.1f}  {post_avg:>5.1f}  {delta:>+5.1f} {arrow}")

    # ── AGENT COMPARISON ──
    cdu_entries = [e for e in all_scores if e["agent"] == "CDU/CSU"]
    spd_entries = [e for e in all_scores if e["agent"] == "SPD"]
    cdu_means = [get_mean_scores(e) for e in cdu_entries if get_mean_scores(e)]
    spd_means = [get_mean_scores(e) for e in spd_entries if get_mean_scores(e)]

    if cdu_means and spd_means:
        print(f"\n{'─' * 70}")
        print("DIMENSION AVERAGES: CDU/CSU vs SPD (mean of reruns)")
        print(f"{'─' * 70}")
        print(f"{'Dimension':<25} {'CDU':>6} {'SPD':>6}")
        print(f"{'─' * 25} {'─' * 6} {'─' * 6}")

        for dim in dimensions:
            cdu_avg = sum(s[dim] for s in cdu_means) / len(cdu_means)
            spd_avg = sum(s[dim] for s in spd_means) / len(spd_means)
            print(f"{dim:<25} {cdu_avg:>5.1f}  {spd_avg:>5.1f}")

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / \
        f"judge_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "judge_model": JUDGE_MODEL,
            "source_transcript": str(sorted(output_dir.glob("moderator_test_*.json"), reverse=True)[0].name),
            "num_reruns": NUM_RERUNS,
            "total_tokens": total_tokens,
            "parse_errors": parse_errors,
            "exact_agreement_pct": round(100 * exact_matches / total_comparisons, 1) if total_comparisons > 0 else None,
            "within_one_pct": round(100 * within_one_matches / total_comparisons, 1) if total_comparisons > 0 else None,
        },
        "scores": all_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # ── ANALYSIS GUIDE ──
    print(f"\n{'─' * 70}")
    print("ANALYSIS: Is the LLM judge viable?")
    print(f"{'─' * 70}")
    print()
    print("1. VALID JSON: Did the judge produce parseable JSON for all turns?")
    print(
        f"   → {len(all_scores) - parse_errors}/{len(all_scores)} successful parses")
    print()
    print("2. SCORE DIFFERENTIATION: Are scores different across dimensions?")
    print("   (If all 5s everywhere, the judge isn't discriminating)")
    print()
    print("3. RESPONSIVENESS DELTA: Does responsiveness increase post-intervention?")
    print("   (This is the key signal — the moderator aimed to improve this)")
    print()
    print("4. AGENT DIFFERENTIATION: Do CDU and SPD get different stance_differentiation")
    print("   scores, or are they treated identically?")
    print()
    print("5. REASONING QUALITY: Are the 1-sentence justifications sensible?")
    print()
    print("If YES to most → LLM-as-a-judge is viable for the full experiment.")
    print("If NO → try GPT-4o as judge, or refine the scoring prompt.")

    return results


if __name__ == "__main__":
    run_judge_test()
