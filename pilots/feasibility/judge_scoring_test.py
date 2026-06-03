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
    user_prompt = build_judge_prompt(turn_text, agent_name, round_num, phase, context_turns)

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
    transcripts = sorted(output_dir.glob("moderator_test_*.json"), reverse=True)

    if not transcripts:
        raise FileNotFoundError(
            "No moderator test transcript found. Run moderator_injection_test.py first."
        )

    latest = transcripts[0]
    print(f"Loading transcript: {latest.name}")

    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def run_judge_test():
    """Score all turns in a debate transcript using the LLM judge."""
    client = create_client()
    transcript = load_transcript()

    print("=" * 70)
    print("FEASIBILITY TEST: LLM-as-a-Judge Scoring")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Scoring transcript with {len(transcript['rounds'])} rounds")
    print("=" * 70)

    all_scores = []
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

            print(f"\n  Scoring [{agent}]...", end=" ")

            result, tokens, error = score_turn(
                client, text, agent, round_num, phase, context_turns
            )
            total_tokens += tokens

            if error:
                print(f"❌ {error}")
                parse_errors += 1
                score_entry = {
                    "round": round_num,
                    "phase": phase,
                    "agent": agent,
                    "error": error,
                }
            else:
                scores = result["scores"]
                avg = sum(scores.values()) / len(scores)
                print(f"✓ avg={avg:.1f}  "
                      f"[civ={scores['civility']} rel={scores['relevance']} "
                      f"log={scores['logical_consistency']} arg={scores['argument_strength']} "
                      f"doc={scores['document_grounding']} res={scores['responsiveness']} "
                      f"sta={scores['stance_differentiation']}]")

                score_entry = {
                    "round": round_num,
                    "phase": phase,
                    "agent": agent,
                    "scores": scores,
                    "reasoning": result.get("reasoning", {}),
                }

            all_scores.append(score_entry)
            context_turns.append({"agent": agent, "text": text})

    # ── ANALYSIS ──
    print(f"\n{'=' * 70}")
    print("SCORING COMPLETE")
    print(f"Total judge tokens: {total_tokens}")
    print(f"Parse errors: {parse_errors}/{len(all_scores)}")
    print(f"{'=' * 70}")

    # Compute averages by phase
    pre_scores = [s for s in all_scores if s.get("phase") == "pre-intervention" and "scores" in s]
    post_scores = [s for s in all_scores if s.get("phase") == "post-intervention" and "scores" in s]

    dimensions = ["civility", "relevance", "logical_consistency", "argument_strength",
                  "document_grounding", "responsiveness", "stance_differentiation"]

    if pre_scores and post_scores:
        print(f"\n{'─' * 70}")
        print("DIMENSION AVERAGES: Pre-Intervention vs Post-Intervention")
        print(f"{'─' * 70}")
        print(f"{'Dimension':<25} {'Pre':>6} {'Post':>6} {'Delta':>7}")
        print(f"{'─' * 25} {'─' * 6} {'─' * 6} {'─' * 7}")

        for dim in dimensions:
            pre_avg = sum(s["scores"][dim] for s in pre_scores) / len(pre_scores)
            post_avg = sum(s["scores"][dim] for s in post_scores) / len(post_scores)
            delta = post_avg - pre_avg
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{dim:<25} {pre_avg:>5.1f}  {post_avg:>5.1f}  {delta:>+5.1f} {arrow}")

    # Compute averages by agent
    cdu_scores = [s for s in all_scores if s.get("agent") == "CDU/CSU" and "scores" in s]
    spd_scores = [s for s in all_scores if s.get("agent") == "SPD" and "scores" in s]

    if cdu_scores and spd_scores:
        print(f"\n{'─' * 70}")
        print("DIMENSION AVERAGES: CDU/CSU vs SPD")
        print(f"{'─' * 70}")
        print(f"{'Dimension':<25} {'CDU':>6} {'SPD':>6}")
        print(f"{'─' * 25} {'─' * 6} {'─' * 6}")

        for dim in dimensions:
            cdu_avg = sum(s["scores"][dim] for s in cdu_scores) / len(cdu_scores)
            spd_avg = sum(s["scores"][dim] for s in spd_scores) / len(spd_scores)
            print(f"{dim:<25} {cdu_avg:>5.1f}  {spd_avg:>5.1f}")

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"judge_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "judge_model": JUDGE_MODEL,
            "source_transcript": str(sorted(output_dir.glob("moderator_test_*.json"), reverse=True)[0].name),
            "total_tokens": total_tokens,
            "parse_errors": parse_errors,
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
    print(f"   → {len(all_scores) - parse_errors}/{len(all_scores)} successful parses")
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
