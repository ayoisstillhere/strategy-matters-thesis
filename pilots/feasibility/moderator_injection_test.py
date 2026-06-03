"""
Feasibility Test: Moderator Intervention Injection
===================================================
Goal: Verify that injecting a moderator intervention (reframing strategy)
between rounds changes agent behavior — specifically, do agents become
more responsive and engage with the opponent's strongest argument?

Setup:
- Same 2-agent debate as previous feasibility test (CDU vs SPD) on minimum wage
- 4 rounds total: 2 rounds free debate → moderator intervention → 2 more rounds
- Compare rounds 1-2 (pre-intervention) vs rounds 3-4 (post-intervention)
- Uses Groq API (free) with llama-3.1-8b-instant

Usage:
    python moderator_injection_test.py
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
MODEL = "llama-3.1-8b-instant"
API_BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")

# For OpenAI:
# MODEL = "gpt-4o-mini"
# API_BASE = "https://api.openai.com/v1"
# API_KEY = os.getenv("OPENAI_API_KEY")

NUM_ROUNDS_BEFORE = 2
NUM_ROUNDS_AFTER = 2
TOPIC = "The current federal minimum wage in Germany is €12.82/hour. Should it be raised to €15/hour?"

# --- Agent System Prompts (same as two_agent_debate.py) ---
CDU_SYSTEM_PROMPT = """You are a political debate agent representing the CDU/CSU (Christlich Demokratische Union / Christlich-Soziale Union) in a German political debate.

Your core positions:
- You support a market-oriented economic policy
- Wage-setting should be handled by the Mindestlohnkommission (minimum wage commission) based on economic data, not by political mandate
- You are concerned about job losses in economically weaker regions (eastern Länder) if wages rise too fast
- You prioritize small and medium business (Mittelstand) viability
- You believe in the social market economy (Soziale Marktwirtschaft) but resist heavy state intervention in wages

Debate rules:
- Keep responses to 3-4 sentences maximum
- You MUST directly address and respond to arguments made by the opposing party (SPD) in their previous turn
- Name the specific claim you are responding to before making your counter-argument
- Stay in character as CDU/CSU throughout
- Argue substantively with policy reasoning, not personal attacks
- When a moderator intervention is provided, follow its instructions carefully in your next response"""

SPD_SYSTEM_PROMPT = """You are a political debate agent representing the SPD (Sozialdemokratische Partei Deutschlands) in a German political debate.

Your core positions:
- A €15 minimum wage is a matter of social justice and worker dignity
- The current minimum wage is insufficient for a dignified life, especially in high-cost cities
- You have historically fought for fair wages and workers' rights
- You believe the state has a responsibility to ensure living wages
- You argue that higher wages boost domestic consumption and reduce inequality

Debate rules:
- Keep responses to 3-4 sentences maximum
- You MUST directly address and respond to arguments made by the opposing party (CDU/CSU) in their previous turn
- Name the specific claim you are responding to before making your counter-argument
- Stay in character as SPD throughout
- Argue substantively with policy reasoning, not personal attacks
- When a moderator intervention is provided, follow its instructions carefully in your next response"""

FRAMING_PROMPT = f"""Topic: {TOPIC}

This is a structured political debate. Each party presents their position and directly engages with the opposing party's arguments. Respond concisely (3-4 sentences) and always reference what the other party said."""

# --- Moderator Intervention (Reframing Strategy) ---
# This mimics Strategy B from the exposé: reframing is triggered when
# responsiveness drops, and the moderator restates the core disagreement
# in neutral terms and directs each party to engage with the strongest
# opposing argument.
MODERATOR_INTERVENTION = """[MODERATOR INTERVENTION — Reframing]:
The discussion so far has focused on competing claims about regional economic effects.

CDU/CSU has argued that a €15 minimum wage risks job losses in economically weaker eastern Länder and that the Mindestlohnkommission's data-driven process should be preserved.

SPD has argued that higher wages reduce poverty and inequality and that economic growth in eastern Germany has not been harmed by past minimum wage increases.

For the next round, each party should DIRECTLY ADDRESS the STRONGEST argument made by the opposing party:
- CDU/CSU: Address the SPD's claim that past minimum wage increases did not harm eastern German economic growth.
- SPD: Address the CDU/CSU's concern about the Mindestlohnkommission's independence and data-driven wage-setting.

Do not repeat your previous points. Engage specifically with the argument assigned to you."""


def create_client():
    """Create the API client."""
    if not API_KEY:
        raise ValueError(
            f"API key not found. Please set GROQ_API_KEY (or OPENAI_API_KEY) in .env file.\n"
            f"Looking for .env at: {Path(__file__).resolve().parents[2] / '.env'}"
        )
    http_client = httpx.Client(verify=False)
    return OpenAI(base_url=API_BASE, api_key=API_KEY, http_client=http_client)


def generate_turn(client, system_prompt, conversation_history):
    """Generate a single agent turn given the conversation history."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens if response.usage else 0
    return reply, tokens_used


def run_debate_with_intervention():
    """Run a debate with a moderator intervention injected after round 2."""
    client = create_client()

    print("=" * 70)
    print("FEASIBILITY TEST: Moderator Intervention (Reframing)")
    print(f"Model: {MODEL}")
    print(f"Structure: {NUM_ROUNDS_BEFORE} rounds → INTERVENTION → {NUM_ROUNDS_AFTER} rounds")
    print("=" * 70)

    conversation_history = [{"role": "user", "content": FRAMING_PROMPT}]

    transcript = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "topic": TOPIC,
            "rounds_before_intervention": NUM_ROUNDS_BEFORE,
            "rounds_after_intervention": NUM_ROUNDS_AFTER,
            "strategy": "reframing",
            "agents": ["CDU/CSU", "SPD"],
        },
        "rounds": [],
        "intervention": None,
    }

    total_tokens = 0

    # ── PRE-INTERVENTION ROUNDS ──
    print(f"\n{'━' * 70}")
    print("  PRE-INTERVENTION PHASE")
    print(f"{'━' * 70}")

    for round_num in range(1, NUM_ROUNDS_BEFORE + 1):
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num}")
        print(f"{'─' * 70}")

        round_data = {"round": round_num, "phase": "pre-intervention", "turns": []}

        # CDU/CSU turn
        cdu_reply, cdu_tokens = generate_turn(client, CDU_SYSTEM_PROMPT, conversation_history)
        total_tokens += cdu_tokens
        print(f"\n[CDU/CSU]:\n{cdu_reply}")

        conversation_history.append({"role": "assistant", "content": f"[CDU/CSU]: {cdu_reply}"})
        spd_history = conversation_history.copy()
        spd_history[-1] = {"role": "user", "content": f"[CDU/CSU]: {cdu_reply}"}

        round_data["turns"].append({"agent": "CDU/CSU", "text": cdu_reply, "tokens": cdu_tokens})

        # SPD turn
        spd_reply, spd_tokens = generate_turn(client, SPD_SYSTEM_PROMPT, spd_history)
        total_tokens += spd_tokens
        print(f"\n[SPD]:\n{spd_reply}")

        conversation_history.append({"role": "user", "content": f"[SPD]: {spd_reply}"})
        round_data["turns"].append({"agent": "SPD", "text": spd_reply, "tokens": spd_tokens})

        transcript["rounds"].append(round_data)

    # ── MODERATOR INTERVENTION ──
    print(f"\n{'━' * 70}")
    print("  ⚡ MODERATOR INTERVENTION (Reframing)")
    print(f"{'━' * 70}")
    print(f"\n{MODERATOR_INTERVENTION}")

    # Inject the intervention into conversation history so both agents see it
    conversation_history.append({"role": "user", "content": MODERATOR_INTERVENTION})

    transcript["intervention"] = {
        "after_round": NUM_ROUNDS_BEFORE,
        "strategy": "reframing",
        "text": MODERATOR_INTERVENTION,
    }

    # ── POST-INTERVENTION ROUNDS ──
    print(f"\n{'━' * 70}")
    print("  POST-INTERVENTION PHASE")
    print(f"{'━' * 70}")

    for round_num in range(NUM_ROUNDS_BEFORE + 1, NUM_ROUNDS_BEFORE + NUM_ROUNDS_AFTER + 1):
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num}")
        print(f"{'─' * 70}")

        round_data = {"round": round_num, "phase": "post-intervention", "turns": []}

        # CDU/CSU turn
        cdu_reply, cdu_tokens = generate_turn(client, CDU_SYSTEM_PROMPT, conversation_history)
        total_tokens += cdu_tokens
        print(f"\n[CDU/CSU]:\n{cdu_reply}")

        conversation_history.append({"role": "assistant", "content": f"[CDU/CSU]: {cdu_reply}"})
        spd_history = conversation_history.copy()
        spd_history[-1] = {"role": "user", "content": f"[CDU/CSU]: {cdu_reply}"}

        round_data["turns"].append({"agent": "CDU/CSU", "text": cdu_reply, "tokens": cdu_tokens})

        # SPD turn
        spd_reply, spd_tokens = generate_turn(client, SPD_SYSTEM_PROMPT, spd_history)
        total_tokens += spd_tokens
        print(f"\n[SPD]:\n{spd_reply}")

        conversation_history.append({"role": "user", "content": f"[SPD]: {spd_reply}"})
        round_data["turns"].append({"agent": "SPD", "text": spd_reply, "tokens": spd_tokens})

        transcript["rounds"].append(round_data)

    # ── SUMMARY ──
    print(f"\n{'=' * 70}")
    print("DEBATE COMPLETE")
    print(f"Total tokens used: {total_tokens}")
    print(f"{'=' * 70}")

    transcript["metadata"]["total_tokens"] = total_tokens

    # Save transcript
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"moderator_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved to: {output_file}")

    # ── ANALYSIS GUIDE ──
    print(f"\n{'─' * 70}")
    print("ANALYSIS: Did the moderator intervention change agent behavior?")
    print(f"{'─' * 70}")
    print("\nCompare PRE-INTERVENTION (rounds 1-2) vs POST-INTERVENTION (rounds 3-4):")
    print()
    print("1. RESPONSIVENESS: Did agents address the specific argument the")
    print("   moderator assigned them, or did they ignore the instruction?")
    print("   - CDU should address SPD's eastern Germany growth claim")
    print("   - SPD should address CDU's Mindestlohnkommission independence concern")
    print()
    print("2. NEW ARGUMENTS: Did agents introduce new points they hadn't")
    print("   raised before, or just repeat pre-intervention talking points?")
    print()
    print("3. ENGAGEMENT DEPTH: Are post-intervention responses more")
    print("   specific and substantive than pre-intervention ones?")
    print()
    print("4. INSTRUCTION FOLLOWING: Did agents follow the moderator's")
    print("   specific directive, or did they treat it as general advice?")
    print()
    print("If YES to most → moderator interventions can steer the debate.")
    print("If NO → may need stronger moderator prompts or a different model.")

    return transcript


if __name__ == "__main__":
    run_debate_with_intervention()
