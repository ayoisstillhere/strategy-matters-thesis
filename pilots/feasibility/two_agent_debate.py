"""
Feasibility Test: Minimal 2-Agent Debate (CDU vs SPD)
=====================================================
Goal: Verify that LLM agents can conduct a multi-round debate where
argumentation evolves and agents respond to each other (not parallel monologues).

Setup:
- 2 agents: CDU/CSU and SPD
- Topic: Minimum wage (Mindestlohn) - should it be raised to €15/hour?
- 3 rounds of alternating responses
- No RAG, no moderation, no judge — just raw debate
- Uses Groq API (free) with llama-3.1-8b-instant

Usage:
    pip install openai python-dotenv
    python two_agent_debate.py
"""

import os
import json
import httpx
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# --- Configuration ---
MODEL = "llama-3.1-8b-instant"  # Free on Groq, good for feasibility
API_BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")

# For OpenAPI:
# MODEL = "gpt-4o-mini"
# API_BASE = "https://api.openai.com/v1"
# API_KEY = os.getenv("OPENAI_API_KEY")

NUM_ROUNDS = 3
TOPIC = "The current federal minimum wage in Germany is €12.82/hour. Should it be raised to €15/hour?"

# --- Agent System Prompts ---
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
- Argue substantively with policy reasoning, not personal attacks"""

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
- Argue substantively with policy reasoning, not personal attacks"""

# --- Framing Prompt (neutral, given to both agents as the opening context) ---
FRAMING_PROMPT = f"""Topic: {TOPIC}

This is a structured political debate. Each party presents their position and directly engages with the opposing party's arguments. Respond concisely (3-4 sentences) and always reference what the other party said."""


def create_client():
    """Create the API client."""
    if not API_KEY:
        raise ValueError(
            f"API key not found. Please set GROQ_API_KEY (or OPENAI_API_KEY) in .env file.\n"
            f"Looking for .env at: {Path(__file__).resolve().parents[2] / '.env'}"
        )
    # return OpenAI(base_url=API_BASE, api_key=API_KEY)
    # Use custom httpx client to bypass SSL verification (TU Dresden network proxy)
    http_client = httpx.Client(verify=False)
    return OpenAI(base_url=API_BASE, api_key=API_KEY, http_client=http_client)


def generate_turn(client, system_prompt, conversation_history, agent_name):
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


def run_debate():
    """Run a 2-agent, 3-round debate and log the transcript."""
    client = create_client()

    print("=" * 70)
    print("FEASIBILITY TEST: 2-Agent Debate (CDU/CSU vs SPD)")
    print(f"Model: {MODEL}")
    print(f"Topic: {TOPIC}")
    print(f"Rounds: {NUM_ROUNDS}")
    print("=" * 70)

    # Shared conversation history (both agents see all previous turns)
    conversation_history = [{"role": "user", "content": FRAMING_PROMPT}]

    # Structured log for later analysis
    transcript = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "topic": TOPIC,
            "num_rounds": NUM_ROUNDS,
            "agents": ["CDU/CSU", "SPD"],
        },
        "rounds": [],
    }

    total_tokens = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num}")
        print(f"{'─' * 70}")

        round_data = {"round": round_num, "turns": []}

        # --- CDU/CSU Turn ---
        cdu_reply, cdu_tokens = generate_turn(
            client, CDU_SYSTEM_PROMPT, conversation_history, "CDU/CSU"
        )
        total_tokens += cdu_tokens

        print(f"\n[CDU/CSU]:\n{cdu_reply}")

        # Add CDU turn to shared history (as "assistant" from CDU's perspective,
        # but we label it clearly so SPD knows who said it)
        conversation_history.append(
            {"role": "assistant", "content": f"[CDU/CSU]: {cdu_reply}"}
        )
        # For SPD's next turn, the CDU response appears as context
        # We swap it to "user" role so SPD sees it as input to respond to
        spd_history = conversation_history.copy()
        spd_history[-1] = {"role": "user",
                           "content": f"[CDU/CSU]: {cdu_reply}"}

        round_data["turns"].append(
            {"agent": "CDU/CSU", "text": cdu_reply, "tokens": cdu_tokens}
        )

        # --- SPD Turn ---
        spd_reply, spd_tokens = generate_turn(
            client, SPD_SYSTEM_PROMPT, spd_history, "SPD"
        )
        total_tokens += spd_tokens

        print(f"\n[SPD]:\n{spd_reply}")

        # Add SPD turn to shared history
        conversation_history.append(
            {"role": "user", "content": f"[SPD]: {spd_reply}"}
        )

        round_data["turns"].append(
            {"agent": "SPD", "text": spd_reply, "tokens": spd_tokens}
        )

        transcript["rounds"].append(round_data)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("DEBATE COMPLETE")
    print(f"Total tokens used: {total_tokens}")
    print(f"{'=' * 70}")

    transcript["metadata"]["total_tokens"] = total_tokens

    # Save transcript to JSON
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / \
        f"debate_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved to: {output_file}")

    # --- Quick Analysis ---
    print(f"\n{'─' * 70}")
    print("QUICK ANALYSIS: Does argumentation evolve?")
    print(f"{'─' * 70}")
    print("\nCheck the following in the transcript above:")
    print("1. Do agents REFERENCE each other's specific claims?")
    print("2. Do arguments become more specific over rounds (not just repeating)?")
    print("3. Do agents introduce NEW points in response to challenges?")
    print("4. Is there back-and-forth engagement (not parallel monologues)?")
    print("\nIf YES to most → design is feasible for multi-round debate.")
    print("If NO → may need stronger prompts or a different model.")

    return transcript


if __name__ == "__main__":
    run_debate()
