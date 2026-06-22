"""
Political Agent
================
Wraps a single party agent: persona prompt loading, RAG grounding
retrieval, context window assembly, and LLM call.

Lifecycle (managed by DebateEngine):
    1. Instantiate once per party per debate run.
    2. Engine calls agent.generate_turn() each time it is the agent's turn.
    3. Agent assembles context → calls LLM → returns Turn model.

Context window assembly per turn:
    System:  persona prompt + AGENT_RULES [+ nudge if Baseline 2]
    User:    framing prompt
             + grounding block (RAG top-3)
             + transcript history (all prior turns in this debate)
             + last moderator intervention (if any)
             + turn instruction

See also:
    - src/prompts/agent_prompts.py — persona prompts + AGENT_RULES
    - src/rag_pipeline.py — RAGPipeline for grounding
    - src/models.py — Turn data model
    - src/llm_client.py — LLM API wrapper
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.llm_client import LLMClient, AGENT_MODEL
from src.models import Turn
from src.prompts.agent_prompts import get_system_prompt
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class PoliticalAgent:
    """A single political party agent in the debate.

    Args:
        party: Party name (e.g. "CDU/CSU").
        llm_client: Shared LLMClient instance.
        rag_pipeline: Shared RAGPipeline instance (already initialised).
        model: LLM model identifier.
        nudge_text: Optional nudge instruction appended to system prompt
            (non-empty for Baseline 2 only).
        max_tokens: Maximum output tokens per turn.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        party: str,
        llm_client: LLMClient,
        rag_pipeline: Optional[RAGPipeline] = None,
        model: str = AGENT_MODEL,
        nudge_text: str = "",
        max_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.party = party
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Build system prompt (persona + rules + optional nudge)
        self.system_prompt = get_system_prompt(party)
        if nudge_text:
            self.system_prompt += f"\n{nudge_text}\n"

    def generate_turn(
        self,
        framing_prompt: str,
        transcript: list[Turn],
        round_number: int,
        turn_in_round: int,
        last_intervention: str = "",
    ) -> Turn:
        """Generate one debate turn.

        Args:
            framing_prompt: The neutral framing prompt for this topic.
            transcript: All prior turns in this debate (for context).
            round_number: Current round (1-indexed).
            turn_in_round: Position within the round (1-indexed).
            last_intervention: Text of the most recent moderator
                intervention, if any. Empty string if none.

        Returns:
            A Turn object with text, token counts, and metadata.
        """
        # 1. RAG grounding
        grounding_block = ""
        rag_passages_used = []
        if self.rag_pipeline is not None:
            recent_texts = [t.text for t in transcript[-12:]]  # last 2 rounds
            passages = self.rag_pipeline.get_grounding(
                party=self.party,
                framing_prompt=framing_prompt,
                recent_turns=recent_texts,
            )
            grounding_block = RAGPipeline.format_grounding_block(passages)
            rag_passages_used = [p.text[:100] for p in passages]  # truncated for log

        # 2. Assemble user prompt
        user_prompt = self._assemble_user_prompt(
            framing_prompt=framing_prompt,
            grounding_block=grounding_block,
            transcript=transcript,
            last_intervention=last_intervention,
            round_number=round_number,
        )

        # 3. Call LLM
        resp = self.llm_client.complete(
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # 4. Build Turn
        turn = Turn(
            turn_id=str(uuid.uuid4()),
            round_number=round_number,
            turn_in_round=turn_in_round,
            agent_name=self.party,
            text=resp.text,
            rag_passages_used=rag_passages_used,
            token_count_input=resp.input_tokens,
            token_count_output=resp.output_tokens,
            latency_s=resp.latency_s,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"[Round {round_number}] {self.party}: "
            f"{resp.output_tokens} tokens, {resp.latency_s:.2f}s"
        )
        return turn

    def _assemble_user_prompt(
        self,
        framing_prompt: str,
        grounding_block: str,
        transcript: list[Turn],
        last_intervention: str,
        round_number: int,
    ) -> str:
        """Build the user-role message for this turn.

        Structure:
            1. Topic framing
            2. RAG grounding block
            3. Transcript history
            4. Last moderator intervention (if any)
            5. Turn instruction
        """
        parts = []

        # 1. Topic
        parts.append(f"## Debate Topic\n{framing_prompt}")

        # 2. Grounding
        if grounding_block:
            parts.append(f"\n{grounding_block}")

        # 3. Transcript
        if transcript:
            parts.append("\n## Debate Transcript So Far")
            for t in transcript:
                parts.append(f"**[{t.agent_name}]** (Round {t.round_number}): {t.text}")

        # 4. Moderator intervention
        if last_intervention:
            parts.append(
                f"\n## Moderator Intervention\n"
                f"The moderator has issued the following instruction to all parties:\n"
                f'"{last_intervention}"\n'
                f"Incorporate this instruction into your next response while "
                f"maintaining your party identity."
            )

        # 5. Turn instruction
        parts.append(
            f"\n## Your Turn (Round {round_number})\n"
            f"Produce your next debate contribution as {self.party}. "
            f"Follow all debate rules."
        )

        return "\n".join(parts)
