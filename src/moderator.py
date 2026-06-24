"""
AI Moderator
=============
Moderator class with dispatch for all intervention sources:

- Strategies A-D: LLM-generated interventions using strategy prompts.
- Baseline 3 (Habermas): LLM-generated consensus summaries every round.
- Baseline 4 (Random): Random selection from generic message pool.

The moderator is invoked by the DebateEngine only when a trigger fires
(strategies / Baseline 4) or unconditionally after each round (Habermas).

See also:
    - src/prompts/moderator_prompts.py — strategy prompts
    - src/experiment_config.py — Habermas prompt, random message pool
    - src/models.py — InterventionEvent
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.llm_client import LLMClient, JUDGE_MODEL
from src.models import InterventionEvent, InterventionSource, Turn
from src.prompts.moderator_prompts import (
    get_moderator_prompt,
    format_moderator_user_prompt,
)
from src.experiment_config import (
    HABERMAS_MODERATOR_SYSTEM_PROMPT,
    HABERMAS_USER_TEMPLATE,
    RANDOM_MODERATOR_MESSAGES,
)

logger = logging.getLogger(__name__)


class Moderator:
    """AI moderator for structured debate interventions.

    Args:
        llm_client: Shared LLMClient instance.
        model: LLM model for generating interventions.
        max_tokens: Maximum output tokens for interventions.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = JUDGE_MODEL,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Strategy interventions (A-D)
    # ------------------------------------------------------------------

    def generate_strategy_intervention(
        self,
        strategy: str,
        trigger_dimension: str,
        trigger_score: float,
        threshold: float,
        triggering_agent: str,
        recent_turns: list[Turn],
        round_number: int,
    ) -> InterventionEvent:
        """Generate a strategy-specific moderator intervention.

        Args:
            strategy: Strategy key (de-escalation, reframing, etc.).
            trigger_dimension: Dimension that triggered the intervention.
            trigger_score: Score that fired the trigger.
            threshold: Threshold that was breached.
            triggering_agent: Agent whose turn triggered.
            recent_turns: Recent debate turns for context.
            round_number: Current round number.

        Returns:
            InterventionEvent with moderator output.
        """
        system_prompt = get_moderator_prompt(strategy)

        transcript_text = "\n".join(
            f"**[{t.agent_name}]** (Round {t.round_number}): {t.text}"
            for t in recent_turns[-12:]  # last ~2 rounds
        )

        user_prompt = format_moderator_user_prompt(
            trigger_dimension=trigger_dimension,
            trigger_value=trigger_score,
            threshold=threshold,
            triggering_agent=triggering_agent,
            strategy=strategy,
            recent_transcript=transcript_text,
            num_turns=len(recent_turns[-12:]),
        )

        resp = self.llm_client.complete(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            parse_json=True,
        )

        # Extract intervention text from JSON
        if resp.parsed_json:
            intervention_text = resp.parsed_json.get("intervention_text", resp.text)
        else:
            intervention_text = resp.text

        event = InterventionEvent(
            intervention_id=str(uuid.uuid4()),
            round_number=round_number,
            source=InterventionSource.STRATEGY,
            strategy=strategy,
            trigger_dimension=trigger_dimension,
            trigger_score=trigger_score,
            trigger_confirmed=True,
            intervention_text=intervention_text,
            moderator_output=resp.parsed_json,
            token_count_input=resp.input_tokens,
            token_count_output=resp.output_tokens,
            latency_s=resp.latency_s,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"[Round {round_number}] Moderator ({strategy}): "
            f"intervention on {trigger_dimension}={trigger_score:.1f}"
        )
        return event

    # ------------------------------------------------------------------
    # Habermas protocol (Baseline 3)
    # ------------------------------------------------------------------

    def generate_habermas_summary(
        self,
        round_turns: list[Turn],
        round_number: int,
    ) -> InterventionEvent:
        """Generate a Habermas Machine consensus summary after a round.

        Args:
            round_turns: All 6 agent turns from the completed round.
            round_number: Current round number.

        Returns:
            InterventionEvent with Habermas output.
        """
        transcript_text = "\n".join(
            f"**[{t.agent_name}]**: {t.text}" for t in round_turns
        )

        user_prompt = HABERMAS_USER_TEMPLATE.format(
            round_number=round_number,
            round_transcript=transcript_text,
        )

        resp = self.llm_client.complete(
            model=self.model,
            system_prompt=HABERMAS_MODERATOR_SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            max_tokens=max(self.max_tokens, 800),  # Habermas needs more for 5-field JSON
            temperature=self.temperature,
            parse_json=True,
        )

        # Extract instruction for next round as intervention text
        if resp.parsed_json:
            intervention_text = resp.parsed_json.get(
                "instruction_for_next_round",
                resp.parsed_json.get("consensus_statement", resp.text),
            )
        else:
            # JSON parse failed — use raw LLM output as intervention text
            intervention_text = resp.text
            logger.warning(
                f"[Round {round_number}] Habermas JSON parse failed, using raw text"
            )

        event = InterventionEvent(
            intervention_id=str(uuid.uuid4()),
            round_number=round_number,
            source=InterventionSource.HABERMAS,
            strategy="habermas",
            trigger_confirmed=True,
            intervention_text=intervention_text,
            habermas_output=resp.parsed_json,
            token_count_input=resp.input_tokens,
            token_count_output=resp.output_tokens,
            latency_s=resp.latency_s,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"[Round {round_number}] Habermas summary generated")
        return event

    # ------------------------------------------------------------------
    # Random messages (Baseline 4)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_random_intervention(
        round_number: int,
        trigger_dimension: Optional[str] = None,
        trigger_score: Optional[float] = None,
    ) -> InterventionEvent:
        """Select a random generic message (Baseline 4).

        Args:
            round_number: Current round number.
            trigger_dimension: Dimension that fired (for logging).
            trigger_score: Score that fired (for logging).

        Returns:
            InterventionEvent with random message.
        """
        message = random.choice(RANDOM_MODERATOR_MESSAGES)

        event = InterventionEvent(
            intervention_id=str(uuid.uuid4()),
            round_number=round_number,
            source=InterventionSource.RANDOM,
            strategy="random",
            trigger_dimension=trigger_dimension,
            trigger_score=trigger_score,
            trigger_confirmed=True,
            intervention_text=message,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"[Round {round_number}] Random moderator message selected")
        return event

    # ------------------------------------------------------------------
    # Silent control placeholder
    # ------------------------------------------------------------------

    @staticmethod
    def create_silent_event(
        round_number: int,
        strategy: str,
        trigger_dimension: Optional[str] = None,
        trigger_score: Optional[float] = None,
    ) -> InterventionEvent:
        """Create a silent control event (trigger confirmed but no message).

        Args:
            round_number: Current round number.
            strategy: Active strategy.
            trigger_dimension: Dimension that fired.
            trigger_score: Score that fired.

        Returns:
            InterventionEvent with empty intervention_text and
            silent_control=True.
        """
        return InterventionEvent(
            intervention_id=str(uuid.uuid4()),
            round_number=round_number,
            source=InterventionSource.SILENT_CONTROL,
            strategy=strategy,
            trigger_dimension=trigger_dimension,
            trigger_score=trigger_score,
            trigger_confirmed=True,
            silent_control=True,
            intervention_text="",
            timestamp=datetime.now(timezone.utc),
        )
