"""
LLM Judge Modules
==================
Two judge classes wrapping the prompt templates from judge_prompts.py:

1. EvaluationJudge — scores each turn on all 7 dimensions (used every turn).
2. TriggerJudge — single-dimension confirmation for the trigger pipeline.

Both use the 70b judge model and parse structured JSON output.

See also:
    - src/prompts/judge_prompts.py — prompt templates and rubrics
    - src/trigger_check.py — consumes TriggerJudge via judge_call_fn
    - src/models.py — DimensionScores, DimensionJustifications
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm_client import LLMClient, JUDGE_MODEL
from src.models import DimensionScores, DimensionJustifications, Turn
from src.prompts.judge_prompts import (
    get_judge_system_prompt,
    get_trigger_judge_system_prompt,
    format_judge_user_prompt,
    format_trigger_judge_user_prompt,
    DIMENSIONS,
)

logger = logging.getLogger(__name__)


class EvaluationJudge:
    """Scores a debate turn on all 7 discourse quality dimensions.

    Used by the orchestrator after every agent turn. Results are
    stored in the Turn model and used for analysis + trigger input.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = JUDGE_MODEL,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._system_prompt = get_judge_system_prompt()

    def score_turn(
        self,
        turn: Turn,
        topic: str,
        preceding_turns: list[Turn],
    ) -> tuple[DimensionScores, DimensionJustifications, int, int, float]:
        """Score a single turn on all 7 dimensions.

        Args:
            turn: The Turn to evaluate.
            topic: Debate topic title.
            preceding_turns: Prior turns for context.

        Returns:
            (scores, justifications, input_tokens, output_tokens, latency_s)
        """
        # Determine debate phase
        if turn.round_number <= 3:
            phase = "opening (rounds 1-3)"
        elif turn.round_number <= 7:
            phase = "mid-debate (rounds 4-7)"
        else:
            phase = "closing (rounds 8-10)"

        # Format preceding turns
        preceding_text = self._format_preceding(preceding_turns)

        user_prompt = format_judge_user_prompt(
            topic=topic,
            round_number=turn.round_number,
            phase=phase,
            agent_name=turn.agent_name,
            turn_text=turn.text,
            preceding_turns=preceding_text,
        )

        resp = self.llm_client.complete(
            model=self.model,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            parse_json=True,
        )

        scores, justifications = self._parse_response(resp.parsed_json, turn)
        return scores, justifications, resp.input_tokens, resp.output_tokens, resp.latency_s

    def _parse_response(
        self, parsed: Optional[dict], turn: Turn
    ) -> tuple[DimensionScores, DimensionJustifications]:
        """Parse judge JSON response into models, with fallback defaults."""
        if parsed is None:
            logger.warning(
                f"Judge returned unparseable JSON for turn {turn.turn_id}. "
                f"Defaulting all scores to 3."
            )
            return (
                DimensionScores(**{d: 3 for d in DIMENSIONS}),
                DimensionJustifications(),
            )

        raw_scores = parsed.get("scores", {})
        raw_justs = parsed.get("justifications", {})

        # Clamp scores to 1-5, default to 3 if missing
        score_dict = {}
        for dim in DIMENSIONS:
            val = raw_scores.get(dim, 3)
            score_dict[dim] = max(1, min(5, int(val)))

        just_dict = {}
        for dim in DIMENSIONS:
            just_dict[dim] = str(raw_justs.get(dim, ""))

        return DimensionScores(**score_dict), DimensionJustifications(**just_dict)

    @staticmethod
    def _format_preceding(turns: list[Turn], max_turns: int = 12) -> str:
        """Format preceding turns for judge context."""
        if not turns:
            return "(No preceding turns — this is the first turn.)"
        recent = turns[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"**[{t.agent_name}]** (Round {t.round_number}): {t.text}")
        return "\n".join(lines)


class TriggerJudge:
    """Single-dimension confirmation judge for the trigger pipeline.

    This class provides the judge_call_fn expected by
    trigger_check.judge_confirmation().
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = JUDGE_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._system_prompt = get_trigger_judge_system_prompt()

    def __call__(self, system_prompt: str, user_prompt: str) -> dict:
        """Callable interface matching trigger_check.judge_call_fn signature.

        Args:
            system_prompt: Trigger judge system prompt (passed by trigger_check).
            user_prompt: Formatted trigger confirmation prompt.

        Returns:
            Parsed JSON dict with keys: dimension, score, trigger_confirmed,
            justification. Falls back to auto-confirm on parse failure.
        """
        resp = self.llm_client.complete(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            parse_json=True,
        )

        if resp.parsed_json is not None:
            return resp.parsed_json

        # Fallback: auto-confirm if parsing fails
        logger.warning("Trigger judge returned unparseable JSON. Auto-confirming.")
        return {
            "dimension": "unknown",
            "score": 1,
            "trigger_confirmed": True,
            "justification": "Auto-confirmed due to JSON parse failure.",
        }
