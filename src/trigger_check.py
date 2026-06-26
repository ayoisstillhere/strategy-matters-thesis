"""
Trigger-Check Module
====================
Two-stage trigger mechanism that decides whether a moderator intervention
is warranted after each agent turn.

Stage 1 — Rule-based threshold check (cheap, runs every turn):
    Evaluates the evaluation-judge scores against strategy-specific
    trigger conditions. No LLM call required.

Stage 2 — LLM judge confirmation (expensive, runs only when Stage 1 fires):
    Invokes the trigger-check judge (separate prompt from eval judge)
    to assess the turn in context and confirm or reject the trigger.

After confirmation, a silent-control coin flip determines whether the
trigger produces an actual moderator intervention or is logged silently
(20% default). The 3-intervention cap is enforced externally by the
orchestrator before calling this module.

Interface:
    Orchestrator calls check_trigger() after each agent turn.
    Returns a TriggerResult dataclass that the orchestrator uses to
    decide whether to invoke the moderator.

See also:
    - expose.tex §4.4 (AI Moderator Design) — trigger mechanism spec
    - src/prompts/judge_prompts.py — TRIGGER_JUDGE_SYSTEM_PROMPT
    - src/prompts/moderator_prompts.py — STRATEGY_TRIGGERS reference
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TriggerStage(str, Enum):
    """How far the trigger-check pipeline progressed."""
    NONE = "none"                       # Stage 1 did not fire
    RULE_FIRED = "rule_fired"           # Stage 1 fired, Stage 2 not yet run
    JUDGE_CONFIRMED = "judge_confirmed" # Stage 2 confirmed the trigger
    JUDGE_REJECTED = "judge_rejected"   # Stage 2 rejected the trigger
    CAP_REACHED = "cap_reached"         # Stage 1 fired but intervention cap already hit


@dataclass
class TriggerResult:
    """Immutable result of the full trigger-check pipeline for one turn."""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    triggered: bool = False
    stage: TriggerStage = TriggerStage.NONE
    dimension: Optional[str] = None
    score: Optional[float] = None
    strategy: str = ""
    silent_control: bool = False
    judge_response: Optional[dict] = None
    # Populated by the orchestrator for logging
    after_turn_id: Optional[str] = None
    round_number: Optional[int] = None
    agent_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Strategy-specific trigger conditions
# ---------------------------------------------------------------------------

# Simple threshold triggers (strategies A-C):
#   Fire when ANY agent's score on the target dimension falls below threshold.
#
# Compound trigger (strategy D — common-ground):
#   Fire when stance_differentiation > 3 AND responsiveness < 4
#   for 2 or more agents in the current round.

SIMPLE_TRIGGERS = {
    "de-escalation": {"dimension": "civility",           "threshold": 4},
    "reframing":     {"dimension": "responsiveness",     "threshold": 4},
    "fact-reminder": {"dimension": "document_grounding", "threshold": 4},
}

COMMON_GROUND_TRIGGER = {
    "stance_min": 3,          # stance_differentiation must be ABOVE this
    "responsiveness_max": 4,  # responsiveness must be BELOW this
    "min_agents": 2,          # at least this many agents must meet both
}


# ---------------------------------------------------------------------------
# Stage 1 — Rule-based check
# ---------------------------------------------------------------------------

def _check_simple_trigger(
    current_scores: dict[str, int],
    strategy: str,
) -> tuple[bool, Optional[str], Optional[float]]:
    """Check a simple single-dimension threshold trigger.

    Args:
        current_scores: The evaluated turn's dimension scores dict
            (e.g. {"civility": 3, "responsiveness": 2, ...}).
        strategy: Active strategy key (de-escalation | reframing | fact-reminder).

    Returns:
        (fired, dimension, score) — dimension and score are None if not fired.
    """
    cfg = SIMPLE_TRIGGERS[strategy]
    dim = cfg["dimension"]
    threshold = cfg["threshold"]
    value = current_scores.get(dim)
    if value is not None and value < threshold:
        return True, dim, float(value)
    return False, None, None


def _check_common_ground_trigger(
    round_scores: list[dict],
) -> tuple[bool, Optional[str], Optional[float]]:
    """Check the compound common-ground trigger across the current round.

    Args:
        round_scores: List of score dicts for agents evaluated so far in
            the current round. Each dict has dimension keys with int values.

    Returns:
        (fired, dimension, lowest_responsiveness_score).
        dimension is "responsiveness" since that is the target.
    """
    cfg = COMMON_GROUND_TRIGGER
    qualifying_agents = 0
    lowest_resp = None

    for scores in round_scores:
        stance = scores.get("stance_differentiation", 0)
        resp = scores.get("responsiveness", 5)
        if stance > cfg["stance_min"] and resp < cfg["responsiveness_max"]:
            qualifying_agents += 1
            if lowest_resp is None or resp < lowest_resp:
                lowest_resp = resp

    if qualifying_agents >= cfg["min_agents"]:
        return True, "responsiveness", float(lowest_resp) if lowest_resp is not None else None
    return False, None, None


def rule_based_check(
    current_scores: dict[str, int],
    strategy: str,
    round_scores: Optional[list[dict]] = None,
) -> tuple[bool, Optional[str], Optional[float]]:
    """Stage 1: Evaluate strategy-specific trigger condition.

    Args:
        current_scores: Dimension scores for the current agent turn.
        strategy: Active moderation strategy.
        round_scores: All agents' scores in the current round so far
            (required for common-ground; ignored for other strategies).

    Returns:
        (fired, dimension, score).
    """
    if strategy in SIMPLE_TRIGGERS:
        return _check_simple_trigger(current_scores, strategy)

    if strategy == "common-ground":
        if round_scores is None:
            raise ValueError(
                "common-ground trigger requires round_scores "
                "(list of all agents' scores in the current round)."
            )
        return _check_common_ground_trigger(round_scores)

    if strategy == "random":
        # Baseline 4: uses a generic trigger — any primary dimension below 4.
        # Primary dimensions: civility, argument_strength, document_grounding,
        # responsiveness, stance_differentiation.
        primary_dims = [
            "civility", "argument_strength", "document_grounding",
            "responsiveness", "stance_differentiation",
        ]
        for dim in primary_dims:
            value = current_scores.get(dim)
            if value is not None and value < 4:
                return True, dim, float(value)
        return False, None, None

    # Baselines 1-3 and unknown strategies: no trigger
    return False, None, None


# ---------------------------------------------------------------------------
# Stage 2 — LLM judge confirmation
# ---------------------------------------------------------------------------

def judge_confirmation(
    dimension: str,
    score: float,
    agent_name: str,
    turn_text: str,
    preceding_turns: str,
    round_number: int,
    judge_call_fn=None,
) -> dict:
    """Invoke the trigger-check judge to confirm or reject the trigger.

    Args:
        dimension: The dimension that fired in Stage 1.
        score: The score that fired in Stage 1.
        agent_name: Name of the agent whose turn is being evaluated.
        turn_text: The text of the agent's turn.
        preceding_turns: Formatted string of preceding turns for context.
        round_number: Current round number.
        judge_call_fn: Callable that sends the assembled prompt to the
            trigger-check judge LLM and returns the parsed JSON response.
            Signature: (system_prompt: str, user_prompt: str) -> dict
            The dict must contain at minimum: {"score": int, "trigger_confirmed": bool}
            If None, Stage 2 is skipped and the trigger is auto-confirmed
            (useful for testing or when running without API access).

    Returns:
        Parsed judge response dict, or a synthetic confirmation if
        judge_call_fn is None.
    """
    if judge_call_fn is None:
        # Auto-confirm: useful for dry runs or testing
        return {
            "dimension": dimension,
            "score": int(score),
            "trigger_confirmed": True,
            "justification": "Auto-confirmed (no judge_call_fn provided).",
        }

    # Lazy import to avoid circular dependency at module level
    from src.prompts.judge_prompts import (
        get_trigger_judge_system_prompt,
        format_trigger_judge_user_prompt,
    )

    system_prompt = get_trigger_judge_system_prompt()
    user_prompt = format_trigger_judge_user_prompt(
        dimension=dimension,
        initial_score=score,
        agent_name=agent_name,
        round_number=round_number,
        turn_text=turn_text,
        preceding_turns=preceding_turns,
    )

    return judge_call_fn(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Silent control assignment
# ---------------------------------------------------------------------------

def assign_silent_control(rate: float = 0.20) -> bool:
    """Randomly assign this confirmed trigger to the silent control group.

    Within each strategy condition, a proportion of confirmed triggers
    (default 20%) are silenced: the trigger event is logged but no
    moderator message is produced. This enables within-condition causal
    comparison (treated vs silent) to isolate the effect of the
    intervention from natural score recovery (regression to the mean).

    Args:
        rate: Probability of silent assignment (0.0–1.0).

    Returns:
        True if this trigger should be silenced.
    """
    return random.random() < rate


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_trigger(
    current_scores: dict[str, int],
    strategy: str,
    agent_name: str,
    turn_text: str,
    preceding_turns: str,
    round_number: int,
    intervention_count: int,
    max_interventions: int = 3,
    silent_control_rate: float = 0.20,
    round_scores: Optional[list[dict]] = None,
    judge_call_fn=None,
) -> TriggerResult:
    """Full two-stage trigger-check pipeline.

    Called by the orchestrator after each agent turn. The pipeline:
      1. Rule-based threshold check (Stage 1)
      2. If fired → check intervention cap
      3. If under cap → LLM judge confirmation (Stage 2)
      4. If confirmed → silent control coin flip
      5. Return TriggerResult for the orchestrator

    Args:
        current_scores: Dimension scores for the current agent turn
            (from the evaluation judge).
        strategy: Active moderation strategy for this debate condition.
        agent_name: Name of the agent whose turn was just evaluated.
        turn_text: The agent's turn text.
        preceding_turns: Formatted preceding turns for judge context.
        round_number: Current round number (1-indexed).
        intervention_count: Number of interventions produced so far in
            this debate run (NOT counting silent controls).
        max_interventions: Cap on moderator interventions per run.
        silent_control_rate: Proportion of triggers silenced (0.0–1.0).
        round_scores: All agents' scores in the current round (needed
            for common-ground strategy; optional for others).
        judge_call_fn: Callable for the trigger-check judge LLM.
            If None, Stage 2 auto-confirms (for testing).

    Returns:
        TriggerResult with all fields populated.
    """
    result = TriggerResult(strategy=strategy, agent_name=agent_name,
                           round_number=round_number)

    # ── Stage 1: Rule-based check ──
    fired, dimension, score = rule_based_check(
        current_scores, strategy, round_scores
    )

    if not fired:
        result.stage = TriggerStage.NONE
        return result

    result.dimension = dimension
    result.score = score
    result.stage = TriggerStage.RULE_FIRED

    # ── Cap check ──
    # The cap applies only to interventions that produce a moderator message.
    # Silent controls don't count toward the cap.
    # However, we check the cap BEFORE the silent control coin flip,
    # because we need to know whether this trigger CAN produce an
    # intervention. Post-cap triggers are logged separately.
    if intervention_count >= max_interventions:
        result.stage = TriggerStage.CAP_REACHED
        result.triggered = False
        return result

    # ── Stage 2: LLM judge confirmation ──
    judge_response = judge_confirmation(
        dimension=dimension,
        score=score,
        agent_name=agent_name,
        turn_text=turn_text,
        preceding_turns=preceding_turns,
        round_number=round_number,
        judge_call_fn=judge_call_fn,
    )

    result.judge_response = judge_response
    confirmed = judge_response.get("trigger_confirmed", False)

    if not confirmed:
        result.stage = TriggerStage.JUDGE_REJECTED
        result.triggered = False
        return result

    # ── Stage 2 confirmed ──
    result.stage = TriggerStage.JUDGE_CONFIRMED
    result.triggered = True

    # ── Silent control assignment ──
    result.silent_control = assign_silent_control(silent_control_rate)

    return result


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def trigger_result_to_log(result: TriggerResult) -> dict:
    """Convert a TriggerResult to a dict suitable for structured logging.

    This produces the fields expected in the InterventionEvent schema
    (see expose.tex §4.1.3) for the trigger-related fields. The
    moderator's intervention_text and other fields are added by the
    orchestrator after the moderator responds.
    """
    return {
        "trigger_id": result.trigger_id,
        "triggered": result.triggered,
        "stage": result.stage.value,
        "dimension": result.dimension,
        "score": result.score,
        "strategy": result.strategy,
        "silent_control": result.silent_control,
        "judge_response": result.judge_response,
        "after_turn_id": result.after_turn_id,
        "round_number": result.round_number,
        "agent_name": result.agent_name,
    }
