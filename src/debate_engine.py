"""
Debate Orchestration Engine
=============================
Central module managing the lifecycle of a single debate run:

    1. Initialise debate (topic, condition, agents, RAG pools).
    2. Execute 10 rounds of round-robin turns.
    3. After each turn: evaluate (judge), check triggers, invoke moderator.
    4. After each round: compute round summary, check plateau.
    5. Produce structured DebateRun log.

Interface:
    engine = DebateEngine(config, llm_client, rag_pipeline)
    result = engine.run()

See also:
    - src/models.py — DebateRun, Turn, etc.
    - src/agent.py — PoliticalAgent
    - src/judge.py — EvaluationJudge, TriggerJudge
    - src/moderator.py — Moderator
    - src/trigger_check.py — check_trigger pipeline
    - src/experiment_config.py — condition definitions
    - expose.tex §4.6 — experiment design
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.agent import PoliticalAgent
from src.experiment_config import (
    CONDITIONS,
    NUDGE_INSTRUCTION,
)
from src.judge import EvaluationJudge, TriggerJudge
from src.llm_client import LLMClient, AGENT_MODEL, JUDGE_MODEL
from src.models import (
    DebateRun,
    DebateRunConfig,
    DimensionScores,
    InterventionEvent,
    RoundSummary,
    Turn,
)
from src.moderator import Moderator
from src.prompts.agent_prompts import TURN_ORDER
from src.prompts.moderator_prompts import STRATEGY_TRIGGERS
from src.rag_pipeline import RAGPipeline
from src.trigger_check import check_trigger, TriggerStage

logger = logging.getLogger(__name__)

# Plateau detection threshold
PLATEAU_THRESHOLD = 0.1


class DebateEngine:
    """Orchestrates a single debate run.

    Args:
        topic_id: Debate topic key (e.g. "mindestlohn").
        framing_prompt: Neutral framing prompt for the topic.
        topic_type: "empirical" or "values-driven".
        condition_id: Experimental condition key (e.g. "strategy_a").
        run_number: Run number within this condition-topic cell.
        llm_client: Shared LLMClient.
        rag_pipeline: Optional RAGPipeline (None to skip RAG).
        agent_model: Model for agent calls.
        judge_model: Model for judge/moderator calls.
        num_rounds: Number of debate rounds (default 10).
        max_interventions: Cap on moderator interventions per run.
        silent_control_rate: Proportion of triggers silenced.
    """

    def __init__(
        self,
        topic_id: str,
        framing_prompt: str,
        topic_type: str,
        condition_id: str,
        run_number: int,
        llm_client: LLMClient,
        rag_pipeline: Optional[RAGPipeline] = None,
        agent_model: str = AGENT_MODEL,
        judge_model: str = JUDGE_MODEL,
        num_rounds: int = 10,
        max_interventions: int = 3,
        silent_control_rate: float = 0.20,
        language: str = "en",
    ):
        self.topic_id = topic_id
        self.framing_prompt = framing_prompt
        self.topic_type = topic_type
        self.condition_id = condition_id
        self.run_number = run_number
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline
        self.agent_model = agent_model
        self.judge_model = judge_model
        self.num_rounds = num_rounds
        self.max_interventions = max_interventions
        self.silent_control_rate = silent_control_rate
        self.language = language

        # Load condition spec
        self.condition = CONDITIONS[condition_id]
        self.turn_order = TURN_ORDER

        # Build components
        self.agents = self._build_agents()
        self.eval_judge = EvaluationJudge(llm_client, model=judge_model)
        self.trigger_judge = TriggerJudge(llm_client, model=judge_model)
        self.moderator = Moderator(llm_client, model=judge_model)

        # State
        self.transcript: list[Turn] = []
        self.interventions: list[InterventionEvent] = []
        self.round_summaries: list[RoundSummary] = []
        self.last_intervention_text: str = ""
        self.intervention_count: int = 0  # active (non-silent) interventions

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_agents(self) -> dict[str, PoliticalAgent]:
        """Create one PoliticalAgent per party."""
        nudge = NUDGE_INSTRUCTION if self.condition_id == "baseline_2" else ""
        agents = {}
        for party in self.turn_order:
            agents[party] = PoliticalAgent(
                party=party,
                llm_client=self.llm_client,
                rag_pipeline=self.rag_pipeline,
                model=self.agent_model,
                nudge_text=nudge,
                language=self.language,
            )
        return agents

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> DebateRun:
        """Execute the full debate and return structured log."""
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        logger.info(
            f"=== Debate {run_id[:8]} | "
            f"topic={self.topic_id} | condition={self.condition_id} | "
            f"run={self.run_number} ==="
        )

        # Initialise RAG pools
        if self.rag_pipeline is not None:
            self.rag_pipeline.init_pools(self.framing_prompt)

        # Run rounds
        for round_num in range(1, self.num_rounds + 1):
            self._run_round(round_num)

        finished_at = datetime.now(timezone.utc)

        config = self._build_config()

        # Compute totals
        total_in = sum(t.token_count_input for t in self.transcript)
        total_out = sum(t.token_count_output for t in self.transcript)
        total_in += sum(i.token_count_input for i in self.interventions)
        total_out += sum(i.token_count_output for i in self.interventions)

        result = DebateRun(
            run_id=run_id,
            config=config,
            turns=self.transcript,
            interventions=self.interventions,
            round_summaries=self.round_summaries,
            total_tokens_input=total_in,
            total_tokens_output=total_out,
            total_latency_s=round(
                sum(t.latency_s for t in self.transcript)
                + sum(i.latency_s for i in self.interventions),
                2,
            ),
            started_at=started_at,
            finished_at=finished_at,
        )

        logger.info(
            f"=== Debate complete | turns={len(self.transcript)} | "
            f"interventions={self.intervention_count} | "
            f"tokens={total_in + total_out:,} ==="
        )
        return result

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(self) -> DebateRunConfig:
        """Build a DebateRunConfig snapshot from current engine state."""
        return DebateRunConfig(
            topic_id=self.topic_id,
            topic_type=self.topic_type,
            framing_prompt=self.framing_prompt,
            condition_id=self.condition_id,
            condition_label=self.condition.label,
            condition_type=self.condition.condition_type.value,
            run_number=self.run_number,
            agent_model=self.agent_model,
            judge_model=self.judge_model,
            moderator_model=self.judge_model if self.condition.has_moderator_agent else "",
            num_rounds=self.num_rounds,
            num_agents=len(self.turn_order),
            max_interventions=self.max_interventions,
            silent_control_rate=self.silent_control_rate,
            turn_order=list(self.turn_order),
            has_moderator_agent=self.condition.has_moderator_agent,
            uses_trigger=self.condition.uses_trigger,
            trigger_strategy=self.condition.trigger_strategy,
            nudge_text=NUDGE_INSTRUCTION if self.condition_id == "baseline_2" else "",
            language=self.language,
        )

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int) -> None:
        """Execute one round: 6 agent turns + evaluation + triggers."""
        round_turns: list[Turn] = []
        round_scores: list[dict] = []

        for turn_idx, party in enumerate(self.turn_order, start=1):
            # 1. Agent generates turn
            agent = self.agents[party]
            turn = agent.generate_turn(
                framing_prompt=self.framing_prompt,
                transcript=self.transcript,
                round_number=round_num,
                turn_in_round=turn_idx,
                last_intervention=self.last_intervention_text,
            )

            # 2. Judge scores the turn
            scores, justifications, judge_in, judge_out, judge_lat = (
                self.eval_judge.score_turn(
                    turn=turn,
                    topic=self.topic_id,
                    preceding_turns=self.transcript[-12:],
                )
            )
            turn.scores = scores
            turn.justifications = justifications
            # Add judge token cost to turn (for accounting)
            turn.token_count_input += judge_in
            turn.token_count_output += judge_out
            turn.latency_s += judge_lat

            # 3. Append to state
            self.transcript.append(turn)
            round_turns.append(turn)
            round_scores.append(scores.to_dict())

            # 4. Trigger check (if condition uses triggers)
            if self.condition.uses_trigger:
                self._check_and_intervene(
                    turn=turn,
                    round_num=round_num,
                    round_scores=round_scores,
                )

        # 5. Habermas intervention (Baseline 3 — after every round)
        if self.condition_id == "baseline_3":
            event = self.moderator.generate_habermas_summary(
                round_turns=round_turns,
                round_number=round_num,
            )
            self.interventions.append(event)
            self.last_intervention_text = event.intervention_text
            self.intervention_count += 1

        # 6. Round summary
        summary = self._compute_round_summary(round_num, round_turns)
        self.round_summaries.append(summary)

    # ------------------------------------------------------------------
    # Trigger and intervention logic
    # ------------------------------------------------------------------

    def _check_and_intervene(
        self,
        turn: Turn,
        round_num: int,
        round_scores: list[dict],
    ) -> None:
        """Run trigger pipeline after an agent turn; invoke moderator if needed."""
        if turn.scores is None:
            return

        strategy = self.condition.trigger_strategy or ""

        # Format preceding turns for trigger judge context
        preceding_text = "\n".join(
            f"**[{t.agent_name}]** (Round {t.round_number}): {t.text}"
            for t in self.transcript[-12:]
        )

        result = check_trigger(
            current_scores=turn.scores.to_dict(),
            strategy=strategy,
            agent_name=turn.agent_name,
            turn_text=turn.text,
            preceding_turns=preceding_text,
            round_number=round_num,
            intervention_count=self.intervention_count,
            max_interventions=self.max_interventions,
            silent_control_rate=self.silent_control_rate,
            round_scores=round_scores,
            judge_call_fn=self.trigger_judge,
        )

        if not result.triggered:
            return

        # Trigger confirmed — decide on intervention
        if result.silent_control:
            event = Moderator.create_silent_event(
                round_number=round_num,
                strategy=strategy,
                trigger_dimension=result.dimension,
                trigger_score=result.score,
            )
            self.interventions.append(event)
            # Silent: no intervention text, don't increment count
            return

        # Generate actual intervention
        if strategy == "random":
            event = Moderator.generate_random_intervention(
                round_number=round_num,
                trigger_dimension=result.dimension,
                trigger_score=result.score,
            )
        else:
            # Get trigger threshold for the strategy
            trigger_cfg = STRATEGY_TRIGGERS.get(strategy, {})
            threshold = trigger_cfg.get("threshold", 2.0)

            event = self.moderator.generate_strategy_intervention(
                strategy=strategy,
                trigger_dimension=result.dimension or "",
                trigger_score=result.score or 0.0,
                threshold=threshold,
                triggering_agent=turn.agent_name,
                recent_turns=self.transcript[-12:],
                round_number=round_num,
            )

        event.after_turn_id = turn.turn_id
        self.interventions.append(event)
        self.last_intervention_text = event.intervention_text
        self.intervention_count += 1

        logger.info(
            f"  → Intervention #{self.intervention_count} "
            f"({strategy}, {result.dimension}={result.score})"
        )

    # ------------------------------------------------------------------
    # Round summary and plateau detection
    # ------------------------------------------------------------------

    def _compute_round_summary(
        self, round_num: int, round_turns: list[Turn]
    ) -> RoundSummary:
        """Compute aggregated scores for one round + plateau check."""
        per_agent: dict[str, DimensionScores] = {}
        all_scores: dict[str, list[int]] = {d: [] for d in DimensionScores.model_fields}

        for turn in round_turns:
            if turn.scores is not None:
                per_agent[turn.agent_name] = turn.scores
                for dim, val in turn.scores.to_dict().items():
                    all_scores[dim].append(val)

        # Mean scores
        mean_dict = {}
        for dim, vals in all_scores.items():
            mean_dict[dim] = round(sum(vals) / len(vals)) if vals else 3
        mean_scores = DimensionScores(**mean_dict)

        # Plateau detection: all primary dims change < 0.1 from previous round
        plateau = False
        if len(self.round_summaries) >= 1:
            prev = self.round_summaries[-1]
            prev_primary = prev.mean_scores.primary_dims
            curr_primary = mean_scores.primary_dims
            plateau = all(
                abs(curr_primary[d] - prev_primary[d]) < PLATEAU_THRESHOLD
                for d in curr_primary
            )

        summary = RoundSummary(
            round_number=round_num,
            mean_scores=mean_scores,
            per_agent_scores=per_agent,
            composite=round(mean_scores.composite, 2),
            plateau=plateau,
            intervention_count_so_far=self.intervention_count,
        )

        if plateau:
            logger.info(f"  [Round {round_num}] Plateau detected")

        return summary
