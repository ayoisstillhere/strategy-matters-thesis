"""
Debate Manager
===============
Manages running debates in background threads, maintains state,
and pushes events to connected WebSocket clients.

Architecture:
    - Debates run in a ThreadPoolExecutor (DebateEngine is synchronous).
    - Each debate has a DebateSession holding state + connected WS clients.
    - A custom callback-based engine wrapper emits events per turn/intervention.
    - WebSocket clients receive events in real-time.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import WebSocket

from demo.backend.schemas import (
    DebateInfo,
    DebateStatus,
    InterventionResponse,
    TurnResponse,
    WSEvent,
    WSEventType,
)

import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.debate_engine import DebateEngine
from src.experiment_config import CONDITIONS, FRAMING_PROMPTS, TOPIC_TYPES
from src.export import save_run_json
from src.llm_client import LLMClient
from src.models import DebateRun, InterventionEvent, Turn
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Thread pool for running debates (blocking LLM calls)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="debate")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class DebateSession:
    """Holds the state of a single debate run."""
    debate_id: str
    topic_id: str
    condition_id: str
    condition_label: str
    num_rounds: int
    language: str
    status: DebateStatus = DebateStatus.PENDING
    current_round: int = 0
    turns: list[Turn] = field(default_factory=list)
    interventions: list[InterventionEvent] = field(default_factory=list)
    round_summaries: list[dict] = field(default_factory=list)
    result: Optional[DebateRun] = None
    error_message: Optional[str] = None
    # Injected interventions queued by the user
    pending_injection: Optional[str] = None
    # Last intervention text (for passing to agents)
    last_intervention_text: str = ""
    # Connected WebSocket clients
    ws_clients: list[WebSocket] = field(default_factory=list)
    # Event loop reference for cross-thread communication
    loop: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DebateManager:
    """Singleton manager for all debate sessions."""

    def __init__(self):
        self._sessions: dict[str, DebateSession] = {}
        self._llm_client: Optional[LLMClient] = None

    def _get_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def get_session(self, debate_id: str) -> Optional[DebateSession]:
        return self._sessions.get(debate_id)

    def list_sessions(self) -> list[DebateInfo]:
        return [self._session_to_info(s) for s in self._sessions.values()]

    def start_debate(
        self,
        topic_id: str,
        condition_id: str,
        num_rounds: int = 10,
        language: str = "de",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> DebateSession:
        """Start a new debate in a background thread."""
        if topic_id not in FRAMING_PROMPTS:
            raise ValueError(f"Unknown topic_id: {topic_id}")
        if condition_id not in CONDITIONS:
            raise ValueError(f"Unknown condition_id: {condition_id}")

        debate_id = str(uuid.uuid4())
        condition = CONDITIONS[condition_id]

        session = DebateSession(
            debate_id=debate_id,
            topic_id=topic_id,
            condition_id=condition_id,
            condition_label=condition.label,
            num_rounds=num_rounds,
            language=language,
            loop=loop or asyncio.get_event_loop(),
        )
        self._sessions[debate_id] = session

        # Submit to thread pool
        _executor.submit(self._run_debate, session)
        return session

    def inject_intervention(self, debate_id: str, text: str) -> bool:
        """Queue a manual intervention for the next turn."""
        session = self._sessions.get(debate_id)
        if not session or session.status != DebateStatus.RUNNING:
            return False
        session.pending_injection = text
        return True

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    def _run_debate(self, session: DebateSession) -> None:
        """Run the debate engine in a background thread."""
        session.status = DebateStatus.RUNNING
        self._broadcast_sync(session, WSEvent(
            event_type=WSEventType.STATUS_CHANGE,
            data={"status": "running", "debate_id": session.debate_id},
        ))

        try:
            client = self._get_client()
            rag = RAGPipeline()  # uses default embeddings dir

            engine = DebateEngine(
                topic_id=session.topic_id,
                framing_prompt=FRAMING_PROMPTS[session.topic_id],
                topic_type=TOPIC_TYPES.get(session.topic_id, "empirical"),
                condition_id=session.condition_id,
                run_number=1,
                llm_client=client,
                rag_pipeline=rag,
                num_rounds=session.num_rounds,
                language=session.language,
            )

            # Override the engine's run loop to emit events per turn
            self._run_engine_with_events(engine, session)

            # Save result
            session.result = DebateRun(
                run_id=session.debate_id,
                config=engine._build_config(),
                turns=session.turns,
                interventions=session.interventions,
                round_summaries=[],
                total_tokens_input=sum(t.token_count_input for t in session.turns),
                total_tokens_output=sum(t.token_count_output for t in session.turns),
                total_latency_s=round(sum(t.latency_s for t in session.turns), 2),
                started_at=session.turns[0].timestamp if session.turns else None,
                finished_at=datetime.now(timezone.utc),
            )

            # Save to disk
            output_dir = PROJECT_ROOT / "runs" / "demo"
            save_run_json(session.result, output_dir)

            session.status = DebateStatus.COMPLETED
            self._broadcast_sync(session, WSEvent(
                event_type=WSEventType.STATUS_CHANGE,
                data={"status": "completed", "debate_id": session.debate_id},
            ))

        except Exception as e:
            logger.error(f"Debate {session.debate_id} failed: {e}", exc_info=True)
            session.status = DebateStatus.ERROR
            session.error_message = str(e)
            self._broadcast_sync(session, WSEvent(
                event_type=WSEventType.ERROR,
                data={"error": str(e), "debate_id": session.debate_id},
            ))

    def _run_engine_with_events(
        self, engine: DebateEngine, session: DebateSession
    ) -> None:
        """Custom run loop that emits WS events after each turn."""
        from src.prompts.agent_prompts import TURN_ORDER
        from src.prompts.moderator_prompts import STRATEGY_TRIGGERS
        from src.trigger_check import check_trigger
        from src.moderator import Moderator

        # Init RAG if available
        if engine.rag_pipeline is not None:
            engine.rag_pipeline.init_pools(engine.framing_prompt)

        session.last_intervention_text = ""

        for round_num in range(1, session.num_rounds + 1):
            session.current_round = round_num
            round_turns = []
            round_trigger_fired = False  # max 1 trigger per round

            for turn_idx, party in enumerate(TURN_ORDER, start=1):
                # Check for pending user injection before this turn
                if session.pending_injection:
                    injection_text = session.pending_injection
                    session.pending_injection = None
                    session.last_intervention_text = injection_text

                    # Log as intervention event
                    from src.models import InterventionEvent, InterventionSource
                    import uuid as _uuid
                    inj_event = InterventionEvent(
                        intervention_id=str(_uuid.uuid4()),
                        round_number=round_num,
                        source=InterventionSource.STRATEGY,
                        strategy="human_injection",
                        trigger_confirmed=True,
                        intervention_text=injection_text,
                        timestamp=datetime.now(timezone.utc),
                    )
                    session.interventions.append(inj_event)
                    self._broadcast_sync(session, WSEvent(
                        event_type=WSEventType.INTERVENTION,
                        data=self._intervention_to_dict(inj_event),
                    ))

                # Generate turn
                agent = engine.agents[party]
                turn = agent.generate_turn(
                    framing_prompt=engine.framing_prompt,
                    transcript=session.turns,
                    round_number=round_num,
                    turn_in_round=turn_idx,
                    last_intervention=session.last_intervention_text,
                )

                # Score turn
                scores, justifications, judge_in, judge_out, judge_lat = (
                    engine.eval_judge.score_turn(
                        turn=turn,
                        topic=engine.topic_id,
                        preceding_turns=session.turns[-12:],
                    )
                )
                turn.scores = scores
                turn.justifications = justifications
                turn.token_count_input += judge_in
                turn.token_count_output += judge_out
                turn.latency_s += judge_lat

                # Append to session state
                session.turns.append(turn)
                round_turns.append(turn)

                # Broadcast turn event
                self._broadcast_sync(session, WSEvent(
                    event_type=WSEventType.TURN,
                    data=self._turn_to_dict(turn),
                ))

                # Trigger check (if condition uses triggers)
                #   Skip round 1: agents have no prior turns to respond to.
                #   Max 1 trigger per round to preserve intervention budget.
                if (engine.condition.uses_trigger
                        and round_num > 1
                        and not round_trigger_fired):
                    fired = self._check_trigger_and_intervene(
                        engine, session, turn, round_num,
                    )
                    if fired:
                        round_trigger_fired = True

            # Habermas (Baseline 3)
            if engine.condition_id == "baseline_3":
                event = engine.moderator.generate_habermas_summary(
                    round_turns=round_turns,
                    round_number=round_num,
                )
                session.interventions.append(event)
                session.last_intervention_text = event.intervention_text
                self._broadcast_sync(session, WSEvent(
                    event_type=WSEventType.INTERVENTION,
                    data=self._intervention_to_dict(event),
                ))

            # Round summary event
            summary = engine._compute_round_summary(round_num, round_turns)
            engine.round_summaries.append(summary)
            summary_dict = {
                "round_number": summary.round_number,
                "composite": summary.composite,
                "plateau": summary.plateau,
                "scores": summary.mean_scores.to_dict(),
            }
            session.round_summaries.append(summary_dict)
            self._broadcast_sync(session, WSEvent(
                event_type=WSEventType.ROUND_SUMMARY,
                data=summary_dict,
            ))

    def _check_trigger_and_intervene(
        self,
        engine: DebateEngine,
        session: DebateSession,
        turn: Turn,
        round_num: int,
    ) -> bool:
        """Run trigger pipeline; broadcast intervention if fired.

        Returns True if a trigger fired, False otherwise.
        """
        if turn.scores is None:
            return False

        from src.trigger_check import check_trigger
        from src.prompts.moderator_prompts import STRATEGY_TRIGGERS

        strategy = engine.condition.trigger_strategy or ""
        preceding_text = "\n".join(
            f"**[{t.agent_name}]** (Round {t.round_number}): {t.text}"
            for t in session.turns[-12:]
        )

        intervention_count = sum(
            1 for i in session.interventions
            if not i.silent_control and i.intervention_text
        )

        result = check_trigger(
            current_scores=turn.scores.to_dict(),
            strategy=strategy,
            agent_name=turn.agent_name,
            turn_text=turn.text,
            preceding_turns=preceding_text,
            round_number=round_num,
            intervention_count=intervention_count,
            max_interventions=engine.max_interventions,
            silent_control_rate=engine.silent_control_rate,
            round_scores=[t.scores.to_dict() for t in session.turns if t.scores],
            judge_call_fn=engine.trigger_judge,
        )

        if not result.triggered:
            return False

        from src.moderator import Moderator

        if result.silent_control:
            event = Moderator.create_silent_event(
                round_number=round_num,
                strategy=strategy,
                trigger_dimension=result.dimension,
                trigger_score=result.score,
            )
        elif strategy == "random":
            event = Moderator.generate_random_intervention(
                round_number=round_num,
                trigger_dimension=result.dimension,
                trigger_score=result.score,
            )
        else:
            trigger_cfg = STRATEGY_TRIGGERS.get(strategy, {})
            threshold = trigger_cfg.get("threshold", 2.0)
            event = engine.moderator.generate_strategy_intervention(
                strategy=strategy,
                trigger_dimension=result.dimension or "",
                trigger_score=result.score or 0.0,
                threshold=threshold,
                triggering_agent=turn.agent_name,
                recent_turns=session.turns[-12:],
                round_number=round_num,
            )

        event.after_turn_id = turn.turn_id
        session.interventions.append(event)

        if not event.silent_control and event.intervention_text:
            session.last_intervention_text = event.intervention_text

        self._broadcast_sync(session, WSEvent(
            event_type=WSEventType.INTERVENTION,
            data=self._intervention_to_dict(event),
        ))
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_to_info(self, session: DebateSession) -> DebateInfo:
        return DebateInfo(
            debate_id=session.debate_id,
            topic_id=session.topic_id,
            condition_id=session.condition_id,
            condition_label=session.condition_label,
            status=session.status,
            current_round=session.current_round,
            total_rounds=session.num_rounds,
            turn_count=len(session.turns),
            intervention_count=len([
                i for i in session.interventions
                if not i.silent_control and i.intervention_text
            ]),
            error_message=session.error_message,
        )

    @staticmethod
    def _turn_to_dict(turn: Turn) -> dict:
        return TurnResponse(
            turn_id=turn.turn_id,
            round_number=turn.round_number,
            turn_in_round=turn.turn_in_round,
            agent_name=turn.agent_name,
            text=turn.text,
            scores=turn.scores.to_dict() if turn.scores else None,
            timestamp=turn.timestamp.isoformat() if turn.timestamp else None,
        ).model_dump()

    @staticmethod
    def _intervention_to_dict(event: InterventionEvent) -> dict:
        return InterventionResponse(
            intervention_id=event.intervention_id,
            round_number=event.round_number,
            source=event.source.value,
            strategy=event.strategy,
            trigger_dimension=event.trigger_dimension,
            trigger_score=event.trigger_score,
            silent_control=event.silent_control,
            intervention_text=event.intervention_text,
            moderator_output=event.moderator_output,
            habermas_output=event.habermas_output,
            timestamp=event.timestamp.isoformat() if event.timestamp else None,
        ).model_dump()

    def _broadcast_sync(self, session: DebateSession, event: WSEvent) -> None:
        """Thread-safe broadcast to all connected WS clients."""
        if not session.ws_clients or session.loop is None:
            return
        message = event.model_dump_json()
        for ws in list(session.ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(message), session.loop
                )
            except Exception:
                # Client disconnected
                session.ws_clients.remove(ws)


# Singleton instance
debate_manager = DebateManager()
