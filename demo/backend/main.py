"""
Interactive Demo — FastAPI Backend
====================================
REST API + WebSocket for real-time debate streaming.

Endpoints:
    POST   /debate/start                  — start a new debate
    GET    /debate/{id}/status            — get current status
    GET    /debate/{id}/transcript        — get full transcript
    POST   /debate/{id}/inject-intervention — inject manual intervention
    WS     /debate/{id}/ws               — real-time turn streaming
    GET    /debates                       — list all sessions
    GET    /config/topics                 — available topics
    GET    /config/conditions             — available conditions

Usage:
    cd strategy-matters-thesis
    uvicorn demo.backend.main:app --reload --port 8000

See also:
    - action_plan.html — Interactive Demo (Week 9-10)
    - demo/backend/debate_manager.py — debate lifecycle management
    - demo/backend/schemas.py — request/response models
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from demo.backend.debate_manager import debate_manager
from demo.backend.schemas import (
    DebateInfo,
    InjectInterventionRequest,
    StartDebateRequest,
    TranscriptResponse,
    TurnResponse,
    InterventionResponse,
)
from src.experiment_config import CONDITIONS, FRAMING_PROMPTS, TOPIC_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Strategy Matters — Debate Demo",
    description="Interactive demo for AI-moderated multi-agent political debates",
    version="0.1.0",
)

# CORS for React frontend (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/config/topics")
def get_topics() -> list[dict]:
    """List available debate topics."""
    return [
        {
            "id": topic_id,
            "title": topic_id.replace("_", " ").title(),
            "type": TOPIC_TYPES.get(topic_id, "empirical"),
            "framing_prompt": prompt[:200] + "...",
        }
        for topic_id, prompt in FRAMING_PROMPTS.items()
    ]


@app.get("/config/conditions")
def get_conditions() -> list[dict]:
    """List available experimental conditions."""
    return [
        {
            "id": cond_id,
            "label": cond.label,
            "type": cond.condition_type.value,
            "has_moderator": cond.has_moderator_agent,
            "uses_trigger": cond.uses_trigger,
            "trigger_strategy": cond.trigger_strategy,
        }
        for cond_id, cond in CONDITIONS.items()
    ]


# ---------------------------------------------------------------------------
# Debate endpoints
# ---------------------------------------------------------------------------

@app.get("/debates", response_model=list[DebateInfo])
def list_debates() -> list[DebateInfo]:
    """List all debate sessions."""
    return debate_manager.list_sessions()


@app.post("/debate/start", response_model=DebateInfo)
async def start_debate(request: StartDebateRequest) -> DebateInfo:
    """Start a new debate in the background."""
    loop = asyncio.get_running_loop()

    try:
        session = debate_manager.start_debate(
            topic_id=request.topic_id,
            condition_id=request.condition_id,
            num_rounds=request.num_rounds,
            language=request.language,
            loop=loop,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return debate_manager._session_to_info(session)


@app.get("/debate/{debate_id}/status", response_model=DebateInfo)
def get_debate_status(debate_id: str) -> DebateInfo:
    """Get current debate status."""
    session = debate_manager.get_session(debate_id)
    if not session:
        raise HTTPException(status_code=404, detail="Debate not found")
    return debate_manager._session_to_info(session)


@app.get("/debate/{debate_id}/transcript", response_model=TranscriptResponse)
def get_transcript(debate_id: str) -> TranscriptResponse:
    """Get the full transcript for a debate."""
    session = debate_manager.get_session(debate_id)
    if not session:
        raise HTTPException(status_code=404, detail="Debate not found")

    turns = [
        TurnResponse(
            turn_id=t.turn_id,
            round_number=t.round_number,
            turn_in_round=t.turn_in_round,
            agent_name=t.agent_name,
            text=t.text,
            scores=t.scores.to_dict() if t.scores else None,
            timestamp=t.timestamp.isoformat() if t.timestamp else None,
        )
        for t in session.turns
    ]

    interventions = [
        InterventionResponse(
            intervention_id=i.intervention_id,
            round_number=i.round_number,
            source=i.source.value,
            strategy=i.strategy,
            trigger_dimension=i.trigger_dimension,
            trigger_score=i.trigger_score,
            silent_control=i.silent_control,
            intervention_text=i.intervention_text,
            moderator_output=i.moderator_output,
            timestamp=i.timestamp.isoformat() if i.timestamp else None,
        )
        for i in session.interventions
    ]

    return TranscriptResponse(
        debate_id=debate_id,
        status=session.status,
        config={
            "topic_id": session.topic_id,
            "condition_id": session.condition_id,
            "condition_label": session.condition_label,
            "num_rounds": session.num_rounds,
            "language": session.language,
        },
        turns=turns,
        interventions=interventions,
        round_summaries=session.round_summaries,
    )


@app.post("/debate/{debate_id}/inject-intervention")
def inject_intervention(
    debate_id: str, request: InjectInterventionRequest
) -> dict:
    """Inject a manual moderator intervention into a running debate."""
    success = debate_manager.inject_intervention(debate_id, request.text)
    if not success:
        session = debate_manager.get_session(debate_id)
        if not session:
            raise HTTPException(status_code=404, detail="Debate not found")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot inject: debate is {session.status.value}",
        )
    return {"status": "queued", "debate_id": debate_id}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/debate/{debate_id}/ws")
async def debate_websocket(websocket: WebSocket, debate_id: str):
    """Real-time turn streaming via WebSocket.

    Client connects, receives all events (turns, interventions,
    round summaries, status changes) as they occur.
    """
    session = debate_manager.get_session(debate_id)
    if not session:
        await websocket.close(code=4004, reason="Debate not found")
        return

    await websocket.accept()
    # Ensure session has the correct running event loop for cross-thread sends
    session.loop = asyncio.get_running_loop()
    session.ws_clients.append(websocket)

    try:
        # Keep connection alive until client disconnects or debate ends
        while True:
            # Wait for client messages (e.g. ping/pong or close)
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data == "ping":
                await websocket.send_text('{"event_type":"pong","data":{}}')
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in session.ws_clients:
            session.ws_clients.remove(websocket)
