"""
API Schemas
============
Pydantic models for FastAPI request/response serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DebateStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class StartDebateRequest(BaseModel):
    """POST /debate/start body."""
    topic_id: str = Field(
        description="Topic key: mindestlohn, rentenpolitik, migrationspolitik, sozialpolitik"
    )
    condition_id: str = Field(
        description="Condition key: baseline_1..4, strategy_a..d"
    )
    num_rounds: int = Field(default=10, ge=1, le=10)
    language: str = Field(default="de", description="de or en")


class InjectInterventionRequest(BaseModel):
    """POST /debate/{id}/inject-intervention body."""
    text: str = Field(description="Custom moderator intervention text")
    source: str = Field(default="human", description="Source label for logging")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class DebateInfo(BaseModel):
    """Summary info for a debate."""
    debate_id: str
    topic_id: str
    condition_id: str
    condition_label: str
    status: DebateStatus
    current_round: int = 0
    total_rounds: int = 10
    turn_count: int = 0
    intervention_count: int = 0
    created_at: Optional[str] = None
    error_message: Optional[str] = None


class TurnResponse(BaseModel):
    """A single turn in the transcript."""
    turn_id: str
    round_number: int
    turn_in_round: int
    agent_name: str
    text: str
    scores: Optional[dict] = None
    timestamp: Optional[str] = None


class InterventionResponse(BaseModel):
    """A moderator intervention event."""
    intervention_id: str
    round_number: int
    source: str
    strategy: str
    trigger_dimension: Optional[str] = None
    trigger_score: Optional[float] = None
    silent_control: bool = False
    intervention_text: str
    moderator_output: Optional[dict] = None
    habermas_output: Optional[dict] = None
    timestamp: Optional[str] = None


class TranscriptResponse(BaseModel):
    """Full transcript for a debate."""
    debate_id: str
    status: DebateStatus
    config: Optional[dict] = None
    turns: list[TurnResponse] = Field(default_factory=list)
    interventions: list[InterventionResponse] = Field(default_factory=list)
    round_summaries: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# WebSocket event types
# ---------------------------------------------------------------------------

class WSEventType(str, Enum):
    TURN = "turn"
    INTERVENTION = "intervention"
    ROUND_SUMMARY = "round_summary"
    STATUS_CHANGE = "status_change"
    ERROR = "error"


class WSEvent(BaseModel):
    """WebSocket message sent to connected clients."""
    event_type: WSEventType
    data: dict = Field(default_factory=dict)
