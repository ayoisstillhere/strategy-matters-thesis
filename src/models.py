"""
Data Models
============
Pydantic models for all structured data produced during a debate run.
These define the schema for structured logging, export, and analysis.

Schema hierarchy:
    DebateRun
    ├── config (DebateRunConfig)
    ├── turns[] (Turn)
    │   └── scores (DimensionScores)
    ├── interventions[] (InterventionEvent)
    └── round_summaries[] (RoundSummary)

See also:
    - expose.tex §4.6 — experiment design and logging spec
    - src/experiment_config.py — condition definitions
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dimension scores
# ---------------------------------------------------------------------------

class DimensionScores(BaseModel):
    """Seven-dimension discourse quality scores for a single turn."""
    civility: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    logical_consistency: int = Field(ge=1, le=5)
    argument_strength: int = Field(ge=1, le=5)
    document_grounding: int = Field(ge=1, le=5)
    responsiveness: int = Field(ge=1, le=5)
    stance_differentiation: int = Field(ge=1, le=5)

    def to_dict(self) -> dict[str, int]:
        return self.model_dump()

    @property
    def primary_dims(self) -> dict[str, int]:
        """Return only the 5 primary dimensions (used for plateau detection)."""
        return {
            "civility": self.civility,
            "argument_strength": self.argument_strength,
            "document_grounding": self.document_grounding,
            "responsiveness": self.responsiveness,
            "stance_differentiation": self.stance_differentiation,
        }

    @property
    def composite(self) -> float:
        """Weighted mean across all 7 dimensions (equal weights)."""
        vals = list(self.to_dict().values())
        return sum(vals) / len(vals)


class DimensionJustifications(BaseModel):
    """Per-dimension justification strings from the judge."""
    civility: str = ""
    relevance: str = ""
    logical_consistency: str = ""
    argument_strength: str = ""
    document_grounding: str = ""
    responsiveness: str = ""
    stance_differentiation: str = ""


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single agent turn in the debate."""
    turn_id: str = Field(default="", description="UUID for this turn")
    round_number: int = Field(ge=1, le=10)
    turn_in_round: int = Field(ge=1, le=6, description="1-indexed position within round")
    agent_name: str
    text: str
    scores: Optional[DimensionScores] = None
    justifications: Optional[DimensionJustifications] = None
    rag_passages_used: list[str] = Field(default_factory=list)
    token_count_input: int = 0
    token_count_output: int = 0
    latency_s: float = 0.0
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Intervention event
# ---------------------------------------------------------------------------

class InterventionSource(str, Enum):
    STRATEGY = "strategy"          # Strategies A-D via moderator LLM
    HABERMAS = "habermas"          # Baseline 3 unconditional
    RANDOM = "random"              # Baseline 4 random message pool
    SILENT_CONTROL = "silent"      # Trigger confirmed but silenced


class InterventionEvent(BaseModel):
    """A moderator intervention or trigger event."""
    intervention_id: str = ""
    after_turn_id: str = ""
    round_number: int = Field(ge=1, le=10)
    source: InterventionSource
    strategy: str = ""
    trigger_dimension: Optional[str] = None
    trigger_score: Optional[float] = None
    trigger_confirmed: bool = False
    silent_control: bool = False
    intervention_text: str = ""
    # Structured moderator output (strategy conditions)
    moderator_output: Optional[dict] = None
    # Habermas-specific output
    habermas_output: Optional[dict] = None
    token_count_input: int = 0
    token_count_output: int = 0
    latency_s: float = 0.0
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Round summary
# ---------------------------------------------------------------------------

class RoundSummary(BaseModel):
    """Aggregated scores for one round."""
    round_number: int = Field(ge=1, le=10)
    mean_scores: DimensionScores
    per_agent_scores: dict[str, DimensionScores] = Field(default_factory=dict)
    composite: float = 0.0
    plateau: bool = False
    intervention_count_so_far: int = 0


# ---------------------------------------------------------------------------
# Debate run config
# ---------------------------------------------------------------------------

class DebateRunConfig(BaseModel):
    """Configuration snapshot for a debate run, stored in the log."""
    topic_id: str
    topic_type: str
    framing_prompt: str
    condition_id: str
    condition_label: str
    condition_type: str  # "baseline" or "strategy"
    run_number: int
    agent_model: str
    judge_model: str
    moderator_model: str = ""
    num_rounds: int = 10
    num_agents: int = 6
    max_interventions: int = 3
    silent_control_rate: float = 0.20
    turn_order: list[str] = Field(default_factory=list)
    has_moderator_agent: bool = False
    uses_trigger: bool = False
    trigger_strategy: Optional[str] = None
    nudge_text: str = ""  # Non-empty for Baseline 2
    language: str = "de"  # "de" or "en"


# ---------------------------------------------------------------------------
# Full debate run
# ---------------------------------------------------------------------------

class DebateRun(BaseModel):
    """Complete structured log of one debate run."""
    run_id: str = ""
    config: DebateRunConfig
    turns: list[Turn] = Field(default_factory=list)
    interventions: list[InterventionEvent] = Field(default_factory=list)
    round_summaries: list[RoundSummary] = Field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_latency_s: float = 0.0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def active_intervention_count(self) -> int:
        """Count interventions that actually produced a moderator message."""
        return sum(
            1 for i in self.interventions
            if not i.silent_control and i.intervention_text
        )
