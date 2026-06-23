"""
Structured Logging and Data Export
====================================
Saves debate runs to JSON (full log) and exports to CSV/Parquet
for statistical analysis.

Output schema:
    runs/{condition_id}/{topic_id}/run_{N}.json — full DebateRun JSON
    analysis/turns.csv — flat table of all turns + scores (160 × 60 rows)
    analysis/interventions.csv — all intervention events
    analysis/round_summaries.csv — per-round aggregated scores

See also:
    - src/models.py — DebateRun schema
    - expose.tex §4.6 — logging specification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.models import DebateRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON export (per run)
# ---------------------------------------------------------------------------

def save_run_json(run: DebateRun, output_dir: Path) -> Path:
    """Save a single debate run as a JSON file.

    Directory structure:
        output_dir / {condition_id} / {topic_id} / run_{N}_{timestamp}.json

    Returns:
        Path to the saved JSON file.
    """
    from datetime import datetime, timezone

    run_dir = output_dir / run.config.condition_id / run.config.topic_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"run_{run.config.run_number:02d}_{ts}.json"
    path = run_dir / filename

    data = run.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved run JSON: {path}")
    return path


def load_run_json(path: Path) -> DebateRun:
    """Load a DebateRun from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DebateRun.model_validate(data)


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_all_runs(output_dir: Path) -> list[DebateRun]:
    """Load all run JSONs from the output directory."""
    runs = []
    for json_path in sorted(output_dir.rglob("run_*.json")):
        try:
            runs.append(load_run_json(json_path))
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    logger.info(f"Loaded {len(runs)} runs from {output_dir}")
    return runs


# ---------------------------------------------------------------------------
# CSV/Parquet export (for analysis)
# ---------------------------------------------------------------------------

def export_turns_csv(runs: list[DebateRun], output_path: Path) -> None:
    """Export a flat table of all turns with scores.

    Columns: run_id, topic_id, condition_id, condition_type,
             round_number, turn_in_round, agent_name, text,
             civility, relevance, ..., stance_differentiation, composite
    """
    import pandas as pd

    rows = []
    for run in runs:
        for turn in run.turns:
            row = {
                "run_id": run.run_id,
                "topic_id": run.config.topic_id,
                "topic_type": run.config.topic_type,
                "condition_id": run.config.condition_id,
                "condition_label": run.config.condition_label,
                "condition_type": run.config.condition_type,
                "run_number": run.config.run_number,
                "round_number": turn.round_number,
                "turn_in_round": turn.turn_in_round,
                "agent_name": turn.agent_name,
                "text": turn.text,
            }
            if turn.scores:
                row.update(turn.scores.to_dict())
                row["composite"] = round(turn.scores.composite, 2)
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Exported {len(df)} turns to {output_path}")


def export_interventions_csv(runs: list[DebateRun], output_path: Path) -> None:
    """Export a flat table of all intervention events."""
    import pandas as pd

    rows = []
    for run in runs:
        for event in run.interventions:
            row = {
                "run_id": run.run_id,
                "topic_id": run.config.topic_id,
                "condition_id": run.config.condition_id,
                "condition_label": run.config.condition_label,
                "run_number": run.config.run_number,
                "intervention_id": event.intervention_id,
                "round_number": event.round_number,
                "source": event.source.value,
                "strategy": event.strategy,
                "trigger_dimension": event.trigger_dimension,
                "trigger_score": event.trigger_score,
                "trigger_confirmed": event.trigger_confirmed,
                "silent_control": event.silent_control,
                "intervention_text": event.intervention_text,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Exported {len(df)} interventions to {output_path}")


def export_round_summaries_csv(runs: list[DebateRun], output_path: Path) -> None:
    """Export per-round aggregated scores."""
    import pandas as pd

    rows = []
    for run in runs:
        for rs in run.round_summaries:
            row = {
                "run_id": run.run_id,
                "topic_id": run.config.topic_id,
                "condition_id": run.config.condition_id,
                "condition_label": run.config.condition_label,
                "run_number": run.config.run_number,
                "round_number": rs.round_number,
                "composite": rs.composite,
                "plateau": rs.plateau,
                "intervention_count_so_far": rs.intervention_count_so_far,
            }
            row.update(rs.mean_scores.to_dict())
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Exported {len(df)} round summaries to {output_path}")


def export_all(
    runs: list[DebateRun],
    analysis_dir: Path,
    parquet: bool = False,
) -> None:
    """Export all analysis tables (turns, interventions, round summaries).

    Args:
        runs: List of DebateRun objects.
        analysis_dir: Directory for output files.
        parquet: If True, also export as Parquet (requires pyarrow).
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)

    export_turns_csv(runs, analysis_dir / "turns.csv")
    export_interventions_csv(runs, analysis_dir / "interventions.csv")
    export_round_summaries_csv(runs, analysis_dir / "round_summaries.csv")

    if parquet:
        try:
            import pandas as pd
            for csv_file in analysis_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                pq_file = csv_file.with_suffix(".parquet")
                df.to_parquet(pq_file, index=False)
                logger.info(f"Exported Parquet: {pq_file}")
        except ImportError:
            logger.warning("pyarrow not installed — skipping Parquet export")
