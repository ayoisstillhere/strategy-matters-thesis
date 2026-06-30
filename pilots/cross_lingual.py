"""
Cross-Lingual Pilot
====================
Runs 1 topic × 2 conditions × 2 languages (EN / DE) and compares
per-dimension judge scores side-by-side.

Decision criterion (per action plan):
  If |EN_mean − DE_mean| > 0.5 for document_grounding → restrict the
  main experiment to German only.

Setup:
  topic      : mindestlohn (empirical, well-covered in programme data)
  conditions : baseline_1 (no moderation) + strategy_b (reframing)
  rounds     : 3 (sufficient to compare scoring behaviour)
  languages  : "en" (default) and "de" (German instruction injected)

Usage:
    cd strategy-matters-thesis
    python pilots/cross_lingual.py

Output:
  runs/cross_lingual/<lang>/<condition>/<topic>/run_*.json
  Console: per-dimension mean table EN vs DE + divergence flags
"""

import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.rag_pipeline import RAGPipeline
from src.experiment_config import FRAMING_PROMPTS
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cross_lingual")

OUTPUT_DIR = PROJECT_ROOT / "runs" / "cross_lingual"
TOPIC      = "mindestlohn"
CONDITIONS = ["baseline_1", "strategy_b"]
LANGUAGES  = ["en", "de"]
NUM_ROUNDS = 3

DIMENSIONS = [
    "civility", "relevance", "logical_consistency",
    "argument_strength", "document_grounding",
    "responsiveness", "stance_differentiation",
]

DIVERGENCE_THRESHOLD = 0.5   # flag if |EN − DE| exceeds this


# ---------------------------------------------------------------------------
# Run a single debate and return (condition_id, language, dim → [scores])
# ---------------------------------------------------------------------------

def run_debate(
    client: LLMClient,
    rag: RAGPipeline,
    condition_id: str,
    language: str,
) -> dict[str, list[int]]:
    """Run one debate, return per-dimension score lists across all turns."""
    logger.info(f"  Running: {condition_id} | lang={language}")
    engine = DebateEngine(
        topic_id=TOPIC,
        framing_prompt=FRAMING_PROMPTS[TOPIC],
        topic_type="empirical",
        condition_id=condition_id,
        run_number=1,
        llm_client=client,
        rag_pipeline=rag,
        num_rounds=NUM_ROUNDS,
        language=language,
    )
    result = engine.run()
    save_run_json(result, OUTPUT_DIR / language / condition_id / TOPIC)

    dim_scores: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    for turn in result.turns:
        if turn.scores:
            for dim in DIMENSIONS:
                dim_scores[dim].append(turn.scores.to_dict()[dim])
    return dim_scores


# ---------------------------------------------------------------------------
# Compare two score dicts and print side-by-side table
# ---------------------------------------------------------------------------

def compare_and_print(
    condition_id: str,
    en_scores: dict[str, list[int]],
    de_scores: dict[str, list[int]],
) -> list[str]:
    """Print per-dimension comparison table. Returns list of flagged dimensions."""
    print(f"\n  Condition: {condition_id}")
    print(f"  {'Dimension':<24} {'EN mean':>8} {'DE mean':>8} {'Δ':>8}  Flag")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}  ----")

    flagged = []
    for dim in DIMENSIONS:
        en_vals = en_scores[dim]
        de_vals = de_scores[dim]
        en_mean = mean(en_vals) if en_vals else 0.0
        de_mean = mean(de_vals) if de_vals else 0.0
        delta = de_mean - en_mean
        flag = "⚠  DIVERGE" if abs(delta) > DIVERGENCE_THRESHOLD else "✓"
        if abs(delta) > DIVERGENCE_THRESHOLD:
            flagged.append(dim)
        sign = "+" if delta >= 0 else ""
        print(f"  {dim:<24} {en_mean:>8.2f} {de_mean:>8.2f} {sign}{delta:>7.2f}  {flag}")
    return flagged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  CROSS-LINGUAL PILOT")
    print(f"  topic={TOPIC} | conditions={CONDITIONS} | rounds={NUM_ROUNDS}")
    print(f"  languages: EN vs DE")
    print(f"  divergence threshold: {DIVERGENCE_THRESHOLD} per dimension")
    print("=" * 65)

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Set GROQ_API_KEY in .env: {e}")
        sys.exit(1)

    rag = RAGPipeline()

    # Store scores: results[lang][condition] = dim_scores_dict
    results: dict[str, dict[str, dict[str, list[int]]]] = {
        lang: {} for lang in LANGUAGES
    }

    for lang in LANGUAGES:
        for condition_id in CONDITIONS:
            results[lang][condition_id] = run_debate(client, rag, condition_id, lang)

    print("\n" + "=" * 65)
    print("  RESULTS — EN vs DE per-dimension means")
    print("=" * 65)

    all_flagged: dict[str, list[str]] = {}
    for condition_id in CONDITIONS:
        flagged = compare_and_print(
            condition_id,
            results["en"][condition_id],
            results["de"][condition_id],
        )
        all_flagged[condition_id] = flagged

    # Overall verdict
    print("\n" + "=" * 65)
    print("  VERDICT")
    print("=" * 65)

    doc_ground_flagged = any(
        "document_grounding" in flags for flags in all_flagged.values()
    )
    any_flagged = any(flags for flags in all_flagged.values())

    if doc_ground_flagged:
        print("  ⚠  document_grounding DIVERGES between EN and DE (>0.5).")
        print("  RECOMMENDATION: Restrict main experiment to German (language='de').")
        print("  This prevents the judge from scoring English claims against")
        print("  German-language RAG passages inconsistently.")
    elif any_flagged:
        flagged_dims = sorted({d for f in all_flagged.values() for d in f})
        print(f"  ⚠  Divergence in: {', '.join(flagged_dims)}")
        print("  document_grounding stable — EN run is acceptable.")
        print("  Note flagged dimensions as a threat in §Threats to Validity.")
    else:
        print("  ✓ No dimension diverges by more than 0.5 between EN and DE.")
        print("  Both language modes produce comparable judge scores.")
        print("  Main experiment may run in either language.")

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
