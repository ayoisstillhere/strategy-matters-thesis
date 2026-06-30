"""
Prompt Refinement Pilots
=========================
Addresses the "Prompt Refinement" step in action_plan.html.

  Task 1 — Validate framing prompts across all 4 topics
    Run baseline_1 (no moderation), 3 rounds, for each topic.
    Check: all 6 agents engage substantively, no generic/off-topic responses.
    Output: per-topic score table + any flagged turns (relevance < 4).

  Task 2 — Test persona consistency across 10 rounds
    Run baseline_1, 10 rounds, mindestlohn topic.
    Check: stance_differentiation stays high throughout; no persona drift.
    Output: per-round stance_differentiation scores.

  Task 3 — Test moderator intervention quality for each strategy (A–D)
    Run 3-round debate per strategy with RAG enabled.
    Check each intervention: has diagnosis, target_parties, intervention_text.
    Output: intervention texts printed for manual review.

  Task 4 — Trigger threshold calibration
    DONE: thresholds already set to < 4 in trigger_check.py.
    This script just prints a summary of trigger behaviour from today's runs.

Usage:
    cd strategy-matters-thesis

    # All tasks:
    python pilots/prompt_refinement.py

    # Individual tasks:
    python pilots/prompt_refinement.py --task 1
    python pilots/prompt_refinement.py --task 2
    python pilots/prompt_refinement.py --task 3
    python pilots/prompt_refinement.py --task 4

Requires:
    - GROQ_API_KEY in .env
    - FAISS indices at data/embeddings/ (for Task 3)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm_client import LLMClient
from src.debate_engine import DebateEngine
from src.rag_pipeline import RAGPipeline
from src.experiment_config import FRAMING_PROMPTS, TOPIC_TYPES
from src.export import save_run_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prompt_refinement")

OUTPUT_DIR = PROJECT_ROOT / "runs" / "prompt_refinement"
ALL_TOPICS = ["mindestlohn", "rentenpolitik", "migrationspolitik", "sozialpolitik"]
STRATEGIES = ["strategy_a", "strategy_b", "strategy_c", "strategy_d"]
STRATEGY_LABELS = {
    "strategy_a": "A — De-escalation",
    "strategy_b": "B — Reframing",
    "strategy_c": "C — Fact Reminder",
    "strategy_d": "D — Common-Ground",
}


# ── Task 1: Framing prompt validation ─────────────────────────────────────

def task1_framing_validation(client: LLMClient) -> None:
    print("\n" + "=" * 65)
    print("  TASK 1: Framing Prompt Validation")
    print("  baseline_1 | 3 rounds | no RAG | all 4 topics")
    print("=" * 65)

    issues = []

    for topic in ALL_TOPICS:
        logger.info(f"  Running: {topic}")
        engine = DebateEngine(
            topic_id=topic,
            framing_prompt=FRAMING_PROMPTS[topic],
            topic_type=TOPIC_TYPES.get(topic, "empirical"),
            condition_id="baseline_1",
            run_number=1,
            llm_client=client,
            rag_pipeline=None,
            num_rounds=3,
        )
        result = engine.run()
        save_run_json(result, OUTPUT_DIR / "framing")

        turns = result.turns
        all_relevance = [t.scores.to_dict()["relevance"] for t in turns if t.scores]
        all_stance   = [t.scores.to_dict()["stance_differentiation"] for t in turns if t.scores]

        low_relevance = [t for t in turns if t.scores
                         and t.scores.to_dict()["relevance"] < 4]
        avg_rel   = round(mean(all_relevance), 2) if all_relevance else 0
        avg_stance = round(mean(all_stance), 2) if all_stance else 0

        status = "✓" if not low_relevance else f"⚠  {len(low_relevance)} low-relevance turns"
        print(f"\n  {topic:<22} avg_relevance={avg_rel}  avg_stance={avg_stance}  {status}")

        if low_relevance:
            for t in low_relevance[:3]:
                preview = t.text[:120].replace("\n", " ")
                print(f"    R{t.round_number} {t.agent_name}: relevance={t.scores.to_dict()['relevance']} "
                      f"→ \"{preview}…\"")
                issues.append((topic, t.agent_name, t.round_number,
                               t.scores.to_dict()["relevance"], preview))

    print()
    if not issues:
        print("  ✓ All framing prompts valid — no off-topic turns detected")
    else:
        print(f"  ⚠  {len(issues)} low-relevance turn(s) found — review framing prompts above")
    print()


# ── Task 2: Persona consistency across 10 rounds ──────────────────────────

def task2_persona_consistency(client: LLMClient) -> None:
    print("\n" + "=" * 65)
    print("  TASK 2: Persona Consistency — 10 rounds")
    print("  baseline_1 | 10 rounds | no RAG | mindestlohn")
    print("=" * 65)

    engine = DebateEngine(
        topic_id="mindestlohn",
        framing_prompt=FRAMING_PROMPTS["mindestlohn"],
        topic_type="empirical",
        condition_id="baseline_1",
        run_number=1,
        llm_client=client,
        rag_pipeline=None,
        num_rounds=10,
    )
    result = engine.run()
    save_run_json(result, OUTPUT_DIR / "persona")

    # Per-round per-party stance_differentiation
    PARTIES = ["CDU/CSU", "SPD", "Bündnis 90/Die Grünen", "FDP", "Die Linke", "AfD"]
    print(f"\n  {'Party':<25} " + "  ".join(f"R{r:02d}" for r in range(1, 11)))
    print(f"  {'-'*25} " + "  ".join("---" for _ in range(10)))

    drift_parties = []
    for party in PARTIES:
        party_turns = [t for t in result.turns if t.agent_name == party and t.scores]
        scores_by_round = {t.round_number: t.scores.to_dict()["stance_differentiation"]
                           for t in party_turns}
        row = [str(scores_by_round.get(r, "-")) for r in range(1, 11)]
        vals = [scores_by_round[r] for r in range(1, 11) if r in scores_by_round]
        drift = max(vals) - min(vals) if vals else 0
        flag = " ⚠ DRIFT" if drift >= 2 else ""
        print(f"  {party:<25} " + "  ".join(f"{s:<3}" for s in row) + flag)
        if drift >= 2:
            drift_parties.append(party)

    print()
    if not drift_parties:
        print("  ✓ No persona drift detected (stance_diff range < 2 for all parties)")
    else:
        print(f"  ⚠  Drift detected for: {', '.join(drift_parties)}")
        print("     Review their persona prompts in src/prompts/agent_prompts.py")
    print()


# ── Task 3: Moderator intervention quality per strategy ───────────────────

def task3_moderator_quality(client: LLMClient) -> None:
    print("\n" + "=" * 65)
    print("  TASK 3: Moderator Intervention Quality — Strategies A–D")
    print("  3 rounds each | RAG enabled | sozialpolitik")
    print("=" * 65)

    rag = RAGPipeline()

    for condition_id in STRATEGIES:
        label = STRATEGY_LABELS[condition_id]
        logger.info(f"  Running: {label}")

        engine = DebateEngine(
            topic_id="sozialpolitik",
            framing_prompt=FRAMING_PROMPTS["sozialpolitik"],
            topic_type="values-driven",
            condition_id=condition_id,
            run_number=1,
            llm_client=client,
            rag_pipeline=rag,
            num_rounds=3,
        )
        result = engine.run()
        save_run_json(result, OUTPUT_DIR / "moderator")

        active_ivs = [i for i in result.interventions if i.intervention_text]
        silent_ivs = [i for i in result.interventions if i.silent_control]

        print(f"\n  ── Strategy {label} ──")
        print(f"     Active interventions: {len(active_ivs)}  |  Silent: {len(silent_ivs)}")

        if not active_ivs:
            print("     (No active interventions fired — triggers not met in 3 rounds)")
        else:
            for iv in active_ivs:
                print(f"\n     Round {iv.round_number} | dim={iv.trigger_dimension} "
                      f"score={iv.trigger_score}")
                if iv.moderator_output:
                    diag = iv.moderator_output.get("diagnosis", "")
                    print(f"     Diagnosis: {diag[:120]}")
                print(f"     Intervention: {iv.intervention_text[:200]}…")

                # Structural checks
                ok_structure = (
                    iv.intervention_text != "" and
                    iv.moderator_output is not None and
                    iv.moderator_output.get("diagnosis") and
                    iv.moderator_output.get("target_parties") and
                    iv.moderator_output.get("intervention_text")
                )
                print(f"     Structure valid: {'✓' if ok_structure else '✗ MISSING FIELDS'}")

    print()


# ── Task 4: Trigger calibration summary ───────────────────────────────────

def task4_trigger_calibration() -> None:
    print("\n" + "=" * 65)
    print("  TASK 4: Trigger Calibration Summary")
    print("  (from existing runs — no new runs needed)")
    print("=" * 65)

    runs_dir = PROJECT_ROOT / "runs" / "demo"
    run_files = sorted(runs_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)

    if not run_files:
        print("  No runs found in runs/demo/")
        return

    print(f"\n  {'Condition':<14} {'Topic':<20} {'Rounds':<8} {'Active':<8} {'Silent':<8} {'Cap'}")
    print(f"  {'-'*14} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    for f in run_files[-10:]:  # last 10 runs
        data = json.loads(f.read_text(encoding="utf-8"))
        cfg = data["config"]
        ivs = data.get("interventions", [])
        active = sum(1 for i in ivs if i.get("intervention_text"))
        silent = sum(1 for i in ivs if i.get("silent_control"))
        cap = cfg.get("max_interventions", 3)
        cap_reached = "⚠ CAP" if active >= cap else ""
        print(f"  {cfg['condition_id']:<14} {cfg['topic_id']:<20} "
              f"{cfg['num_rounds']:<8} {active:<8} {silent:<8} {cap_reached}")

    print()
    print("  Current thresholds (post-fix):")
    print("    Strategy A: civility < 4")
    print("    Strategy B: responsiveness < 4")
    print("    Strategy C: argument_strength < 4  [CHANGED from document_grounding]")
    print("    Strategy D: stance_diff > 3 AND resp < 4 for 2+ agents")
    print("    Stage-2 judge confirmation: score <= 3")
    print("    Silent control: 20% of confirmed triggers")
    print("    Max interventions per run: 3")
    print()


# ── Task 5: Strategy A & B post-fix intervention review ───────────────────

def task5_strategy_ab_review(client: LLMClient) -> None:
    print("\n" + "=" * 65)
    print("  TASK 5: Strategy A & B Post-Fix Intervention Review")
    print("  5-round debates | RAG enabled | sozialpolitik")
    print("  Validates: intervention structure, strategy-specificity,")
    print("  judge 700-token fix (no score defaults to 3)")
    print("=" * 65)

    rag = RAGPipeline()

    for condition_id, label in [("strategy_a", "A — De-escalation"),
                                  ("strategy_b", "B — Reframing")]:
        logger.info(f"  Running: Strategy {label}")
        engine = DebateEngine(
            topic_id="sozialpolitik",
            framing_prompt=FRAMING_PROMPTS["sozialpolitik"],
            topic_type="values-driven",
            condition_id=condition_id,
            run_number=1,
            llm_client=client,
            rag_pipeline=rag,
            num_rounds=5,
        )
        result = engine.run()
        save_run_json(result, OUTPUT_DIR / "strategy_ab_review")

        turns = result.turns
        active_ivs = [i for i in result.interventions if i.intervention_text]
        silent_ivs = [i for i in result.interventions if i.silent_control]

        # Judge fix validation: count score-3 defaults
        defaulted = sum(
            1 for t in turns
            if t.scores and all(
                v == 3 for v in t.scores.to_dict().values()
            )
        )

        print(f"\n  ── Strategy {label} ──")
        print(f"     Turns: {len(turns)}  |  Active: {len(active_ivs)}  "
              f"|  Silent: {len(silent_ivs)}  |  All-3 defaults: {defaulted}")
        if defaulted > 0:
            print(f"     ⚠  {defaulted} turns have all scores = 3 — judge truncation may still occur")
        else:
            print("     ✓ No all-3 score defaults — judge 700-token fix confirmed")

        if not active_ivs:
            print("     (No active interventions in 5 rounds — "
                  "try a longer run or check trigger thresholds)")
        else:
            for iv in active_ivs:
                print(f"\n     Round {iv.round_number} | dim={iv.trigger_dimension} "
                      f"score={iv.trigger_score}")
                if iv.moderator_output:
                    diag = iv.moderator_output.get("diagnosis", "")
                    target = iv.moderator_output.get("target_parties", [])
                    print(f"     Diagnosis:      {diag[:140]}")
                    print(f"     Target parties: {target}")
                print(f"     Intervention:   {iv.intervention_text[:250]}...")
                ok_structure = (
                    iv.intervention_text != "" and
                    iv.moderator_output is not None and
                    iv.moderator_output.get("diagnosis") and
                    iv.moderator_output.get("target_parties") and
                    iv.moderator_output.get("intervention_text")
                )
                print(f"     Structure valid: {'✓' if ok_structure else '✗ MISSING FIELDS'}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prompt Refinement Pilots")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific task (1-5)")
    args = parser.parse_args()

    run_all = args.task is None

    try:
        client = LLMClient()
    except ValueError as e:
        logger.error(f"Set GROQ_API_KEY in .env: {e}")
        sys.exit(1)

    if run_all or args.task == 1:
        task1_framing_validation(client)
    if run_all or args.task == 2:
        task2_persona_consistency(client)
    if run_all or args.task == 3:
        task3_moderator_quality(client)
    if run_all or args.task == 4:
        task4_trigger_calibration()
    if run_all or args.task == 5:
        task5_strategy_ab_review(client)

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("  Review output above for any ⚠ warnings before proceeding.\n")


if __name__ == "__main__":
    main()
