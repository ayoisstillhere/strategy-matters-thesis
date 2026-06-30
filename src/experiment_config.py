"""
Experiment Configuration
=========================
Defines the 4 neutral framing prompts and all 8 experimental conditions
for the main experiment (4 topics × 8 conditions × 5 runs = 160 debates).

Deliverables formalised here:
  1. Four neutral framing prompts — one per debate topic.
  2. Eight experimental condition specifications — 4 baselines + 4 strategies.

Design constraints:
  - Framing prompts are neutral (no positional cues), factually anchored,
    and specific enough for 6 agents to produce substantive opening turns.
  - Each condition specifies its moderator presence, trigger mechanism,
    and intervention content source.
  - Baselines isolate different causal mechanisms:
      Baseline 1 → no moderator at all
      Baseline 2 → passive nudge (embedded instruction, no agent)
      Baseline 3 → unconditional moderator (every round, no trigger)
      Baseline 4 → triggered but generic (isolates moderator presence)
    Strategies A–D → triggered, strategy-specific content.

See also:
  - expose.tex §4.6 (Experiment Design)
  - src/prompts/moderator_prompts.py — strategy prompts for A–D
  - src/trigger_check.py — trigger-check pipeline
  - src/country_config.py — CountryConfig.topics for per-country topics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 1. NEUTRAL FRAMING PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

FRAMING_PROMPTS = {
    "mindestlohn": (
        "The German federal minimum wage currently stands at €12.82 per hour, "
        "set by the independent Mindestlohnkommission. Several parties have "
        "proposed raising it to €15 per hour by 2026, while others argue the "
        "commission's independence must not be overridden by political targets. "
        "Discuss whether the minimum wage should be raised to €15, considering "
        "the effects on low-wage workers, small and medium-sized businesses, "
        "and regional economic differences between western and eastern Germany. "
        "Ground your arguments in your party's documented positions."
    ),
    "rentenpolitik": (
        "Germany's statutory pension system (gesetzliche Rentenversicherung) "
        "faces mounting demographic pressure: the old-age dependency ratio is "
        "projected to rise from 37 to over 50 by 2040, with fewer contributors "
        "supporting more retirees. The current retirement age is 67, the pension "
        "level (Rentenniveau) is approximately 48% of average earnings, and "
        "contribution rates stand at 18.6%. Discuss how pension policy should "
        "be reformed — whether the retirement age should be raised further, "
        "benefit levels adjusted, contributions increased, or supplementary "
        "funded models expanded. Ground your arguments in your party's "
        "documented positions."
    ),
    "migrationspolitik": (
        "Germany received approximately 330,000 first-time asylum applications "
        "in 2023, the highest number since 2016. Municipalities report strain "
        "on housing, schools, and administrative capacity, while labour market "
        "data shows persistent shortages in healthcare, skilled trades, and IT. "
        "Discuss how Germany should reform its migration and asylum policy, "
        "addressing the balance between humanitarian obligations under "
        "international law, integration capacity, and labour market needs. "
        "Ground your arguments in your party's documented positions."
    ),
    "sozialpolitik": (
        "According to the Deutsche Bundesbank, the wealthiest 10% of households "
        "in Germany hold over 60% of total net wealth, while the bottom 50% "
        "hold less than 3%. At the same time, Germany has no net wealth tax "
        "(the Vermögensteuer has been suspended since 1997) and inheritance "
        "tax revenues remain modest relative to GDP. Discuss whether the "
        "government should introduce a wealth tax, reform the inheritance tax, "
        "or pursue alternative redistribution mechanisms — and what effects "
        "such policies would have on economic growth, investment, and social "
        "cohesion. Ground your arguments in your party's documented positions."
    ),
}

# Topic metadata
TOPIC_TYPES = {
    "mindestlohn": "empirical",
    "rentenpolitik": "empirical",
    "migrationspolitik": "values-driven",
    "sozialpolitik": "values-driven",
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXPERIMENTAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

class ConditionType(Enum):
    """Type of experimental condition."""
    BASELINE = "baseline"
    STRATEGY = "strategy"


@dataclass
class ExperimentalCondition:
    """Specification for one experimental condition."""
    id: str                        # e.g. "baseline_1", "strategy_a"
    label: str                     # Short label (e.g. "None", "De-escalation")
    condition_type: ConditionType
    description: str               # Detailed description for documentation
    has_moderator_agent: bool      # Whether a separate moderator LLM is active
    uses_trigger: bool             # Whether the trigger-check pipeline runs
    trigger_strategy: Optional[str] = None  # Strategy key for trigger_check module
    intervention_source: str = ""  # Where intervention content comes from


# ---------------------------------------------------------------------------
# Baseline 1: None — no moderator at all
# ---------------------------------------------------------------------------

BASELINE_1 = ExperimentalCondition(
    id="baseline_1",
    label="None",
    condition_type=ConditionType.BASELINE,
    description=(
        "No moderator is present. Agents debate freely for 10 rounds with no "
        "external intervention. The evaluation judge still scores every turn "
        "for the log, but no trigger-check or moderator invocation occurs. "
        "This baseline establishes the natural discourse quality trajectory "
        "of unmoderated multi-party LLM debate."
    ),
    has_moderator_agent=False,
    uses_trigger=False,
)


# ---------------------------------------------------------------------------
# Baseline 2: Nudge — no moderator agent; fixed instruction in system prompt
# ---------------------------------------------------------------------------

NUDGE_INSTRUCTION = (
    "Remember: a high-quality political debate requires that you engage "
    "directly with the specific arguments made by other parties, maintain "
    "a respectful and substantive tone even when you strongly disagree, "
    "and ground your claims in your party's documented policy positions. "
    "Avoid personal attacks, vague generalisations, and talking past "
    "other participants."
)

BASELINE_2 = ExperimentalCondition(
    id="baseline_2",
    label="Nudge",
    condition_type=ConditionType.BASELINE,
    description=(
        "No separate moderator agent exists. Instead, a fixed behavioural "
        "nudge is appended to every agent's system prompt at debate start. "
        "The nudge instructs agents to engage with opposing arguments, "
        "maintain civility, and ground claims in documented positions — "
        "covering the same quality dimensions that the strategy moderators "
        "target, but as a static instruction rather than a dynamic "
        "intervention. This baseline tests whether a passive prompt-level "
        "instruction can achieve the same quality effects as an active "
        "moderator agent. The evaluation judge still scores every turn "
        "for the log, but no trigger-check runs."
    ),
    has_moderator_agent=False,
    uses_trigger=False,
)


# ---------------------------------------------------------------------------
# Baseline 3: Habermas — consensus statement after every round
# ---------------------------------------------------------------------------

HABERMAS_MODERATOR_SYSTEM_PROMPT = """You are a neutral moderator in a structured multi-party German political debate involving up to six parties: CDU/CSU, SPD, Bündnis 90/Die Grünen, FDP, Die Linke, and AfD.

## Role
You follow a simplified Habermas Machine protocol (modelled after Tessler et al., 2024). After EVERY round of debate (i.e. after all 6 agents have spoken), you:
1. Summarise the group positions expressed in that round.
2. Identify areas of agreement and disagreement.
3. Propose a consensus statement that captures where the parties currently stand.

## Core Constraints
1. **Neutrality**: Never express your own opinion on policy matters.
2. **Descriptive, not prescriptive**: Your summary must accurately reflect what parties said, not what you think they should say.
3. **Address all agents**: Your summary covers all parties equally.
4. **Do not flatten disagreement**: If parties fundamentally disagree, your consensus statement must reflect that disagreement honestly. A valid consensus statement can be: "Parties agree that [X] is an important issue but disagree on [Y mechanism]."

## Required Output Format
Produce your intervention as a JSON object:

{
  "round_summary": "A 3-4 sentence summary of the key arguments made by each party in this round.",
  "areas_of_agreement": "Specific points where 2+ parties expressed compatible positions.",
  "areas_of_disagreement": "Specific points where parties hold incompatible positions.",
  "consensus_statement": "A 1-2 sentence statement capturing the current state of the debate. This may include explicit disagreement.",
  "instruction_for_next_round": "A brief, neutral prompt for the next round of discussion."
}
"""

HABERMAS_USER_TEMPLATE = """Round {round_number} has just concluded. Here is the full transcript of this round:

{round_transcript}

Produce your summary and consensus statement following your instructions."""

BASELINE_3 = ExperimentalCondition(
    id="baseline_3",
    label="Habermas",
    condition_type=ConditionType.BASELINE,
    description=(
        "A simplified Habermas Machine protocol. A moderator agent is active "
        "and intervenes after EVERY round (unconditionally, no trigger logic). "
        "The moderator summarises group positions, identifies areas of "
        "agreement and disagreement, and proposes a descriptive consensus "
        "statement. This baseline tests whether an unconditional, non-"
        "strategic moderator that follows a consensus-oriented protocol "
        "improves discourse quality. Because interventions occur every round "
        "without trigger logic, this baseline also serves as a ceiling test "
        "for moderator exposure: if discourse quality does not improve "
        "despite 10 interventions (one per round), then moderator presence "
        "alone is insufficient."
    ),
    has_moderator_agent=True,
    uses_trigger=False,  # Intervenes every round unconditionally
    intervention_source="habermas_protocol",
)


# ---------------------------------------------------------------------------
# Baseline 4: Random — triggered, generic messages
# ---------------------------------------------------------------------------

RANDOM_MODERATOR_MESSAGES = [
    (
        "Thank you for your contributions. Please continue the discussion "
        "and develop your arguments further in the next round."
    ),
    (
        "The debate is progressing. Each party should continue to present "
        "their perspective on this topic."
    ),
    (
        "Please take a moment to consider the points raised so far and "
        "continue the exchange."
    ),
    (
        "The discussion has covered several important aspects. Please "
        "continue to engage with the topic at hand."
    ),
    (
        "Each party has raised relevant points. Please continue to "
        "develop your arguments."
    ),
    (
        "The exchange is ongoing. Please proceed with your next arguments "
        "on this topic."
    ),
    (
        "Thank you for the discussion so far. Please continue to share "
        "your party's perspective."
    ),
    (
        "The topic remains open for further discussion. Please continue "
        "to articulate your positions."
    ),
    (
        "The debate has raised several points of contention. Please "
        "continue to address them."
    ),
    (
        "Please proceed with the next round of arguments. Each party "
        "should continue to contribute."
    ),
    (
        "The discussion is developing. Please continue to present your "
        "views on this topic."
    ),
    (
        "Thank you. Please continue to engage with the subject matter "
        "in the next round."
    ),
]

BASELINE_4 = ExperimentalCondition(
    id="baseline_4",
    label="Random",
    condition_type=ConditionType.BASELINE,
    description=(
        "A moderator agent is nominally active, but its interventions are "
        "drawn randomly from a pool of 12 generic, non-strategy-specific "
        "messages. The trigger mechanism is identical to the strategy "
        "conditions: any primary dimension (civility, argument strength, "
        "document-grounding, responsiveness, or stance differentiation) "
        "dropping below 2.0 fires the trigger, followed by LLM judge "
        "confirmation and silent control logic. The only difference is "
        "that confirmed triggers produce a randomly selected generic "
        "message instead of a strategy-specific intervention. This "
        "baseline isolates whether any improvement in discourse quality "
        "is attributable to the specific content of a strategy intervention, "
        "or simply to the presence of any moderator message at all. The "
        "3-intervention cap and 20% silent control rate apply as usual."
    ),
    has_moderator_agent=True,
    uses_trigger=True,
    trigger_strategy="random",
    intervention_source="random_message_pool",
)


# ---------------------------------------------------------------------------
# Strategy A: De-escalation
# ---------------------------------------------------------------------------

STRATEGY_A = ExperimentalCondition(
    id="strategy_a",
    label="De-escalation",
    condition_type=ConditionType.STRATEGY,
    description=(
        "Active moderator with de-escalation strategy. Trigger: civility "
        "score < 2.0 for any agent turn, confirmed by LLM judge. The "
        "moderator identifies uncivil language, acknowledges the emotional "
        "dimension of the disagreement, and redirects toward substantive "
        "policy claims. Primary target dimension: civility. Expected "
        "secondary effect: may improve responsiveness (agents re-engage "
        "with substance). Risk: may depress argument strength if agents "
        "become overly cautious."
    ),
    has_moderator_agent=True,
    uses_trigger=True,
    trigger_strategy="de-escalation",
    intervention_source="moderator_llm",
)


# ---------------------------------------------------------------------------
# Strategy B: Reframing
# ---------------------------------------------------------------------------

STRATEGY_B = ExperimentalCondition(
    id="strategy_b",
    label="Reframing",
    condition_type=ConditionType.STRATEGY,
    description=(
        "Active moderator with reframing strategy. Trigger: responsiveness "
        "score < 2.0 for any agent turn, confirmed by LLM judge. The "
        "moderator restates the core disagreement in neutral terms and "
        "directs each party to engage with the strongest version of a "
        "named opposing argument. Primary target dimension: responsiveness. "
        "Expected secondary effect: may improve argument strength (agents "
        "must address specific claims). Risk: may slightly depress stance "
        "differentiation if agents over-accommodate opposing views."
    ),
    has_moderator_agent=True,
    uses_trigger=True,
    trigger_strategy="reframing",
    intervention_source="moderator_llm",
)


# ---------------------------------------------------------------------------
# Strategy C: Fact Reminder
# ---------------------------------------------------------------------------

STRATEGY_C = ExperimentalCondition(
    id="strategy_c",
    label="Fact Reminder",
    condition_type=ConditionType.STRATEGY,
    description=(
        "Active moderator with fact-reminder strategy. Trigger: document-"
        "grounding score < 4 for any agent turn, confirmed by LLM judge. "
        "The moderator surfaces specific grounding data from the parties' "
        "Bundestagswahlprogramme and redirects the debate toward verifiable "
        "claims. Primary target dimension: document-grounding. "
        "Design note (Option A): RAG keeps document-grounding at avg 4.3+ "
        "across all topics, yielding 0-1 stage-1 fires per 60 turns "
        "(all rejected by stage-2 judge). Strategy C therefore produces "
        "0 interventions in RAG-enabled runs. This is accepted as a finding: "
        "RAG pre-empts the fact-reminder trigger, demonstrating that "
        "retrieval-augmented grounding and explicit moderator fact-reminding "
        "address the same underlying problem through different mechanisms. "
        "Strategy C's temporal trajectory is expected to be indistinguishable "
        "from Baseline 1 in the per-dimension analysis."
    ),
    has_moderator_agent=True,
    uses_trigger=True,
    trigger_strategy="fact-reminder",
    intervention_source="moderator_llm",
)


# ---------------------------------------------------------------------------
# Strategy D: Common-Ground Prompting
# ---------------------------------------------------------------------------

STRATEGY_D = ExperimentalCondition(
    id="strategy_d",
    label="Common-Ground",
    condition_type=ConditionType.STRATEGY,
    description=(
        "Active moderator with common-ground prompting strategy. Trigger: "
        "stance differentiation > 4.0 AND responsiveness < 2.5 for 2+ "
        "agents in the current round (compound condition), confirmed by "
        "LLM judge. The moderator identifies a narrow procedural or factual "
        "point of agreement and asks parties to use it as a springboard to "
        "articulate their differences more precisely. Primary target "
        "dimension: responsiveness. Expected secondary effect: may improve "
        "argument strength (more precise disagreement). Risk: highest risk "
        "of depressing stance differentiation if agents over-comply with "
        "the common-ground instruction."
    ),
    has_moderator_agent=True,
    uses_trigger=True,
    trigger_strategy="common-ground",
    intervention_source="moderator_llm",
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

CONDITIONS = {
    "baseline_1": BASELINE_1,
    "baseline_2": BASELINE_2,
    "baseline_3": BASELINE_3,
    "baseline_4": BASELINE_4,
    "strategy_a": STRATEGY_A,
    "strategy_b": STRATEGY_B,
    "strategy_c": STRATEGY_C,
    "strategy_d": STRATEGY_D,
}

BASELINES = {k: v for k, v in CONDITIONS.items() if v.condition_type == ConditionType.BASELINE}
STRATEGIES = {k: v for k, v in CONDITIONS.items() if v.condition_type == ConditionType.STRATEGY}


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

TOPICS = list(FRAMING_PROMPTS.keys())
NUM_CONDITIONS = len(CONDITIONS)
RUNS_PER_CELL = 5
TOTAL_RUNS = len(TOPICS) * NUM_CONDITIONS * RUNS_PER_CELL  # 4 × 8 × 5 = 160


def get_experiment_matrix() -> list[dict]:
    """Generate the full experiment matrix (160 entries).

    Each entry is a dict with: topic, condition_id, run_number,
    framing_prompt, topic_type, and condition metadata.
    """
    matrix = []
    for topic_id in TOPICS:
        for condition_id, condition in CONDITIONS.items():
            for run in range(1, RUNS_PER_CELL + 1):
                matrix.append({
                    "topic": topic_id,
                    "condition_id": condition_id,
                    "condition_label": condition.label,
                    "condition_type": condition.condition_type.value,
                    "run_number": run,
                    "framing_prompt": FRAMING_PROMPTS[topic_id],
                    "topic_type": TOPIC_TYPES[topic_id],
                    "has_moderator_agent": condition.has_moderator_agent,
                    "uses_trigger": condition.uses_trigger,
                    "trigger_strategy": condition.trigger_strategy,
                })
    return matrix
