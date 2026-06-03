# Feasibility Report: Early LLM Runs

**Date:** June 3, 2026  
**Author:** Ayodele Fagbami  
**Thesis:** AI Moderation Techniques in LLM-Simulated Political Debates  
**Supervisor:** Michael  

---

## Summary

Three pilot tests were conducted to validate the core thesis design before committing to full implementation. All three tests passed successfully using free-tier Groq API access.

---

## Test 1: Multi-Round Debate (2 agents, 3 rounds)

**Setup:** CDU/CSU vs SPD debate on minimum wage (€15/hour), 3 rounds, no RAG, no moderation.  
**Model:** llama-3.1-8b-instant (Groq, free tier)  
**Tokens used:** 4,067

**Result: ✓ Argumentation evolves across rounds.**

- Agents directly reference each other's specific claims (not parallel monologues)
- Arguments deepen over rounds: Round 1 (general positions) → Round 2 (Denmark comparison, Mindestlohnkommission) → Round 3 (EU funds vs structural reform debate)
- Both agents introduce new sub-arguments in response to challenges

**Issues noted:**
- Minor prefix duplication in output (`[CDU/CSU]: [CDU/CSU]:`) — cosmetic, easy to strip
- Without RAG, agents hallucinate data (e.g., specific claims about Denmark unemployment) — confirms RAG is necessary for the full system

---

## Test 2: Moderator Intervention (Reframing Strategy)

**Setup:** Same debate, 4 rounds total: 2 free rounds → reframing intervention → 2 more rounds.  
**Model:** llama-3.1-8b-instant (Groq, free tier)  
**Tokens used:** 6,992

**Result: ✓ Moderator intervention steers debate behavior.**

- CDU/CSU immediately addressed the specific argument the moderator assigned (SPD's eastern Germany growth claim)
- SPD engaged with the Mindestlohnkommission independence concern (fully by Round 4)
- Post-intervention arguments were distinct from pre-intervention ones — agents did not simply repeat
- One agent (SPD) needed one extra turn to fully comply with the directive

**Issues noted:**
- SPD briefly reverted to generic "social justice" framing in Round 3 before complying in Round 4 — suggests prompt strength matters; may need per-strategy prompt refinement during calibration
- No design changes required. The reframing strategy works as intended.

---

## Test 3: LLM-as-a-Judge Scoring (7 dimensions, 3 reruns)

**Setup:** Score all 8 turns from Test 2 on 7 evaluation dimensions (civility, relevance, logical consistency, argument strength, document-grounding, responsiveness, stance differentiation). Each turn scored 3 times.  
**Judge model:** llama-3.3-70b-versatile (Groq, free tier) — different from agent model to avoid self-evaluation bias.  
**Tokens used:** 34,184

**Result: ✓ Judge produces reliable, structured scores.**

| Metric | Value |
|--------|-------|
| Parse errors | 0/24 (all valid JSON) |
| Exact agreement (3 reruns) | **98.8%** |
| Within ±1 agreement | **100%** |
| Max deviation on any dimension | 1 point (single instance) |

**Score differentiation observed:**
- SPD Round 3 correctly received lower scores (argument_strength=3, relevance=4, logical_consistency=4) — the judge detected the weak response where SPD drifted to generic claims
- CDU Round 2 civility docked to 4 for slightly confrontational framing
- All other turns scored appropriately high

**Issues noted:**
- Score inflation: many turns get all 5s. For the full experiment, calibration anchors (what does a "3" look like?) should be added to the judge prompt.
- Responsiveness stayed at 5.0 pre- and post-intervention — the judge may need more discriminating criteria for this dimension specifically.
- Neither issue requires design changes; both can be addressed with prompt refinement during Phase 4 (Pilot Testing & Calibration).

---

## Key Decisions Confirmed

| Design Element | Status |
|---------------|--------|
| Multi-round debate evolves | ✓ Confirmed — no parallel monologues |
| Moderator can steer debate | ✓ Confirmed — agents follow directives |
| LLM-as-a-judge is reliable | ✓ Confirmed — 98.8% exact consistency |
| Agent model (8B class) | ✓ Sufficient for debate — responsive, in-character |
| Judge model (70B class) | ✓ Sufficient — structured output, differentiated scores |
| Free tier viable for pilots | ✓ All tests ran on Groq free tier (~45K tokens total) |

## Cross-Check Against Exposé

**Aligned:** Reply-target model (agents name opponent's claim before responding), reframing strategy structure matches exposé example (lines 368–376), 7 evaluation dimensions on 1–5 scale, judge model stronger than agent model, Mindestlohn topic used, RAG confirmed necessary.

**Acceptable gaps for feasibility (to address in Phase 4):**

- Only 2 of 6 parties tested (CDU/CSU, SPD) — sufficient for validation; full 6-party test in Pilot Testing
- Only 3–4 rounds instead of 10 — enough to prove evolution; full length in Pilot Testing
- Only one agent model tested (llama-3.1-8b) — model comparison is Phase 4 per exposé: "Final selection will be made after comparative pilot tests"
- Moderator addressed 2 agents by name — 6-party version must address all agents simultaneously (exposé line 409)

**Action item for Phase 2 (Design):**

- Moderator intervention prompt is missing element 5 from the exposé's required structure (lines 349–355): *"Where appropriate, a warning not to flatten legitimate party differences."* This must be added to all moderator prompt templates during design phase.

## No Design Changes Required

The system architecture described in the exposé is realizable with current LLMs. Proceed to Phase 1 (Literature Review) and Phase 2 (Design) as planned.

---

*Transcript files and raw JSON scores available in `pilots/feasibility/outputs/`*
