# /src — Core System Code

This folder contains the main implementation of the AI-moderated multi-agent debate system.

**Prompt templates:**
- `prompts/agent_prompts.py` — 6 political agent system prompts (persona + shared rules)
- `prompts/moderator_prompts.py` — Moderator base prompt + 4 strategy variants + user prompt template
- `prompts/judge_prompts.py` — Evaluation judge (7-dim rubrics) + trigger-check judge prompts

**Core framework:**
- `debate_engine.py` — Orchestrator: round-robin turns, judge scoring, trigger checks, moderator dispatch, plateau detection
- `agent.py` — PoliticalAgent: persona prompt + RAG grounding + context assembly + LLM call
- `judge.py` — EvaluationJudge (7-dim scoring) + TriggerJudge (single-dim confirmation)
- `moderator.py` — Moderator: strategy interventions, Habermas summaries, random messages, silent control
- `trigger_check.py` — Two-stage trigger pipeline (rule-based + LLM judge confirmation + silent control)
- `rag_pipeline.py` — Per-turn RAG retrieval + re-ranking (cosine, multilingual bi-encoder)

**Configuration and data:**
- `models.py` — Pydantic models (Turn, InterventionEvent, DebateRun, DimensionScores, etc.)
- `llm_client.py` — Groq API wrapper with SSL bypass, retries, JSON parsing
- `experiment_config.py` — 4 framing prompts, 8 condition specs, nudge text, Habermas prompt, random message pool
- `country_config.py` — CountryConfig abstraction (Germany production + Netherlands placeholder)
- `export.py` — Structured logging (JSON per run) + CSV/Parquet export for analysis
