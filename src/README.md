# /src — Core System Code

This folder contains the main implementation of the AI-moderated multi-agent debate system.

**Current modules:**
- `prompts/agent_prompts.py` — 6 political agent system prompts (persona + shared rules)
- `prompts/moderator_prompts.py` — Moderator base prompt + 4 strategy variants + user prompt template
- `prompts/judge_prompts.py` — Evaluation judge (7-dim rubrics) + trigger-check judge prompts
- `trigger_check.py` — Two-stage trigger pipeline (rule-based + LLM judge confirmation + silent control)
- `rag_pipeline.py` — Per-turn RAG retrieval + re-ranking (cosine, multilingual bi-encoder)
- `country_config.py` — CountryConfig abstraction (Germany production + Netherlands placeholder)

**Will contain:**
- Debate orchestration engine (round-robin turn management, state machine)
- AI moderator module (base + 4 pluggable strategy variants)
- LLM judge modules (trigger judge + evaluation judge)
- Structured logging and data export utilities
- Shared config, data models (Pydantic), and LLM API wrappers
