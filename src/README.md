# /src — Core System Code

This folder contains the main implementation of the AI-moderated multi-agent debate system.

**Current modules:**
- `prompts/agent_prompts.py` — 6 political agent system prompts (persona + shared rules)

**Will contain:**
- Debate orchestration engine (round-robin turn management, state machine)
- `prompts/moderator_prompts.py` — Moderator base + 4 strategy variant prompts
- `prompts/judge_prompts.py` — LLM judge prompt templates (7 dimensions)
- AI moderator module (base + 4 pluggable strategy variants)
- RAG pipeline (embedding, FAISS indices, per-turn re-ranking)
- Trigger-check module (rule-based + LLM judge confirmation)
- LLM judge modules (trigger judge + evaluation judge)
- Structured logging and data export utilities
- Shared config, data models (Pydantic), and LLM API wrappers
