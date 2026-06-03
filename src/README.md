# /src — Core System Code

This folder contains the main implementation of the AI-moderated multi-agent debate system.

**Will contain:**
- Debate orchestration engine (round-robin turn management, state machine)
- Political agent modules (6 party agents with persona prompts)
- AI moderator module (base + 4 pluggable strategy variants)
- RAG pipeline (embedding, FAISS indices, per-turn re-ranking)
- Trigger-check module (rule-based + LLM judge confirmation)
- LLM judge modules (trigger judge + evaluation judge)
- Structured logging and data export utilities
- Shared config, data models (Pydantic), and LLM API wrappers
