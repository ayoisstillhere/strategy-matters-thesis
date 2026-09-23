# Copilot Instructions: strategy-matters-thesis

AI-moderated multi-agent political debate simulation for the thesis *Strategy Matters*. Six German party agents (CDU/CSU, SPD, Grüne, FDP, Linke, AfD) debate for 10 rounds; a moderator intervenes per condition; an LLM judge scores 7 discourse-quality dimensions. Main experiment: 4 topics × 8 conditions × 5 runs = 160 debates. The parent folder's `AGENTS.md` has the full handbook.

## Environment

- Run Python with `.venv\Scripts\python.exe <script>` from this repo root. Never install packages globally.
- API keys in `.env` (`GROQ_API_KEY`, `OPENAI_API_KEY`). Never print or commit them.
- Keep the SSL fix `httpx.Client(verify=False)` in `src/llm_client.py` (TU Dresden/work network proxy).
- In PowerShell, run multi-line Python as a `.py` file instead of `python -c`.

## Key entry points

- `run_experiment.py` — main runner, resumable via checkpoint (`--condition`, `--topic`, `--max-runs`, `--reverse-order`).
- `run_posthoc_scoring.py` — blinded post-hoc judge scoring of all turns.
- `src/llm_client.py` (Groq client, JSON repair), `src/judge.py`, `src/moderator.py`, `src/trigger_check.py`, `src/rag_pipeline.py`, `src/debate_engine.py`.
- `data/extract_text.py` → `data/chunk_programmes.py` → `data/build_faiss_indices.py` — manifesto data pipeline.
- `experiments/comparative_analysis.py`, `experiments/generate_figures.py`, `experiments/human_validation_analysis.py`.

## Models

- Agent and judge: `qwen/qwen3.8-27b` via Groq. Before changing a model, run an agreement pilot and report exact and ±1 agreement.
- Avoid models that are non-deterministic at `temperature=0` or emit thinking blocks.
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`. Do not use English-only models (`all-MiniLM-L6-v2`, `ms-marco` cross-encoders) for German text.
- Construct the Groq client with a keyword argument: `Groq(api_key=...)`.

## Established decisions

- Judge JSON: use `response_format={"type": "json_object"}`, `_repair_truncated_json()`, and a parse-retry loop. Only fall back to a default score after retries fail, and log it at ERROR. Silent default scores must never corrupt experiment data.
- Keep `EvaluationJudge.max_tokens` generous (currently 1200).
- Anchor file paths in scripts to `Path(__file__).resolve().parent` so they work from any working directory.
- PDF extraction: `dedupe_chars(tolerance=1)`, strip `(cid:NN)` codes, skip cover pages.
- Strategy C (fact-reminder trigger) firing rarely is an accepted finding (RAG pre-empts it), not a bug.
- Prefer minimal upstream fixes; add regression tests where feasible.
