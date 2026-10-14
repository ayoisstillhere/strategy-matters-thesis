# Strategy Matters: AI Moderation in LLM-Simulated Political Debates

Code, data, and analysis for the master's thesis *Strategy Matters: Evaluating AI Moderation
Techniques in LLM-Simulated Political Debates* (TU Dresden, 2026). Six LLM agents, each grounded
in one German party's 2025 election programme and Wahl-O-Mat positions, debate four policy topics
under four moderation strategies and four baselines; an LLM judge scores every turn on seven
discourse-quality dimensions.

> **All debate transcripts in this repository are AI-generated simulations, not actual party
> statements.**

## Record of the experiment

Every model used for the experiment has since been removed from Groq, so the stored transcripts
and scores in this repository are the only record of the experiment. They are not regenerable.

| Role | Exact model ID (Groq API) | Used for | Dates (2026) |
|---|---|---|---|
| Debate agents | `llama-3.1-8b-instant` | 160 main runs, 5 ablation runs | 6–7 July |
| Judge (in-debate and trigger check) and moderator | `meta-llama/llama-4-scout-17b-16e-instruct` | 160 main runs, 5 ablation runs | 6–7 July |
| Post-hoc judge (all analyses) | `meta-llama/llama-4-scout-17b-16e-instruct` | condition-blind re-scoring of all 9,900 turns | 8 July |
| Feasibility-pilot judge | `llama-3.3-70b-versatile` | feasibility and prompt pilots | 3–29 June |
| Second judge (validity check) | `qwen/qwen3.8-27b` | 400-turn cross-judge re-scoring | 13 October |
| Interactive demo (agents and judge) | `qwen/qwen3.8-27b` | demo only, not used for results | from September |

Other models (run locally):

- Embeddings for RAG: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (input truncated
  at 128 tokens; see `data/check_embedding_truncation.py`).
- NLI attribution check: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.

Groq removed Llama 4 Scout with a listed shutdown date of 17 July 2026; the removal was noticed on
24 July 2026. Generation settings are listed in the thesis's hyperparameter table (agents:
temperature 0.7; judges: 0.0; moderator: 0.3).

The human validation study (5 annotators, 50 exchanges) ran on Prolific on 26–27 July 2026.
Outputs in `experiments/outputs/human_validation/` identify annotators only by pseudonyms
(A1–A5) or random IDs.

## Repository layout

- `src/`: debate engine, agents, moderator, judge, RAG pipeline, prompts.
- `data/`: party programmes, Wahl-O-Mat data, chunks, and FAISS indices.
- `runs/experiment/`: the 160 main debate logs (JSON); `runs/ablation_turn_order/`: the 5
  reversed-order runs; `runs/posthoc_scores/posthoc_scores.csv`: post-hoc scores.
- `experiments/`: analysis scripts (`comparative_analysis.py`, `generate_figures.py`,
  `generate_diagrams.py`, `human_validation_analysis.py`, `judge_validity/`) and their outputs.
- `demo/`: interactive demo (React frontend, FastAPI backend), deployed with the `Dockerfile`.

## Reproducing the analyses

Analyses need no API access; they work from the stored logs and scores.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-deploy.txt pandas scipy statsmodels seaborn
.venv\Scripts\python.exe experiments\comparative_analysis.py
.venv\Scripts\python.exe experiments\generate_figures.py
.venv\Scripts\python.exe experiments\generate_diagrams.py
```

Re-running debates or the demo needs a `GROQ_API_KEY` in `.env` (not committed) and a model that
is still available.

## Licence

Code: MIT. Generated transcripts, scores, and analysis outputs: CC BY 4.0. The party programmes
and the Wahl-O-Mat data remain under their owners' terms. See [LICENSE](LICENSE).
