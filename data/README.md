# /data — Party Programmes & Wahl-O-Mat Data

This folder contains all grounding documents used to anchor political agents to real party positions.

**Will contain:**
- `wahlprogramme/` — Full-text Bundestagswahlprogramme 2025 for all 6 parties (CDU/CSU, SPD, Grüne, FDP, Die Linke, AfD), both raw PDFs and extracted clean text
- `wahlomat/` — Structured Wahl-O-Mat 2025 position data (38 policy theses with per-party positions and reasoning)
- `chunks/` — Pre-chunked passages (~256 tokens, 64-token overlap) per party, ready for embedding
- `embeddings/` — FAISS indices per party (generated from chunks)
- Any curated factual knowledge base for constraining numerical claims
