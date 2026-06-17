# /data — Party Programmes & Wahl-O-Mat Data

This folder contains all grounding documents used to anchor political agents to real party positions.

**Current contents:**
- `wahlprogramme/` — Full-text Bundestagswahlprogramme 2025 for all 6 parties (raw PDFs + extracted text)
- `wahlomat/` — Structured Wahl-O-Mat 2025 position data (38 theses × 6 parties with reasoning)
- `chunks/` — Pre-chunked passages (~256 tokens, 64-token overlap) per party (~2,400 total)
- `embeddings/` — Per-party FAISS indices (paraphrase-multilingual-MiniLM-L12-v2, 384-dim)
- `build_faiss_indices.py` — Builds FAISS indices from chunks + Wahl-O-Mat entries
- `chunk_programmes.py` — Chunks raw text into JSONL passages
- `extract_text.py` — Extracts text from PDF programmes
- `validate_chunks.py` — Validates chunk quality

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (selected over English-only model after pilot comparison)
