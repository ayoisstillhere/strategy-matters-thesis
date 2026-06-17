"""
Build per-party FAISS indices from pre-chunked Wahlprogramme.

Embedding model: all-MiniLM-L6-v2 (384-dim, fast, multilingual-capable)
Index type:      FAISS IndexFlatIP (exact inner product on L2-normalised vectors = cosine)

Also embeds Wahl-O-Mat 2025 theses as separate entries per party,
so RAG can retrieve both programme chunks and structured position data.

Usage:
    pip install sentence-transformers faiss-cpu
    python data/build_faiss_indices.py

Output:
    data/embeddings/{party}.faiss   — FAISS index
    data/embeddings/{party}_meta.jsonl — parallel metadata (chunk_id, text, source)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = DATA_DIR / "chunks"
WAHLOMAT_PATH = DATA_DIR / "wahlomat" / "wahlomat_2025.json"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128

# Party key mapping: chunk filename stems → canonical keys
PARTY_MAP = {
    "afd_2025":     "AfD",
    "cdu_csu_2025": "CDU / CSU",
    "fdp_2025":     "FDP",
    "gruene_2025":  "GRÜNE",
    "linke_2025":   "Die Linke",
    "spd_2025":     "SPD",
}

# Reverse map for Wahl-O-Mat lookup
WAHLOMAT_KEY_TO_STEM = {v: k for k, v in PARTY_MAP.items()}


def load_wahlomat_entries(party_wahlomat_key: str) -> list[dict]:
    """Load Wahl-O-Mat thesis entries for a single party."""
    with open(WAHLOMAT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for thesis in data["theses"]:
        party_data = thesis["parties"].get(party_wahlomat_key)
        if party_data is None:
            continue
        # Combine thesis text + party reasoning into one passage
        text = (
            f"[Wahl-O-Mat] {thesis['title']}: {thesis['text']} "
            f"— Position: {party_data['position']}. "
            f"{party_data['reasoning']}"
        )
        entries.append({
            "source": "wahlomat",
            "thesis_id": thesis["id"],
            "title": thesis["title"],
            "text": text,
        })
    return entries


def build_index_for_party(
    model: SentenceTransformer,
    party_stem: str,
    wahlomat_key: str,
) -> tuple[int, int]:
    """Build and save a FAISS index + metadata for one party.

    Returns (num_programme_chunks, num_wahlomat_entries).
    """
    import faiss

    # --- Load programme chunks ---
    chunk_path = CHUNKS_DIR / f"{party_stem}_chunks.jsonl"
    programme_records = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            programme_records.append({
                "source": "programme",
                "chunk_id": rec["chunk_id"],
                "text": rec["text"],
            })

    # --- Load Wahl-O-Mat entries ---
    wahlomat_records = load_wahlomat_entries(wahlomat_key)

    # --- Combine ---
    all_records = programme_records + wahlomat_records
    texts = [r["text"] for r in all_records]

    # --- Embed ---
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalise → dot product = cosine
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # --- Build FAISS index (exact cosine via inner product) ---
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # --- Save ---
    faiss.write_index(index, str(EMBEDDINGS_DIR / f"{party_stem}.faiss"))

    meta_path = EMBEDDINGS_DIR / f"{party_stem}_meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(programme_records), len(wahlomat_records)


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    total_chunks = 0
    for party_stem, wahlomat_key in PARTY_MAP.items():
        n_prog, n_wahl = build_index_for_party(model, party_stem, wahlomat_key)
        total = n_prog + n_wahl
        total_chunks += total
        print(f"  {party_stem}: {n_prog} programme + {n_wahl} wahlomat = {total} entries")

    print(f"\nDone. {total_chunks} total entries across {len(PARTY_MAP)} parties.")
    print(f"Indices saved to: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
