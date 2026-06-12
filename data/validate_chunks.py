# data/validate_chunks.py
from sentence_transformers import SentenceTransformer
import numpy as np, json
from pathlib import Path

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

TOPICS = [
    "Mindestlohn Erhöhung auf 15 Euro",
    "Rentenpolitik Rentenalter",
    "Vermögenssteuer Umverteilung Sozialpolitik",
    "Migration Asylpolitik Integration"
]

DATA_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = DATA_DIR / "chunks"

for party_file in sorted(CHUNKS_DIR.glob("*.jsonl")):
    chunks = [json.loads(l) for l in party_file.read_text(encoding="utf-8").splitlines()]
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    print(f"\n=== {party_file.stem} ===")
    for topic in TOPICS:
        q_emb = model.encode([topic])
        sims = np.dot(embeddings, q_emb.T).flatten()
        top3 = np.argsort(sims)[-3:][::-1]
        print(f"  {topic[:30]}... → top chunk sim={sims[top3[0]]:.3f}")
        # Print first 80 chars of top chunk for manual inspection
        print(f"    \"{texts[top3[0]][:80]}...\"")