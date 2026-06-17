"""
RAG Re-ranking Comparison: Cosine vs Cross-Encoder
====================================================
Tests two re-ranking strategies on the debate grounding task:

  (A) Cosine re-ranking — same bi-encoder (all-MiniLM-L6-v2), embed the
      re-ranking query, compute cosine similarity against cached pool,
      select top-3. Fast (vector dot product only).

  (B) Cross-encoder re-ranking — cross-encoder/ms-marco-MiniLM-L-6-v2,
      score each (query, passage) pair, select top-3. More accurate but
      slower (must forward-pass every pair).

Both methods operate on the same initial top-K pool retrieved from the
per-party FAISS index built by data/build_faiss_indices.py.

Metrics collected per (topic, party, method):
  - Latency (ms)
  - Top-3 passage texts (for qualitative inspection)
  - Top-3 scores
  - Overlap between methods (Jaccard of top-3 indices)

Usage:
    1. First run:  python data/build_faiss_indices.py
    2. Then run:   python pilots/rag/rag_reranking_comparison.py

    pip install sentence-transformers faiss-cpu

Output:
    pilots/rag/outputs/reranking_comparison_{timestamp}.json
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BI_ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 15   # Initial retrieval pool size
RERANK_TOP_N = 3     # Final selection after re-ranking
NUM_RERUNS = 3        # Reruns for latency averaging

PARTIES = [
    "afd_2025",
    "cdu_csu_2025",
    "fdp_2025",
    "gruene_2025",
    "linke_2025",
    "spd_2025",
]

# Debate topics with framing queries + simulated turn context
TEST_QUERIES = [
    {
        "topic": "Mindestlohn",
        "framing": "The current federal minimum wage in Germany is €12.82/hour. "
                   "Discuss whether it should be raised to €15/hour.",
        "simulated_turns": [
            "The minimum wage must be raised to ensure social justice for workers.",
            "A jump to €15 risks job losses in eastern Länder where productivity is lower.",
        ],
    },
    {
        "topic": "Rentenpolitik",
        "framing": "Germany's statutory pension system faces demographic pressure. "
                   "Should the retirement age be raised, benefits cut, or contributions increased?",
        "simulated_turns": [
            "Raising the retirement age is unavoidable given demographic trends.",
            "We need a solidarity-based pension that guarantees a dignified old age.",
        ],
    },
    {
        "topic": "Sozialpolitik",
        "framing": "Wealth inequality in Germany has risen over the past decade. "
                   "Should the government implement a wealth tax or expand redistribution?",
        "simulated_turns": [
            "A wealth tax would fund essential public services and reduce inequality.",
            "Wealth taxes drive capital abroad and harm investment in Germany.",
        ],
    },
    {
        "topic": "Migrationspolitik",
        "framing": "Germany received over 300,000 asylum applications in 2023. "
                   "How should the country reform its migration and asylum policy?",
        "simulated_turns": [
            "We need a humanitarian migration policy that respects human rights.",
            "Illegal migration must be stopped with border controls and faster deportations.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_party_index(party_stem: str):
    """Load FAISS index and metadata for a party."""
    import faiss

    index_path = EMBEDDINGS_DIR / f"{party_stem}.faiss"
    meta_path = EMBEDDINGS_DIR / f"{party_stem}_meta.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Run: python data/build_faiss_indices.py"
        )

    index = faiss.read_index(str(index_path))
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return index, metadata


def build_rerank_query(framing: str, turns: list[str]) -> str:
    """Build re-ranking query from framing prompt + last 2 turns."""
    parts = [framing]
    for t in turns[-2:]:
        parts.append(t)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Re-ranking methods
# ---------------------------------------------------------------------------

def cosine_rerank(
    bi_encoder: SentenceTransformer,
    query: str,
    passages: list[str],
    top_n: int,
) -> tuple[list[int], list[float], float]:
    """Re-rank passages by cosine similarity using the bi-encoder.

    Returns (top_n_indices, top_n_scores, latency_ms).
    """
    start = time.perf_counter()

    query_emb = bi_encoder.encode(
        [query], normalize_embeddings=True
    )[0]
    passage_embs = bi_encoder.encode(
        passages, normalize_embeddings=True
    )

    # Cosine similarity (dot product on normalised vectors)
    scores = np.dot(passage_embs, query_emb)

    # Top-N
    top_indices = np.argsort(scores)[::-1][:top_n].tolist()
    top_scores = [float(scores[i]) for i in top_indices]

    latency = (time.perf_counter() - start) * 1000
    return top_indices, top_scores, latency


def crossencoder_rerank(
    cross_encoder: CrossEncoder,
    query: str,
    passages: list[str],
    top_n: int,
) -> tuple[list[int], list[float], float]:
    """Re-rank passages using a cross-encoder model.

    Returns (top_n_indices, top_n_scores, latency_ms).
    """
    start = time.perf_counter()

    pairs = [[query, p] for p in passages]
    scores = cross_encoder.predict(pairs)

    # Top-N
    top_indices = np.argsort(scores)[::-1][:top_n].tolist()
    top_scores = [float(scores[i]) for i in top_indices]

    latency = (time.perf_counter() - start) * 1000
    return top_indices, top_scores, latency


# ---------------------------------------------------------------------------
# Initial retrieval
# ---------------------------------------------------------------------------

def initial_retrieve(
    bi_encoder: SentenceTransformer,
    index,
    metadata: list[dict],
    query: str,
    top_k: int,
) -> tuple[list[int], list[str], list[dict]]:
    """Retrieve top-K passages from FAISS index.

    Returns (indices, texts, metadata_entries).
    """
    query_emb = bi_encoder.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_emb, min(top_k, index.ntotal))
    indices = indices[0].tolist()

    texts = [metadata[i]["text"] for i in indices]
    metas = [metadata[i] for i in indices]
    return indices, texts, metas


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison():
    print("=" * 70)
    print("RAG Re-ranking Comparison: Cosine vs Cross-Encoder")
    print("=" * 70)

    # Load models
    print(f"\nLoading bi-encoder: {BI_ENCODER_MODEL}")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Load all party indices
    print("\nLoading FAISS indices...")
    party_data = {}
    for party in PARTIES:
        index, metadata = load_party_index(party)
        party_data[party] = (index, metadata)
        print(f"  {party}: {index.ntotal} entries")

    results = []
    all_latencies_cosine = []
    all_latencies_crossenc = []
    overlap_scores = []

    for query_cfg in TEST_QUERIES:
        topic = query_cfg["topic"]
        rerank_query = build_rerank_query(
            query_cfg["framing"], query_cfg["simulated_turns"]
        )

        print(f"\n{'─' * 70}")
        print(f"TOPIC: {topic}")
        print(f"Re-rank query: {rerank_query[:100]}...")
        print(f"{'─' * 70}")

        for party in PARTIES:
            index, metadata = party_data[party]

            # Initial retrieval (using framing as query)
            ret_indices, ret_texts, ret_metas = initial_retrieve(
                bi_encoder, index, metadata,
                query_cfg["framing"], INITIAL_TOP_K
            )

            # --- Cosine re-ranking (averaged over reruns) ---
            cosine_latencies = []
            for _ in range(NUM_RERUNS):
                cos_indices, cos_scores, cos_lat = cosine_rerank(
                    bi_encoder, rerank_query, ret_texts, RERANK_TOP_N
                )
                cosine_latencies.append(cos_lat)
            cos_mean_lat = sum(cosine_latencies) / len(cosine_latencies)

            # --- Cross-encoder re-ranking (averaged over reruns) ---
            ce_latencies = []
            for _ in range(NUM_RERUNS):
                ce_indices, ce_scores, ce_lat = crossencoder_rerank(
                    cross_encoder, rerank_query, ret_texts, RERANK_TOP_N
                )
                ce_latencies.append(ce_lat)
            ce_mean_lat = sum(ce_latencies) / len(ce_latencies)

            # --- Overlap (Jaccard of top-3 indices) ---
            cos_set = set(cos_indices)
            ce_set = set(ce_indices)
            if cos_set or ce_set:
                jaccard = len(cos_set & ce_set) / len(cos_set | ce_set)
            else:
                jaccard = 1.0

            all_latencies_cosine.append(cos_mean_lat)
            all_latencies_crossenc.append(ce_mean_lat)
            overlap_scores.append(jaccard)

            # --- Report ---
            print(f"\n  [{party}]")
            print(f"    Cosine:       {cos_mean_lat:6.1f}ms  scores={[f'{s:.3f}' for s in cos_scores]}")
            print(f"    Cross-enc:    {ce_mean_lat:6.1f}ms  scores={[f'{s:.3f}' for s in ce_scores]}")
            print(f"    Top-3 overlap (Jaccard): {jaccard:.2f}")

            if cos_indices != ce_indices:
                print(f"    ⚠ Methods disagree on ranking")
                # Show where they differ
                cos_only = cos_set - ce_set
                ce_only = ce_set - cos_set
                if cos_only:
                    for idx in cos_only:
                        print(f"      Cosine-only #{idx}: {ret_texts[idx][:80]}...")
                if ce_only:
                    for idx in ce_only:
                        print(f"      CrossEnc-only #{idx}: {ret_texts[idx][:80]}...")
            else:
                print(f"    ✓ Methods agree on top-3")

            entry = {
                "topic": topic,
                "party": party,
                "initial_pool_size": len(ret_texts),
                "cosine": {
                    "top_indices": cos_indices,
                    "top_scores": cos_scores,
                    "top_texts": [ret_texts[i][:200] for i in cos_indices],
                    "mean_latency_ms": round(cos_mean_lat, 2),
                },
                "cross_encoder": {
                    "top_indices": ce_indices,
                    "top_scores": ce_scores,
                    "top_texts": [ret_texts[i][:200] for i in ce_indices],
                    "mean_latency_ms": round(ce_mean_lat, 2),
                },
                "jaccard_overlap": round(jaccard, 3),
            }
            results.append(entry)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    mean_cos_lat = sum(all_latencies_cosine) / len(all_latencies_cosine)
    mean_ce_lat = sum(all_latencies_crossenc) / len(all_latencies_crossenc)
    mean_overlap = sum(overlap_scores) / len(overlap_scores)
    speedup = mean_ce_lat / mean_cos_lat if mean_cos_lat > 0 else float("inf")

    print(f"\n  Mean latency (cosine):        {mean_cos_lat:7.1f} ms")
    print(f"  Mean latency (cross-encoder): {mean_ce_lat:7.1f} ms")
    print(f"  Speedup (cosine vs CE):       {speedup:7.1f}x")
    print(f"  Mean top-3 Jaccard overlap:   {mean_overlap:7.3f}")
    print(f"  Total comparisons:            {len(results)}")

    agree_count = sum(1 for r in results if r["jaccard_overlap"] == 1.0)
    print(f"  Exact agreement (Jaccard=1):  {agree_count}/{len(results)}")

    # Per-topic summary
    topics = list(dict.fromkeys(r["topic"] for r in results))
    print(f"\n  Per-topic overlap:")
    for topic in topics:
        topic_results = [r for r in results if r["topic"] == topic]
        topic_overlap = sum(r["jaccard_overlap"] for r in topic_results) / len(topic_results)
        topic_cos_lat = sum(r["cosine"]["mean_latency_ms"] for r in topic_results) / len(topic_results)
        topic_ce_lat = sum(r["cross_encoder"]["mean_latency_ms"] for r in topic_results) / len(topic_results)
        print(f"    {topic:<20} overlap={topic_overlap:.3f}  cos={topic_cos_lat:.0f}ms  ce={topic_ce_lat:.0f}ms")

    # --- Decision guidance ---
    print(f"\n{'─' * 70}")
    print("DECISION GUIDANCE")
    print(f"{'─' * 70}")
    print()
    print("Choose COSINE re-ranking if:")
    print("  - Overlap is high (>0.8): methods agree, CE adds no quality")
    print("  - Speedup is large (>5x): latency savings matter for 160 runs")
    print("  - Per-turn re-ranking budget must stay minimal")
    print()
    print("Choose CROSS-ENCODER re-ranking if:")
    print("  - Overlap is low (<0.6): CE finds different, potentially better passages")
    print("  - Latency is acceptable (<500ms per re-rank)")
    print("  - Qualitative inspection shows CE passages are more relevant")
    print()
    print("INSPECT the top-3 texts in the output JSON to compare quality.")

    # --- Save ---
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "bi_encoder": BI_ENCODER_MODEL,
            "cross_encoder": CROSS_ENCODER_MODEL,
            "initial_top_k": INITIAL_TOP_K,
            "rerank_top_n": RERANK_TOP_N,
            "num_reruns": NUM_RERUNS,
        },
        "summary": {
            "mean_cosine_latency_ms": round(mean_cos_lat, 2),
            "mean_crossencoder_latency_ms": round(mean_ce_lat, 2),
            "speedup_factor": round(speedup, 2),
            "mean_jaccard_overlap": round(mean_overlap, 3),
            "exact_agreement_count": agree_count,
            "total_comparisons": len(results),
        },
        "results": results,
    }

    out_path = OUTPUT_DIR / f"reranking_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")
    return output


if __name__ == "__main__":
    run_comparison()
