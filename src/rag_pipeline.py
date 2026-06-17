"""
RAG Pipeline for Agent Grounding
=================================
Retrieves and re-ranks grounding passages from party programmes and
Wahl-O-Mat positions to provide per-turn context for political agents.

Pipeline (per agent turn):
  1. At debate start: initial retrieval of top-K passages per party
     using the framing prompt as query (cached for the whole debate).
  2. Per turn: re-rank the cached pool using last 2 turns + framing
     as the query, select top-N passages.
  3. Format selected passages as a grounding block injected into the
     agent's prompt.

Production method: cosine re-ranking with paraphrase-multilingual-MiniLM-L12-v2.

Decision rationale (pilot comparison, 4 topics × 6 parties):
  - Multilingual bi-encoder achieves 0.60–0.81 cosine similarity on
    German political text, correctly surfacing topic-relevant passages.
  - English cross-encoder (ms-marco-MiniLM-L-6-v2) produced uniformly
    negative scores (-3.5 to -7.0) with off-topic retrievals.
  - Cross-encoder option retained for potential future multilingual CE
    but not used in production.

See also:
  - data/build_faiss_indices.py — index builder
  - pilots/rag/rag_reranking_comparison.py — comparison test results
  - expose.tex §4.3 — Persona Reinforcement via RAG
  - src/prompts/agent_prompts.py — AGENT_RULES includes RAG instruction
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_EMBEDDINGS_DIR = DEFAULT_DATA_DIR / "embeddings"

BI_ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 15
RERANK_TOP_N = 3

PARTY_STEMS = {
    "CDU/CSU":                  "cdu_csu_2025",
    "SPD":                      "spd_2025",
    "Bündnis 90/Die Grünen":    "gruene_2025",
    "FDP":                      "fdp_2025",
    "Die Linke":                "linke_2025",
    "AfD":                      "afd_2025",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPassage:
    """A single retrieved and ranked passage."""
    text: str
    score: float
    source: str          # "programme" or "wahlomat"
    rank: int            # 1-indexed rank after re-ranking
    chunk_id: Optional[int] = None
    thesis_id: Optional[int] = None


@dataclass
class PartyPool:
    """Cached retrieval pool for one party within a debate run."""
    party: str
    indices: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Per-debate RAG pipeline managing retrieval and re-ranking.

    Lifecycle:
      1. Instantiate once per debate run.
      2. Call init_pools(framing_prompt) at debate start.
      3. Call get_grounding(party, recent_turns) each agent turn.
      4. Discard after the debate run.
    """

    def __init__(
        self,
        rerank_method: str = "cosine",
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_n: int = RERANK_TOP_N,
        embeddings_dir: Optional[Path] = None,
    ):
        """
        Args:
            rerank_method: "cosine" or "cross-encoder".
            initial_top_k: Size of the initial retrieval pool per party.
            rerank_top_n: Number of passages to select after re-ranking.
            embeddings_dir: Path to FAISS indices directory.
        """
        if rerank_method not in ("cosine", "cross-encoder"):
            raise ValueError(f"Unknown rerank_method: {rerank_method}")

        self.rerank_method = rerank_method
        self.initial_top_k = initial_top_k
        self.rerank_top_n = rerank_top_n
        self.embeddings_dir = embeddings_dir or DEFAULT_EMBEDDINGS_DIR

        # Lazy-loaded models
        self._bi_encoder = None
        self._cross_encoder = None

        # Per-party cached pools (populated by init_pools)
        self._pools: dict[str, PartyPool] = {}

        # Per-party FAISS indices + metadata (loaded on demand)
        self._indices: dict[str, tuple] = {}

    # ── Model loading ──

    def _get_bi_encoder(self):
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        return self._bi_encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    def _load_party_index(self, party_stem: str):
        """Load FAISS index and metadata for a party."""
        if party_stem in self._indices:
            return self._indices[party_stem]

        import faiss

        index_path = self.embeddings_dir / f"{party_stem}.faiss"
        meta_path = self.embeddings_dir / f"{party_stem}_meta.jsonl"

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

        self._indices[party_stem] = (index, metadata)
        return index, metadata

    # ── Initial retrieval ──

    def init_pools(self, framing_prompt: str, parties: Optional[list[str]] = None):
        """Retrieve initial top-K pools for all parties at debate start.

        Args:
            framing_prompt: The neutral framing prompt for the debate topic.
            parties: List of party names to initialise. Defaults to all 6.
        """
        if parties is None:
            parties = list(PARTY_STEMS.keys())

        bi_encoder = self._get_bi_encoder()
        query_emb = bi_encoder.encode(
            [framing_prompt], normalize_embeddings=True
        ).astype(np.float32)

        for party in parties:
            stem = PARTY_STEMS.get(party)
            if stem is None:
                raise ValueError(f"Unknown party: {party}")

            index, metadata = self._load_party_index(stem)
            k = min(self.initial_top_k, index.ntotal)
            scores, indices = index.search(query_emb, k)

            idx_list = indices[0].tolist()
            pool = PartyPool(
                party=party,
                indices=idx_list,
                texts=[metadata[i]["text"] for i in idx_list],
                metadata=[metadata[i] for i in idx_list],
            )
            self._pools[party] = pool

    # ── Per-turn re-ranking ──

    def _build_rerank_query(
        self, framing_prompt: str, recent_turns: list[str]
    ) -> str:
        """Build re-ranking query from framing + last 2 turns."""
        parts = [framing_prompt]
        for t in recent_turns[-2:]:
            parts.append(t)
        return " ".join(parts)

    def _cosine_rerank(
        self, query: str, passages: list[str]
    ) -> list[tuple[int, float]]:
        """Re-rank by cosine similarity. Returns [(index, score), ...]."""
        bi_encoder = self._get_bi_encoder()
        query_emb = bi_encoder.encode([query], normalize_embeddings=True)[0]
        passage_embs = bi_encoder.encode(passages, normalize_embeddings=True)
        scores = np.dot(passage_embs, query_emb)
        ranked = np.argsort(scores)[::-1][:self.rerank_top_n]
        return [(int(i), float(scores[i])) for i in ranked]

    def _crossencoder_rerank(
        self, query: str, passages: list[str]
    ) -> list[tuple[int, float]]:
        """Re-rank with cross-encoder. Returns [(index, score), ...]."""
        ce = self._get_cross_encoder()
        pairs = [[query, p] for p in passages]
        scores = ce.predict(pairs)
        ranked = np.argsort(scores)[::-1][:self.rerank_top_n]
        return [(int(i), float(scores[i])) for i in ranked]

    def get_grounding(
        self,
        party: str,
        framing_prompt: str,
        recent_turns: list[str],
    ) -> list[RetrievedPassage]:
        """Retrieve and re-rank grounding passages for one agent turn.

        Args:
            party: Party name (e.g. "CDU/CSU").
            framing_prompt: The debate framing prompt.
            recent_turns: List of recent turn texts (last 2 used for query).

        Returns:
            List of top-N RetrievedPassage objects, ranked by relevance.
        """
        pool = self._pools.get(party)
        if pool is None:
            raise RuntimeError(
                f"Pool not initialised for {party}. Call init_pools() first."
            )

        query = self._build_rerank_query(framing_prompt, recent_turns)

        if self.rerank_method == "cosine":
            ranked = self._cosine_rerank(query, pool.texts)
        else:
            ranked = self._crossencoder_rerank(query, pool.texts)

        passages = []
        for rank_pos, (pool_idx, score) in enumerate(ranked, start=1):
            meta = pool.metadata[pool_idx]
            passages.append(RetrievedPassage(
                text=pool.texts[pool_idx],
                score=score,
                source=meta.get("source", "programme"),
                rank=rank_pos,
                chunk_id=meta.get("chunk_id"),
                thesis_id=meta.get("thesis_id"),
            ))

        return passages

    # ── Formatting ──

    @staticmethod
    def format_grounding_block(passages: list[RetrievedPassage]) -> str:
        """Format retrieved passages into a text block for prompt injection.

        This block is appended to the agent's user prompt before each turn.
        """
        if not passages:
            return ""

        lines = ["[GROUNDING — Your party's documented positions relevant to this discussion:]"]
        for p in passages:
            source_tag = "Wahl-O-Mat" if p.source == "wahlomat" else "Programme"
            lines.append(f"  [{source_tag}] {p.text}")
        lines.append("[END GROUNDING — Base your arguments on these positions.]")
        return "\n".join(lines)
