"""Check how much text the bi-encoder sees after its max_seq_length truncation.

Reports token lengths of indexed passages and re-ranking queries under the
embedding model's own tokenizer, and whether the logged top-3 passages vary
across turns within a debate (they cannot if the query is truncated to the
framing prompt).

Usage: python data/check_embedding_truncation.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_config import FRAMING_PROMPTS  # noqa: E402
from src.rag_pipeline import BI_ENCODER_MODEL  # noqa: E402

model = SentenceTransformer(BI_ENCODER_MODEL)
tok = model.tokenizer
max_len = model.max_seq_length
print(f"max_seq_length = {max_len}")


def n_tokens(text: str) -> int:
    return len(tok(text, add_special_tokens=True)["input_ids"])


# --- Indexed passages ---
prog, wom, prog_chars, kept_chars = [], [], [], []
for meta in sorted((ROOT / "data" / "embeddings").glob("*_meta.jsonl")):
    for line in meta.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        n = n_tokens(rec["text"])
        if rec.get("source") == "wahlomat":
            wom.append(n)
        else:
            prog.append(n)
            ids = tok(rec["text"], add_special_tokens=False,
                      truncation=True, max_length=max_len - 2)["input_ids"]
            prog_chars.append(len(rec["text"]))
            kept_chars.append(len(tok.decode(ids)))


def summary(name, arr):
    a = np.array(arr)
    print(f"{name}: n={len(a)}, median={np.median(a):.0f}, mean={a.mean():.0f}, "
          f"min={a.min()}, max={a.max()}, >{max_len}: {np.mean(a > max_len):.1%}")


summary("Programme chunks (model tokens)", prog)
summary("Wahl-O-Mat entries (model tokens)", wom)
pc, kc = np.array(prog_chars), np.array(kept_chars)
print(f"Programme chunks: mean {pc.mean():.0f} chars, embedded part covers "
      f"{np.mean(np.minimum(kc / pc, 1)):.1%} of characters on average")

# --- Queries ---
for topic, fp in FRAMING_PROMPTS.items():
    print(f"Framing prompt {topic}: {n_tokens(fp)} model tokens")

# --- Logged passages: do they vary within a debate? ---
run_files = sorted((ROOT / "runs" / "experiment").glob("*/*/run_*.json"))
by_topic = defaultdict(lambda: [0, 0])
for rf in run_files:
    data = json.loads(rf.read_text(encoding="utf-8"))
    sets = defaultdict(set)
    for t in data["turns"]:
        sets[t["agent_name"]].add(tuple(t.get("rag_passages_used", [])))
    for s in sets.values():
        by_topic[rf.parent.name][0] += 1
        by_topic[rf.parent.name][1] += len(s) > 1
for topic, (total, varies) in sorted(by_topic.items()):
    print(f"{topic}: debate-party pairs whose top-3 passages changed across turns: {varies}/{total}")
