"""Shared loaders for the judge-validity checks."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from comparative_analysis import DIMENSIONS, load_posthoc  # noqa: E402,F401

OUT_DIR = ROOT / "experiments" / "outputs" / "judge_validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_turns() -> pd.DataFrame:
    """Post-hoc scores for the 160 main runs, joined with turn text and logged RAG prefixes."""
    df = load_posthoc()
    df = df[~df["is_ablation"]].copy()
    texts, rag = {}, {}
    for rf in df["run_file"].unique():
        data = json.loads((ROOT / rf).read_text(encoding="utf-8"))
        for t in data["turns"]:
            texts[t["turn_id"]] = t["text"]
            rag[t["turn_id"]] = t.get("rag_passages_used", [])
    df["text"] = df["turn_id"].map(texts)
    df["rag_prefixes"] = df["turn_id"].map(rag)
    return df.reset_index(drop=True)
