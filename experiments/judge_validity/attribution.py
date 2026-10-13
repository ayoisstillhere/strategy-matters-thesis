"""Automatic attribution check for document-grounding.

The judge scored document-grounding without seeing any party documents. This script
checks, with a multilingual NLI model, whether the sentences of a turn are entailed by
(a) the three passages that were actually retrieved for that turn and injected into the
agent's prompt, and (b) those passages plus the passage of the party's full document
index (programme + Wahl-O-Mat) most similar to each sentence.

The logged RAG passages are truncated to 100 characters; full texts are recovered by
matching the prefix against the FAISS metadata files.

Usage:
    python experiments/judge_validity/attribution.py --per-run 4    # NLI pass (cached, resumable)
    python experiments/judge_validity/attribution.py --analyse       # analysis only

Outputs (experiments/outputs/judge_validity/):
    attribution_turns.csv, attribution.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT_DIR, ROOT, load_turns  # noqa: E402
from comparative_analysis import compare_groups, holm  # noqa: E402

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
BI_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIR = ROOT / "data" / "embeddings"
PARTY_STEMS = {
    "CDU/CSU": "cdu_csu_2025", "SPD": "spd_2025", "Bündnis 90/Die Grünen": "gruene_2025",
    "FDP": "fdp_2025", "Die Linke": "linke_2025", "AfD": "afd_2025",
}
ENTAIL_THRESHOLD = 0.5
PROGRAMME_TOP_K = 1
MIN_WORDS = 5
CACHE = OUT_DIR / "attribution_turns.csv"
HUMAN_TIERS = OUT_DIR / "human_tiers.json"
EXPORT = ROOT.parent / "annotation-app" / "exports" / "annotation-export-2026-07-28.json"
STRATEGIES = {"strategy_a": "De-escalation", "strategy_b": "Reframing",
              "strategy_c": "Fact-reminder", "strategy_d": "Common-ground"}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„])", text.replace("\n", " "))
    return [s.strip() for s in parts if len(s.split()) >= MIN_WORDS]


def load_meta() -> dict:
    meta = {}
    for party, stem in PARTY_STEMS.items():
        rows = [json.loads(l) for l in (EMB_DIR / f"{stem}_meta.jsonl").read_text(encoding="utf-8").splitlines() if l]
        meta[party] = rows
    return meta


def select_turns(df: pd.DataFrame, per_run: int, seed: int = 2026) -> pd.DataFrame:
    """Up to `per_run` turns per debate (round 1 opening turns excluded), plus all annotated turns."""
    pool = df[~df["is_opening_turn"]]
    picked = pool.groupby("run_id", group_keys=False).apply(
        lambda g: g.sample(min(per_run, len(g)), random_state=seed))
    annotated = {ex["turn_id"] for ex in json.loads(EXPORT.read_text(encoding="utf-8"))["exchanges"]}
    extra = df[df["turn_id"].isin(annotated) & ~df["turn_id"].isin(picked["turn_id"])]
    # Annotated turns first, then a shuffled order so that a partial pass is still balanced.
    picked = picked.sample(frac=1, random_state=seed)
    return pd.concat([extra, picked])


class NLI:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        self.entail_idx = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}["entailment"]

    def entail(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """Entailment probability for (premise, hypothesis) pairs."""
        out = np.empty(len(pairs))
        order = np.argsort([len(p) + len(h) for p, h in pairs])
        with self.torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                idx = order[i:i + batch_size]
                enc = self.tok([pairs[j][0] for j in idx], [pairs[j][1] for j in idx],
                               truncation="only_first", max_length=384, padding=True, return_tensors="pt")
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.entail_idx]
                out[idx] = probs.numpy()
        return out


def run_nli(per_run: int):
    import faiss
    from sentence_transformers import SentenceTransformer

    df = load_turns()
    sample = select_turns(df, per_run)
    done = set(pd.read_csv(CACHE)["turn_id"]) if CACHE.exists() else set()
    todo = sample[~sample["turn_id"].isin(done)]
    print(f"{len(sample)} turns selected, {len(done)} cached, {len(todo)} to score")
    if todo.empty:
        return

    meta = load_meta()
    prefix_maps = {p: {r["text"][:100]: r["text"] for r in rows} for p, rows in meta.items()}
    indices = {p: faiss.read_index(str(EMB_DIR / f"{s}.faiss")) for p, s in PARTY_STEMS.items()}
    encoder = SentenceTransformer(BI_ENCODER)
    nli = NLI()

    t0, batch_rows = time.time(), []
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        party = row["agent_name"]
        passages = [prefix_maps[party].get(p) for p in row["rag_prefixes"]]
        recovered = [p for p in passages if p]
        sents = split_sentences(row["text"])
        rec = {"turn_id": row["turn_id"], "n_sentences": len(sents), "n_passages_logged": len(passages),
               "n_passages_recovered": len(recovered)}
        if sents:
            emb = encoder.encode(sents, normalize_embeddings=True).astype(np.float32)
            _, nn = indices[party].search(emb, PROGRAMME_TOP_K)
            pairs, owner, kind = [], [], []
            for si, s in enumerate(sents):
                for p in recovered:
                    pairs.append((p, s)); owner.append(si); kind.append("retrieved")
                for ci in nn[si]:
                    if meta[party][ci]["text"] not in recovered:
                        pairs.append((meta[party][ci]["text"], s)); owner.append(si); kind.append("programme")
            probs = nli.entail(pairs)
            owner, kind = np.array(owner), np.array(kind)
            for k, kinds in [("retrieved", {"retrieved"}), ("programme", {"retrieved", "programme"})]:
                best = np.zeros(len(sents))
                for si in range(len(sents)):
                    m = (owner == si) & np.isin(kind, list(kinds))
                    if m.any():
                        best[si] = probs[m].max()
                supported = best >= ENTAIL_THRESHOLD
                rec[f"attr_{k}"] = supported.mean()
                rec[f"any_{k}"] = int(supported.any())
                rec[f"mean_entail_{k}"] = best.mean()
        batch_rows.append(rec)
        if n % 10 == 0 or n == len(todo):
            pd.DataFrame(batch_rows).to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
            batch_rows = []
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)} turns  ({rate:.2f} turns/s, ~{(len(todo) - n) / rate / 60:.0f} min left)", flush=True)


def analyse():
    df = load_turns()
    attr = pd.read_csv(CACHE)
    d = df.merge(attr, on="turn_id")
    d = d[d["n_sentences"] > 0]
    markers = pd.read_csv(OUT_DIR / "surface_markers_turns.csv")[["turn_id", "programme_ref"]]
    d = d.merge(markers, on="turn_id", how="left")
    res = {
        "n_turns": int(len(d)),
        "recovery_rate": float(d["n_passages_recovered"].sum() / d["n_passages_logged"].sum()),
        "overall": {k: float(d[k].mean()) for k in ["attr_retrieved", "attr_programme", "any_retrieved", "any_programme"]},
    }

    # Agreement with the judge's document-grounding score.
    res["spearman_judge"] = {}
    for k in ["attr_retrieved", "attr_programme"]:
        rho, p = stats.spearmanr(d["document_grounding"], d[k])
        res["spearman_judge"][k] = {"rho": float(rho), "p": float(p)}
    res["by_judge_score"] = d.groupby("document_grounding")[["attr_retrieved", "attr_programme"]].mean().to_dict(orient="index")

    # Condition means and debate-level tests vs. None.
    res["by_condition"] = d.groupby("condition_id")[["attr_retrieved", "attr_programme", "document_grounding"]].mean().to_dict(orient="index")
    tests = {}
    for k in ["attr_retrieved", "attr_programme"]:
        for cid, label in STRATEGIES.items():
            r = compare_groups(d, k, [cid], ["baseline_1"], bootstrap=True)
            tests[f"{label}|{k}"] = {kk: r[kk] for kk in ["a_mean", "b_mean", "diff", "diff_ci", "cliff_delta", "p_perm"]}
    for key, ph in zip(tests, holm([t["p_perm"] for t in tests.values()])):
        tests[key]["p_holm"] = ph
    res["vs_none"] = tests

    # Do explicit programme references come with traceable content?
    fr = d[d["condition_id"] == "strategy_c"]
    res["fact_reminder_by_programme_ref"] = fr.groupby("programme_ref")[
        ["attr_retrieved", "attr_programme", "document_grounding"]].mean().to_dict(orient="index")
    res["all_by_programme_ref"] = d.groupby("programme_ref")[
        ["attr_retrieved", "attr_programme", "document_grounding"]].mean().to_dict(orient="index")

    # Human sample: is the judge-human gap larger where attribution is low?
    exp = json.loads(EXPORT.read_text(encoding="utf-8"))
    excluded = set(exp["excluded_annotator_ids"])
    anns = pd.DataFrame(exp["annotations"])
    anns = anns[(anns["phase"] == "initial") & (~anns["perceived_bias"]) & (~anns["annotator_id"].isin(excluded))]
    anns["doc_h"] = anns["scores"].apply(lambda s: s.get("document_grounding"))
    human = anns.groupby("turn_id")["doc_h"].mean().rename("human_doc")
    h = d.merge(human, left_on="turn_id", right_index=True)
    h["gap"] = h["document_grounding"] - h["human_doc"]
    res["human_sample"] = {"n": int(len(h)), "mean_gap": float(h["gap"].mean())}
    for k in ["attr_retrieved", "attr_programme"]:
        rho_h, p_h = stats.spearmanr(h["human_doc"], h[k])
        rho_g, p_g = stats.spearmanr(h["gap"], h[k])
        res["human_sample"][k] = {"rho_human": float(rho_h), "p_human": float(p_h),
                                  "rho_gap": float(rho_g), "p_gap": float(p_g)}

    (OUT_DIR / "attribution.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="score only N turns (timing test)")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse()
    else:
        if args.limit:
            _select = select_turns
            select_turns = lambda df, k: _select(df, k).head(args.limit)  # noqa: E731
        run_nli(args.per_run)
