"""Cross-family judge check with and without the retrieved passages.

The original judge and moderator (Llama 4 Scout) are no longer available on Groq. This
script re-scores a stratified subsample with a judge from a different model family
(qwen/qwen3.8-27b, the project's current judge model), using the original judge
prompt in two variants:

  blind    - exactly the information the original judge had (topic, round, phase,
             agent, preceding turns, turn);
  passages - the same, plus the three passages retrieved for that turn.

Comparing the variants isolates the effect of seeing the sources on document-grounding;
comparing the blind variant with the original scores checks whether the strategy
effects replicate with a judge that did not write the moderator interventions.

Usage:
    python experiments/judge_validity/cross_judge.py --per-cell 25   # scoring (cached, resumable)
    python experiments/judge_validity/cross_judge.py --analyse

Outputs (experiments/outputs/judge_validity/): cross_judge_scores.csv, cross_judge.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import DIMENSIONS, OUT_DIR, ROOT, load_turns  # noqa: E402
from comparative_analysis import compare_groups, holm  # noqa: E402

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from src.judge import EvaluationJudge  # noqa: E402
from src.llm_client import JUDGE_MODEL, LLMClient  # noqa: E402
from src.models import Turn  # noqa: E402
from src.prompts.judge_prompts import format_judge_user_prompt  # noqa: E402

CONDITIONS = {"baseline_1": "None", "strategy_b": "Reframing",
              "strategy_c": "Fact-reminder", "strategy_d": "Common-ground"}
CACHE = OUT_DIR / "cross_judge_scores.csv"
EMB_DIR = ROOT / "data" / "embeddings"
PARTY_STEMS = {
    "CDU/CSU": "cdu_csu_2025", "SPD": "spd_2025", "Bündnis 90/Die Grünen": "gruene_2025",
    "FDP": "fdp_2025", "Die Linke": "linke_2025", "AfD": "afd_2025",
}
TURN_MARKER = "---\n\n### Turn to Evaluate:"


def recover_passages() -> dict:
    maps = {}
    for party, stem in PARTY_STEMS.items():
        lines = (
            EMB_DIR / f"{stem}_meta.jsonl").read_text(encoding="utf-8").splitlines()
        maps[party] = {r["text"][:100]: r["text"]
                       for r in map(json.loads, filter(None, lines))}
    return maps


def select_sample(df: pd.DataFrame, per_cell: int, seed: int = 2026) -> pd.DataFrame:
    """`per_cell` turns per (condition, topic), excluding round 1 (no intervention yet)."""
    pool = df[df["condition_id"].isin(CONDITIONS) & (df["round_number"] >= 2)]
    return pool.groupby(["condition_id", "topic_id"], group_keys=False).apply(
        lambda g: g.sample(per_cell, random_state=seed))


def build_prompts(row, run_turns: list[Turn], passages: list[str]) -> dict:
    idx = next(i for i, t in enumerate(run_turns)
               if t.turn_id == row["turn_id"])
    turn = run_turns[idx]
    phase = ("opening (rounds 1-3)" if turn.round_number <= 3 else
             "mid-debate (rounds 4-7)" if turn.round_number <= 7 else "closing (rounds 8-10)")
    blind = format_judge_user_prompt(
        topic=row["topic_id"], round_number=turn.round_number, phase=phase,
        agent_name=turn.agent_name, turn_text=turn.text,
        preceding_turns=EvaluationJudge._format_preceding(run_turns[max(0, idx - 12):idx]))
    block = "### RAG Grounding Passages (retrieved for this agent before the turn):\n" + "\n".join(
        f"[{i}] {p}" for i, p in enumerate(passages, 1)) + "\n\n"
    if TURN_MARKER not in blind:
        raise RuntimeError(
            "Judge user template changed; cannot insert passages")
    return {"blind": blind, "passages": blind.replace(TURN_MARKER, block + TURN_MARKER, 1)}


def run_scoring(per_cell: int):
    df = load_turns()
    sample = select_sample(df, per_cell)
    done = set(zip(*[pd.read_csv(CACHE)[c]
               for c in ["turn_id", "variant"]])) if CACHE.exists() else set()
    print(f"{len(sample)} turns x 2 variants; {len(done)} scores cached; judge model {JUDGE_MODEL}")

    client = LLMClient()
    judge = EvaluationJudge(client)
    system_prompt = judge._system_prompt
    passage_maps = recover_passages()
    run_cache: dict[str, list[Turn]] = {}

    t0, n_new = time.time(), 0
    for _, row in sample.iterrows():
        if row["run_file"] not in run_cache:
            raw = json.loads(
                (ROOT / row["run_file"]).read_text(encoding="utf-8"))["turns"]
            run_cache[row["run_file"]] = [Turn(turn_id=t["turn_id"], round_number=t["round_number"],
                                               turn_in_round=t["turn_in_round"], agent_name=t["agent_name"],
                                               text=t["text"]) for t in raw]
        passages = [passage_maps[row["agent_name"]].get(
            p) for p in row["rag_prefixes"]]
        passages = [p for p in passages if p]
        prompts = build_prompts(row, run_cache[row["run_file"]], passages)
        for variant, user_prompt in prompts.items():
            if (row["turn_id"], variant) in done:
                continue
            parsed = None
            for _ in range(3):
                resp = client.complete(model=JUDGE_MODEL, system_prompt=system_prompt, user_prompt=user_prompt,
                                       max_tokens=judge.max_tokens, temperature=0.0, parse_json=True)
                if resp.parsed_json and "scores" in resp.parsed_json:
                    parsed = resp.parsed_json
                    break
            rec = {"turn_id": row["turn_id"], "variant": variant, "model": JUDGE_MODEL,
                   "n_passages": len(passages), "parse_ok": int(parsed is not None)}
            scores = (parsed or {}).get("scores", {})
            for d in DIMENSIONS:
                v = scores.get(d)
                rec[d] = max(1, min(5, int(v))) if isinstance(
                    v, (int, float)) else np.nan
            rec["doc_justification"] = (parsed or {}).get(
                "justifications", {}).get("document_grounding", "")
            pd.DataFrame([rec]).to_csv(CACHE, mode="a",
                                       header=not CACHE.exists(), index=False)
            n_new += 1
            if n_new % 20 == 0:
                print(
                    f"  {n_new} new scores ({n_new / (time.time() - t0):.2f}/s)", flush=True)
    print(f"done: {n_new} new scores")


def analyse():
    df = load_turns()
    q = pd.read_csv(CACHE)
    q = q[q["parse_ok"] == 1]
    wide = q.pivot_table(index="turn_id", columns="variant", values=DIMENSIONS)
    wide.columns = [f"{d}_{v}" for d, v in wide.columns]
    d = df.merge(wide, left_on="turn_id", right_index=True)
    res = {"n_turns": int(len(d)), "model": JUDGE_MODEL,
           "parse_rate": float(pd.read_csv(CACHE)["parse_ok"].mean())}

    # Agreement between the original judge (Llama 4 Scout) and qwen-blind.
    agree = {}
    for dim in DIMENSIONS:
        a, b = d[dim], d[f"{dim}_blind"]
        m = a.notna() & b.notna()
        agree[dim] = {"exact": float((a[m] == b[m]).mean()), "within1": float(((a[m] - b[m]).abs() <= 1).mean()),
                      "spearman": float(stats.spearmanr(a[m], b[m])[0]), "mean_orig": float(a[m].mean()),
                      "mean_qwen": float(b[m].mean())}
    res["orig_vs_qwen_blind"] = agree

    # Effect of seeing the passages on document-grounding.
    dg = d.dropna(subset=["document_grounding_blind",
                  "document_grounding_passages"])
    diff = dg["document_grounding_passages"] - dg["document_grounding_blind"]
    res["passages_effect"] = {
        "mean_blind": float(dg["document_grounding_blind"].mean()),
        "mean_passages": float(dg["document_grounding_passages"].mean()),
        "mean_diff": float(diff.mean()), "share_changed": float((diff != 0).mean()),
        "share_lower": float((diff < 0).mean()), "share_higher": float((diff > 0).mean()),
        "wilcoxon_p": float(stats.wilcoxon(dg["document_grounding_passages"], dg["document_grounding_blind"])[1]),
        "by_condition": {CONDITIONS[c]: {"blind": float(g["document_grounding_blind"].mean()),
                                         "passages": float(g["document_grounding_passages"].mean())}
                         for c, g in dg.groupby("condition_id")},
    }

    # Strategy effects vs. None under each judge (debate means of the sampled turns).
    eff, pvals = {}, []
    for metric in ["responsiveness", "responsiveness_blind", "document_grounding",
                   "document_grounding_blind", "document_grounding_passages",
                   "stance_differentiation", "stance_differentiation_blind"]:
        for cid in ["strategy_b", "strategy_c", "strategy_d"]:
            r = compare_groups(d, metric, [cid], [
                               "baseline_1"], bootstrap=True)
            eff[f"{CONDITIONS[cid]}|{metric}"] = {k: r[k]
                                                  for k in ["a_mean", "b_mean", "diff", "diff_ci", "p_perm"]}
    res["vs_none"] = eff

    (OUT_DIR / "cross_judge.json").write_text(json.dumps(res,
                                                         indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    analyse() if args.analyse else run_scoring(args.per_cell)
