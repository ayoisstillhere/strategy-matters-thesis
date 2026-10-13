import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comparative_analysis import load_posthoc, DIMENSIONS  # noqa: E402

full = load_posthoc()
df = full[~full["is_ablation"]].copy()
print("columns:", list(df.columns))
print("turns:", len(df), "runs:", df["run_file"].nunique())

print("\ncomposite by condition")
print(df.groupby("condition_id")["composite"].mean().round(3).to_dict())

print("\nresponsiveness per round")
for cid in ["baseline_1", "strategy_b", "strategy_d"]:
    s = df[df["condition_id"] == cid].groupby(
        "round_number")["responsiveness"].mean().round(2)
    print(cid, s.tolist())
print("stance all", df.groupby("round_number")[
      "stance_differentiation"].mean().round(2).tolist())

print("\nstrategy means by topic type")
strat = df[df["is_strategy"]]
print(strat.groupby("topic_type")[DIMENSIONS].mean().round(3).to_string())

party_col = next(c for c in ["party", "agent",
                 "agent_name", "speaker"] if c in df.columns)
print("\nparty means (all conditions)")
print(df.groupby(party_col)[DIMENSIONS].mean().round(3).to_string())
