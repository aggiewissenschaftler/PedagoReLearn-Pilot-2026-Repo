import glob, math
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PRJ = Path(__file__).resolve().parent
R = PRJ / "results"

def load_group(pattern, label):
    files = sorted(glob.glob(str(R / pattern)))
    if not files:
        raise FileNotFoundError(f"No files for {label}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        episode = cols.get("episode")
        reward  = cols.get("reward") or cols.get("return") or cols.get("avg_return")
        steps   = cols.get("steps") or cols.get("n_steps") or cols.get("ep_len")
        acc     = cols.get("acc") or cols.get("accuracy")
        seedcol = cols.get("seed")
        keep = [episode, seedcol]
        for c in [reward, steps, acc]:
            if c: keep.append(c)
        df = df[keep].copy()
        df.columns = ["episode", "seed"] + [x for x in ["reward", "steps", "acc"] if x in [reward, steps, acc]]
        df["config"] = label
        frames.append(df)
    out = pd.concat(frames)
    last = out.sort_values(["seed", "episode"]).groupby(["config", "seed"]).tail(1)
    return out, last

full_all, full_last = load_group("curves_full_seed*.csv", "SARSA_full")
agg_all,  agg_last  = load_group("curves_aggregated_seed*.csv", "SARSA_agg")
fix_all,  fix_last  = load_group("curves_fixed_seed*.csv", "Fixed")
rnd_all,  rnd_last  = load_group("curves_random_seed*.csv", "Random")

all_last = pd.concat([full_last, agg_last, fix_last, rnd_last])

def summarize(df, metric):
    if metric not in df.columns:
        return None
    g = df.groupby("config")[metric].agg(["mean", "std", "count"])
    ci = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    g = g.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_sd", "count": f"{metric}_n"})
    g[f"{metric}_ci95_half"] = ci
    return g

parts = []
for m in ["reward", "steps", "acc"]:
    s = summarize(all_last, m)
    if s is not None:
        parts.append(s)

summary = pd.concat(parts, axis=1).reset_index()
summary.to_csv(R / "summary_week11.csv", index=False)

def paired_vs_fixed(metric, cfg):
    A = all_last[all_last["config"] == cfg][["seed", metric]].dropna()
    B = all_last[all_last["config"] == "Fixed"][["seed", metric]].dropna().rename(columns={metric: f"{metric}_b"})
    M = A.merge(B, on="seed", how="inner")
    if len(M) < 2:
        return {"config": cfg, "metric": metric, "t": np.nan, "p": np.nan, "cohen_d": np.nan, "n_pairs": len(M)}
    a, b = M[metric].to_numpy(), M[f"{metric}_b"].to_numpy()
    t, p = stats.ttest_rel(a, b)
    d = (a - b).mean() / (a - b).std(ddof=1)
    return {"config": cfg, "metric": metric, "t": t, "p": p, "cohen_d": d, "n_pairs": len(M)}

rows = []
for cfg in ["SARSA_full", "SARSA_agg", "Random"]:
    for m in ["steps", "reward", "acc"]:
        if m in all_last.columns:
            rows.append(paired_vs_fixed(m, cfg))

stats_df = pd.DataFrame(rows)
k = stats_df["p"].notna().sum()
stats_df["p_bonf"] = stats_df["p"] * (k if k > 0 else np.nan)
stats_df.to_csv(R / "stats_week11.csv", index=False)

print("✅ Wrote summary_week11.csv and stats_week11.csv")
