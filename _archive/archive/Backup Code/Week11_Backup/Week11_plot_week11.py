import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRJ = Path(__file__).resolve().parent
R = PRJ / "results"

def stack(pattern, label):
    dfs = []
    for f in glob.glob(str(R / pattern)):
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        episode = cols.get("episode")
        reward  = cols.get("reward") or cols.get("return") or cols.get("avg_return")
        steps   = cols.get("steps") or cols.get("n_steps")
        acc     = cols.get("acc") or cols.get("accuracy")
        keep = [episode]
        for c in [reward, steps, acc]:
            if c: keep.append(c)
        df = df[keep].copy()
        df.columns = ["episode"] + [x for x in ["reward", "steps", "acc"] if x in [reward, steps, acc]]
        dfs.append(df)
    out = pd.concat(dfs)
    grp = out.groupby("episode")
    agg = pd.DataFrame({"episode": grp.size().index})
    for m in [x for x in ["reward", "steps", "acc"] if x in out.columns]:
        agg[f"{m}_mean"] = grp[m].mean()
        agg[f"{m}_sd"] = grp[m].std()
        agg[f"{m}_n"] = grp[m].count()
    agg["config"] = label
    return agg

plots = [
    stack("curves_full_seed*.csv", "SARSA (full)"),
    stack("curves_aggregated_seed*.csv", "SARSA (agg)"),
    stack("curves_fixed_seed*.csv", "Fixed"),
    stack("curves_random_seed*.csv", "Random"),
]
df = pd.concat(plots)

def draw(metric, ylabel, fname):
    if f"{metric}_mean" not in df.columns:
        return
    plt.figure()
    for label, g in df.groupby("config"):
        m = g[f"{metric}_mean"]
        sd = g[f"{metric}_sd"]
        n = g[f"{metric}_n"].replace(0, 1)
        ci = 1.96 * sd / np.sqrt(n)
        plt.plot(g["episode"], m, label=label)
        plt.fill_between(g["episode"], m - ci, m + ci, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(R / fname, dpi=200)

draw("steps", "Steps per episode (↓ better)", "curve_steps.png")
draw("reward", "Cumulative reward (↑ better)", "curve_reward.png")
draw("acc", "Quiz accuracy (↑ better)", "curve_acc.png")

print("✅ Saved results/curve_*.png")
