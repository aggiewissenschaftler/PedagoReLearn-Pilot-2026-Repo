"""
Week 12: Statistical Analysis of PedagoReLearn Experiments
----------------------------------------------------------
Calculates mean, std, confidence intervals, and effect sizes
for SARSA vs Fixed and Random baselines.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
results_dir = Path("results")
summary_file = results_dir / "summary_week11.csv"
output_file = results_dir / "stats_week12.csv"

# Load the Week 11 summary data
df = pd.read_csv(summary_file)

# Group by configuration and compute statistics
grouped = df.groupby("config").agg(["mean", "std", "count"])
stats_df = grouped.copy()

# Compute 95% confidence intervals
tval = stats.t.ppf(0.975, grouped[('reward_mean','count')].min() - 1)
stats_df[('reward_ci95_half','mean')] = (
    tval * grouped[('reward_mean','std')] / np.sqrt(grouped[('reward_mean','count')])
)

# Optional: calculate Cohen’s d effect size SARSA vs baselines
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd

sarsa = df[df["config"] == "SARSA"]["reward_mean"]
fixed = df[df["config"] == "Fixed"]["reward_mean"]
random = df[df["config"] == "Random"]["reward_mean"]

d_fixed = cohens_d(sarsa, fixed)
d_random = cohens_d(sarsa, random)

print(f"Cohen's d (SARSA vs Fixed): {d_fixed:.3f}")
print(f"Cohen's d (SARSA vs Random): {d_random:.3f}")

# Save stats
stats_df.to_csv(output_file)
print(f"Saved Week 12 statistics → {output_file}")

# Plot comparisons
plt.figure(figsize=(6,4))
means = df.groupby("config")["reward_mean"].mean()
cis = df.groupby("config")["reward_mean"].std() / np.sqrt(10) * tval
plt.bar(means.index, means, yerr=cis, capsize=5)
plt.ylabel("Mean Reward ± 95% CI")
plt.title("Week 12: SARSA vs Baseline Comparison")
plt.tight_layout()
plt.savefig(results_dir / "week12_barplot.png")
plt.show()