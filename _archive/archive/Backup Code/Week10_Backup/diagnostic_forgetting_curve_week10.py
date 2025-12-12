"""
Week 10 Diagnostic — Ebbinghaus Forgetting Curve
Generates a simple 50-episode decay test and figure for your report.
"""

import os, math, csv
import matplotlib.pyplot as plt

# === Parameters ===
DECAY_RATE = 0.05       # λ — tune if desired
EPISODES = 50            # time steps to simulate
INITIAL_MASTERY = 1.0    # starting mastery level
REVIEW_AT = [10, 30]     # episodes where review occurs
BOOST = 0.25             # mastery boost on review
OUT_DIR = "figs"
CSV_OUT = "results/forgetting_curve_week10.csv"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

# === Forgetting function ===
def forgetting_curve(t, decay_rate=DECAY_RATE):
    return math.exp(-decay_rate * t)

# === Simulation ===
mastery = INITIAL_MASTERY
history = []

for ep in range(EPISODES + 1):
    if ep in REVIEW_AT:
        mastery = min(1.0, mastery + BOOST)
    else:
        # Apply Ebbinghaus decay for one time step
        mastery *= forgetting_curve(1, DECAY_RATE)
    history.append((ep, mastery))

# === Save CSV ===
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "mastery"])
    writer.writerows(history)

# === Plot ===
episodes, levels = zip(*history)
plt.figure(figsize=(6, 4))
plt.plot(episodes, levels, marker="o", label=f"λ={DECAY_RATE}")
for r in REVIEW_AT:
    plt.axvline(r, color="orange", linestyle="--", alpha=0.6, label="Review" if r == REVIEW_AT[0] else None)
plt.title("Ebbinghaus Forgetting Curve with Reviews (Week 10)")
plt.xlabel("Episodes (time)")
plt.ylabel("Retention / Mastery Level")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forgetting_curve_week10.png"))
plt.close()

print(f"✅ Figure saved to {OUT_DIR}/forgetting_curve_week10.png")
print(f"✅ Data saved to {CSV_OUT}")