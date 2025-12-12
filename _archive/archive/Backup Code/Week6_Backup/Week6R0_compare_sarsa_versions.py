# compare_sarsa_versions.py
# Compare Week 5 (baseline SARSA) vs Week 6 (reward-shaped SARSA)
# Loads latest CSVs from trace_results/logs and saves timestamped comparison plots.

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOGS_DIR_DEFAULT = Path("trace_results/logs/")
FIGS_DIR_DEFAULT = Path("trace_results/figs/")


def find_latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def moving_average(x, w: int):
    if not w or w <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, c[0])
    return np.concatenate([pad, c])


def extract_ts(p: Path) -> str | None:
    m = re.search(r"(\d{8}-\d{6})", p.name)
    return m.group(1) if m else None


def plot_returns(baseline_df, rewarded_df, smooth: int, outdir: Path, ts: str):
    plt.figure()
    ep_b = baseline_df["episode"] if "episode" in baseline_df else np.arange(len(baseline_df))
    ep_r = rewarded_df["episode"] if "episode" in rewarded_df else np.arange(len(rewarded_df))

    plt.plot(ep_b, moving_average(baseline_df["return"], smooth), label="SARSA (Baseline)")
    plt.plot(ep_r, moving_average(rewarded_df["return"], smooth), label="SARSA (Reward-Shaped)")

    plt.title("Returns by Episode — Baseline vs Reward-Shaped")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"compare_returns_{ts}.png"
    plt.savefig(out)
    print(f"[Saved] {out}")


def plot_mastery(baseline_df, rewarded_df, smooth: int, outdir: Path, ts: str):
    # Use avg_mastery if present; otherwise fall back to a proxy (zeros)
    am_b = baseline_df.get("avg_mastery", pd.Series(np.zeros(len(baseline_df))))
    am_r = rewarded_df.get("avg_mastery", pd.Series(np.zeros(len(rewarded_df))))
    ep_b = baseline_df["episode"] if "episode" in baseline_df else np.arange(len(baseline_df))
    ep_r = rewarded_df["episode"] if "episode" in rewarded_df else np.arange(len(rewarded_df))

    plt.figure()
    plt.plot(ep_b, moving_average(am_b, smooth), label="Avg Mastery — Baseline")
    plt.plot(ep_r, moving_average(am_r, smooth), label="Avg Mastery — Reward-Shaped")
    plt.title("Average Mastery by Episode — Baseline vs Reward-Shaped")
    plt.xlabel("Episode")
    plt.ylabel("Avg Mastery (0–3)")
    plt.legend()
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"compare_mastery_{ts}.png"
    plt.savefig(out)
    print(f"[Saved] {out}")


def main():
    ap = argparse.ArgumentParser(description="Compare baseline SARSA vs reward-shaped SARSA.")
    ap.add_argument("--logs_dir", type=str, default=str(LOGS_DIR_DEFAULT),
                    help="Directory holding CSV logs (default: trace_results/logs/)")
    ap.add_argument("--figs_dir", type=str, default=str(FIGS_DIR_DEFAULT),
                    help="Directory to save comparison figures (default: trace_results/figs/)")
    ap.add_argument("--smooth", type=int, default=25, help="Moving average window (default: 25)")
    ap.add_argument("--baseline_csv", type=str, default="", help="Optional explicit baseline CSV path")
    ap.add_argument("--rewarded_csv", type=str, default="", help="Optional explicit rewarded CSV path")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    figs_dir = Path(args.figs_dir)

    # Find files
    if args.baseline_csv:
        baseline_csv = Path(args.baseline_csv)
    else:
        baseline_csv = find_latest(logs_dir, "sarsa_train_log_*.csv")

    if args.rewarded_csv:
        rewarded_csv = Path(args.rewarded_csv)
    else:
        rewarded_csv = find_latest(logs_dir, "sarsa_rewarded_train_log_*.csv")

    # Safety checks
    if baseline_csv is None or not baseline_csv.exists():
        print(f"❌ Could not find baseline CSV. Looked for sarsa_train_log_*.csv in {logs_dir}")
        return
    if rewarded_csv is None or not rewarded_csv.exists():
        print(f"❌ Could not find rewarded CSV. Looked for sarsa_rewarded_train_log_*.csv in {logs_dir}")
        return

    print(f"[Baseline] {baseline_csv}")
    print(f"[Rewarded] {rewarded_csv}")

    # Load
    df_b = pd.read_csv(baseline_csv)
    df_r = pd.read_csv(rewarded_csv)

    # Timestamp for outputs: try matching the rewarded run; fall back to now()
    ts = extract_ts(rewarded_csv) or extract_ts(baseline_csv) or datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plots
    plot_returns(df_b, df_r, smooth=args.smooth, outdir=figs_dir, ts=ts)
    plot_mastery(df_b, df_r, smooth=args.smooth, outdir=figs_dir, ts=ts)

    # Console summary
    def safe_mean(s): return float(np.mean(s)) if len(s) else float("nan")
    print("\n--- Summary (last run stats) ---")
    print(f"Baseline  avg_return={safe_mean(df_b['return']):.2f}  avg_mastery={safe_mean(df_b.get('avg_mastery', [])):.2f}")
    print(f"Rewarded  avg_return={safe_mean(df_r['return']):.2f}  avg_mastery={safe_mean(df_r.get('avg_mastery', [])):.2f}")


if __name__ == "__main__":
    main()