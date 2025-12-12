#!/usr/bin/env python3
"""
Unified CSV analyzer for PedagoReLearn.

It auto-detects one of two schemas:
A) TRAINING CSV (your original):
   columns may include: episode, return, steps, avg_mastery, mastered_soft, mastered_hard, epsilon, ...
B) RUNNER CSV (from train_runner.py):
   columns: policy, seed, episode, steps, reward, success, time_s

Usage examples:
  python analyze_csv.py
  python analyze_csv.py --csv trace_results/run_log.csv --plot trace_results/plot.png
  python analyze_csv.py --logs_dir trace_results/logs/ --smooth 25 --save_figs trace_results/figs/
"""

from __future__ import annotations
import argparse
import pathlib
import sys
import re
from datetime import datetime

import pandas as pd
import numpy as np

# matplotlib is optional; only used if plotting is requested
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# -------------------- Common helpers --------------------
def find_latest_csv(logs_dir: pathlib.Path) -> pathlib.Path | None:
    if not logs_dir.exists():
        return None
    csvs = sorted(
        logs_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


def moving_average(x, w):
    if not w or w <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, c[0])
    return np.concatenate([pad, c])


def extract_ts_from_csv(csv_path: pathlib.Path) -> str | None:
    m = re.search(r'(\d{8}-\d{6})', csv_path.name)
    return m.group(1) if m else None


# -------------------- Schema detection --------------------
def detect_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if {"policy", "seed", "episode", "steps", "reward", "success"}.issubset(cols):
        return "runner"
    if {"episode", "return"}.issubset(cols) or {"avg_mastery", "epsilon"}.intersection(cols):
        return "training"
    return "unknown"


# -------------------- TRAINING schema (your original) --------------------
def summarize_training(df: pd.DataFrame) -> dict:
    out = {}
    out["episodes"] = int(df["episode"].iloc[-1] + 1) if "episode" in df else len(df)
    out["avg_return"] = float(df["return"].mean()) if "return" in df else float("nan")
    out["mean_steps"] = float(df["steps"].mean()) if "steps" in df else float("nan")
    out["soft_mastery_%"] = float(100 * df.get("mastered_soft", pd.Series([0]*len(df))).mean())
    out["hard_mastery_%"] = float(100 * df.get("mastered_hard", pd.Series([0]*len(df))).mean())
    out["avg_of_avg_mastery"] = float(df.get("avg_mastery", pd.Series(dtype=float)).mean())
    out["final_avg_mastery"] = float(df.get("avg_mastery", pd.Series(dtype=float)).iloc[-1]) if "avg_mastery" in df else float("nan")
    out["final_epsilon"] = float(df.get("epsilon", pd.Series(dtype=float)).iloc[-1]) if "epsilon" in df else float("nan")
    return out


def print_training_summary(stats: dict, label: str, path: pathlib.Path):
    line = "-" * 66
    print(line)
    print(f"{label} :: {path}")
    print(line)
    print(f"Episodes          : {stats['episodes']}")
    print(f"Avg Return        : {stats['avg_return']:.2f}")
    print(f"Mean Steps        : {stats['mean_steps']:.2f}")
    print(f"Soft Mastery %    : {stats['soft_mastery_%']:.1f}")
    print(f"Hard Mastery %    : {stats['hard_mastery_%']:.1f}")
    print(f"Avg of AvgMastery : {stats['avg_of_avg_mastery']:.3f}")
    print(f"Final AvgMastery  : {stats['final_avg_mastery']:.3f}")
    print(f"Final Epsilon     : {stats['final_epsilon']:.3f}")
    print(line)


def plot_training(df: pd.DataFrame, smooth: int, save_dir: pathlib.Path | None, csv_path: pathlib.Path | None):
    if not _HAVE_MPL:
        print("[plot skipped] matplotlib not available.")
        return
    ep = df["episode"] if "episode" in df else np.arange(len(df))
    ret = df["return"] if "return" in df else pd.Series(np.nan, index=ep.index)
    avgm = df.get("avg_mastery", pd.Series(np.nan, index=ep.index))

    fig, ax1 = plt.subplots()
    ax1.plot(ep, moving_average(ret, smooth), label="Return")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")

    if "avg_mastery" in df:
        ax2 = ax1.twinx()
        ax2.plot(ep, moving_average(avgm, smooth), label="Avg Mastery", linestyle="--")
        ax2.set_ylabel("Avg Mastery (0–3)")

    plt.title("Training Summary")
    plt.tight_layout()

    if save_dir is None:
        plt.show()
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = extract_ts_from_csv(csv_path) or datetime.now().strftime("%Y%m%d-%H%M%S")
        out = save_dir / f"training_summary_{ts}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Saved plot] {out}")


# -------------------- RUNNER schema (from train_runner.py) --------------------
def summarize_runner(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize dtypes
    df["policy"] = df["policy"].astype(str)
    for col in ("seed", "episode", "steps", "reward", "success", "time_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Aggregate per policy
    grp = df.groupby("policy", as_index=False).agg(
        episodes=("episode", "count"),
        seeds=("seed", lambda s: len(pd.unique(s.dropna()))),
        mean_steps=("steps", "mean"),
        std_steps=("steps", "std"),
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        success_rate=("success", "mean"),
        mean_time_s=("time_s", "mean"),
    )
    grp["std_steps"] = grp["std_steps"].fillna(0.0)
    grp["std_reward"] = grp["std_reward"].fillna(0.0)
    grp["success_rate"] = 100.0 * grp["success_rate"].fillna(0.0)
    return grp.sort_values("policy")


def print_runner_summary(summary: pd.DataFrame):
    line = "-" * 78
    print(line)
    print("Runner Log Summary (per policy)")
    print(line)
    if summary.empty:
        print("(no rows)")
        print(line)
        return
    # Pretty plain-text columns
    print(f"{'policy':<10} {'episodes':>9} {'seeds':>7} {'mean_steps':>12} {'std_steps':>10} "
          f"{'mean_reward':>12} {'std_reward':>11} {'success_%':>10} {'mean_time_s':>12}")
    for _, r in summary.iterrows():
        print(f"{str(r['policy'])[:10]:<10} {int(r['episodes']):>9} {int(r['seeds']):>7} "
              f"{r['mean_steps']:.2f:>12} {r['std_steps']:.2f:>10} "
              f"{r['mean_reward']:.3f:>12} {r['std_reward']:.3f:>11} {r['success_rate']:.1f:>10} {r['mean_time_s']:.3f:>12}")
    print(line)


def plot_runner(df: pd.DataFrame, smooth: int, save_dir: pathlib.Path | None, csv_path: pathlib.Path | None):
    if not _HAVE_MPL:
        print("[plot skipped] matplotlib not available.")
        return
    # Plot mean reward by episode (averaged over seeds) for each policy
    plt.figure()
    for policy, g in df.groupby("policy"):
        g = g.copy()
        g["episode"] = pd.to_numeric(g["episode"], errors="coerce")
        g["reward"] = pd.to_numeric(g["reward"], errors="coerce")
        g = g.dropna(subset=["episode", "reward"])
        mean_by_ep = g.groupby("episode", as_index=True)["reward"].mean().sort_index()
        plt.plot(mean_by_ep.index.values, moving_average(mean_by_ep.values, smooth), label=str(policy))
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Runner: Mean Reward by Episode")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        plt.show()
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = extract_ts_from_csv(csv_path) or datetime.now().strftime("%Y%m%d-%H%M%S")
        out = save_dir / f"runner_mean_reward_{ts}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Saved plot] {out}")


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Unified analyzer for PedagoReLearn CSV logs.")
    parser.add_argument("--logs_dir", type=str, default="trace_results/logs/",
                        help="Directory to search for latest CSV if --csv not given")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to a specific CSV (overrides logs_dir if provided)")
    parser.add_argument("--smooth", type=int, default=25,
                        help="Moving average window for plots")
    parser.add_argument("--save_figs", type=str, default="trace_results/figs/",
                        help="Directory to save plots (empty to show interactively)")
    parser.add_argument("--plot", action="store_true",
                        help="If set, render/save a plot appropriate to the detected schema")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv) if args.csv else find_latest_csv(pathlib.Path(args.logs_dir))
    if csv_path is None or not csv_path.exists():
        print(f"No CSV found. Searched: {args.csv or args.logs_dir}")
        return

    df = pd.read_csv(csv_path)
    schema = detect_schema(df)

    if schema == "training":
        stats = summarize_training(df)
        print_training_summary(stats, label="Training Log Summary", path=csv_path)
        if args.plot:
            save_dir = pathlib.Path(args.save_figs) if args.save_figs else None
            plot_training(df, smooth=args.smooth, save_dir=save_dir, csv_path=csv_path)

    elif schema == "runner":
        summary = summarize_runner(df)
        print_runner_summary(summary)
        if args.plot:
            save_dir = pathlib.Path(args.save_figs) if args.save_figs else None
            plot_runner(df, smooth=args.smooth, save_dir=save_dir, csv_path=csv_path)

    else:
        print(f"Unrecognized CSV schema for file: {csv_path}")
        print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
