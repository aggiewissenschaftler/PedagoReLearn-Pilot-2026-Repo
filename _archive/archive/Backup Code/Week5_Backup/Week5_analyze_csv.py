import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime


def find_latest_csv(logs_dir: Path) -> Path | None:
    if not logs_dir.exists():
        return None
    csvs = sorted(logs_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def moving_average(x, w):
    if w is None or w <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w)/w, mode="valid")
    pad = np.full(w-1, c[0])
    return np.concatenate([pad, c])


def summarize(df: pd.DataFrame):
    out = {}
    out["episodes"] = int(df["episode"].iloc[-1] + 1) if "episode" in df else len(df)
    out["avg_return"] = float(df["return"].mean())
    out["mean_steps"] = float(df["steps"].mean()) if "steps" in df else float("nan")
    out["soft_mastery_%"] = float(100 * df.get("mastered_soft", pd.Series([0]*len(df))).mean())
    out["hard_mastery_%"] = float(100 * df.get("mastered_hard", pd.Series([0]*len(df))).mean())
    out["avg_of_avg_mastery"] = float(df.get("avg_mastery", pd.Series(dtype=float)).mean())
    out["final_avg_mastery"] = float(df.get("avg_mastery", pd.Series(dtype=float)).iloc[-1]) if "avg_mastery" in df else float("nan")
    out["final_epsilon"] = float(df.get("epsilon", pd.Series(dtype=float)).iloc[-1]) if "epsilon" in df else float("nan")
    return out


def print_summary(stats: dict, label: str, path: Path):
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


def extract_ts_from_csv(csv_path: Path) -> str | None:
    m = re.search(r'(\d{8}-\d{6})', csv_path.name)
    return m.group(1) if m else None


def plot_from_df(df: pd.DataFrame, smooth: int, save_dir: Path | None, csv_path: Path | None):
    ep = df["episode"] if "episode" in df else np.arange(len(df))
    ret = df["return"]
    avgm = df.get("avg_mastery", pd.Series(np.nan, index=ep.index))

    plt.figure()
    plt.plot(ep, moving_average(ret, smooth), label="Return")
    if "avg_mastery" in df:
        ax2 = plt.gca().twinx()
        ax2.plot(ep, moving_average(avgm, smooth), label="Avg Mastery", linestyle="--")
        ax2.set_ylabel("Avg Mastery (0–3)")
    plt.title("Training Summary")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # try to match csv timestamp if possible
        ts = extract_ts_from_csv(csv_path) if csv_path else None
        if ts is None:
            ts = datetime.now().strftime("%YMMDD-%H%M%S")
        out = save_dir / f"analyze_returns_mastery_{ts}.png"
        plt.savefig(out)
        print(f"[Saved plot] {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze SARSA training CSV logs.")
    parser.add_argument("--logs_dir", type=str, default="trace_results/logs/",
                        help="Directory containing CSV logs (default: trace_results/logs/)")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to a specific CSV (overrides logs_dir if provided)")
    parser.add_argument("--smooth", type=int, default=25,
                        help="Moving average window for plots (default: 25)")
    parser.add_argument("--save_figs", type=str, default="trace_results/figs/",
                        help="Directory to save plots (empty to show interactively)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv(Path(args.logs_dir))
    if csv_path is None or not csv_path.exists():
        print(f"No CSV found. Searched: {args.csv or args.logs_dir}")
        return

    df = pd.read_csv(csv_path)
    stats = summarize(df)
    print_summary(stats, label="Training Log Summary", path=csv_path)

    save_dir = Path(args.save_figs) if args.save_figs else None
    plot_from_df(df, smooth=args.smooth, save_dir=save_dir, csv_path=csv_path)


if __name__ == "__main__":
    main()