"""
plot_trace_results.py
----------------------------------------
Loads recent trace_results_*.csv files and produces:
1) Line plots: running_mean_steps & running_mean_reward vs episode (per tag × algo)
2) Grouped bar charts: mean steps, mean reward, timeout rate % (per tag × algo)

Requires: pandas, matplotlib
Run:
    python plot_trace_results.py

Outputs (saved in trace_results/):
    trace_results_lines.png
    trace_results_bars.png
----------------------------------------
"""

from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_recent_csvs(trace_dir: Path, max_files: int = 5) -> pd.DataFrame:
    files = sorted(glob.glob(str(trace_dir / "trace_results_*.csv")))
    if not files:
        raise FileNotFoundError("No CSV files found in trace_results/. Run train_eval.py first.")
    files = files[-max_files:]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source"] = Path(f).name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # Ensure required columns exist
    for col, default in [("tag", "untagged")]:
        if col not in out.columns:
            out[col] = default
    return out


def make_line_plots(df: pd.DataFrame, out_path: Path):
    # Normalize types
    df = df.copy()
    df["episode"] = df["episode"].astype(int)
    # Select only the columns we need
    cols_needed = {"tag", "algo", "episode", "running_mean_steps", "running_mean_reward"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for line plots: {missing}")

    # Figure with two rows (steps, reward)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot running mean steps by (tag, algo)
    for (tag, algo), sub in df.groupby(["tag", "algo"]):
        sub = sub.sort_values("episode")
        axes[0].plot(sub["episode"], sub["running_mean_steps"], label=f"{algo} ({tag})")
    axes[0].set_ylabel("Running Mean Steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot running mean reward by (tag, algo)
    for (tag, algo), sub in df.groupby(["tag", "algo"]):
        sub = sub.sort_values("episode")
        axes[1].plot(sub["episode"], sub["running_mean_reward"], label=f"{algo} ({tag})")
    axes[1].set_ylabel("Running Mean Reward")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Columns we need for bars
    cols_needed = {"tag", "algo", "steps_to_mastery", "total_reward", "done_reason"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for bar charts: {missing}")
    g = (
        df.groupby(["tag", "algo"], as_index=False)
          .agg(
              mean_steps=("steps_to_mastery", "mean"),
              mean_reward=("total_reward", "mean"),
              n=("algo", "size"),
              timeouts=("done_reason", lambda s: (s == "timeout").sum()),
          )
    )
    g["timeout_rate_pct"] = 100.0 * g["timeouts"] / g["n"]
    return g


def make_grouped_bars(metrics: pd.DataFrame, out_path: Path):
    # One figure with 3 horizontal subplots: mean steps, mean reward, timeout rate %
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Ensure stable order: sort by tag then algo
    metrics = metrics.sort_values(["tag", "algo"]).reset_index(drop=True)

    # Build x labels like "a-SARSA", "a-Random", "b-SARSA", ...
    xlabels = [f"{t}-{a}" for t, a in zip(metrics["tag"], metrics["algo"])]
    x = range(len(metrics))

    # Bar 1: mean steps
    axes[0].bar(x, metrics["mean_steps"])
    axes[0].set_title("Mean Steps")
    axes[0].set_ylabel("Steps")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, rotation=45, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Bar 2: mean reward
    axes[1].bar(x, metrics["mean_reward"])
    axes[1].set_title("Mean Reward")
    axes[1].set_ylabel("Reward")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Bar 3: timeout rate %
    axes[2].bar(x, metrics["timeout_rate_pct"])
    axes[2].set_title("Timeout Rate (%)")
    axes[2].set_ylabel("%")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xlabels, rotation=45, ha="right")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    trace_dir = Path("trace_results")
    trace_dir.mkdir(exist_ok=True)

    df = load_recent_csvs(trace_dir, max_files=5)

    # 1) Lines
    lines_path = trace_dir / "trace_results_lines.png"
    make_line_plots(df, lines_path)
    print(f"✅ Saved line plots to: {lines_path.resolve()}")

    # 2) Bars
    metrics = compute_group_metrics(df)
    bars_path = trace_dir / "trace_results_bars.png"
    make_grouped_bars(metrics, bars_path)
    print(f"✅ Saved grouped bars to: {bars_path.resolve()}")


if __name__ == "__main__":
    main()