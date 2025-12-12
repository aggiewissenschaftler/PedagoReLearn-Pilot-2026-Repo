import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

from tutor_baselines import run_random_policy, run_fixed_curriculum
from tutor_train_sarsa import train_sarsa


# ---------- helpers ----------
def summarize(label, returns, steps, mastered):
    return {
        "Policy": label,
        "Avg Return": float(np.mean(returns)),
        "Mastery %": float(100 * np.mean(mastered)) if mastered else 0.0,
        "Episodes": int(len(returns)),
        "Mean Steps": float(np.mean(steps)) if steps else float("nan"),
    }


def print_table(rows):
    head = ["Policy", "Avg Return", "Mastery %", "Episodes", "Mean Steps"]
    widths = [16, 12, 10, 9, 11]
    line = "-" * (sum(widths) + len(widths) * 3 + 1)
    print(line)
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(head, widths)) + " |")
    print(line)
    for r in rows:
        print("| " + " | ".join([
            f"{r['Policy']:<{widths[0]}}",
            f"{r['Avg Return']:{widths[1]}.2f}",
            f"{r['Mastery %']:{widths[2]}.1f}",
            f"{r['Episodes']:{widths[3]}}",
            f"{r['Mean Steps']:{widths[4]}.2f}",
        ]) + " |")
    print(line)


def moving_average(x, w):
    if not w or w <= 1:
        return x
    x = np.asarray(x, dtype=float)
    if len(x) < w:  # too short to smooth
        return x
    c = np.convolve(x, np.ones(w)/w, mode="valid")
    pad = np.full(w-1, c[0])
    return np.concatenate([pad, c])


def plot_curves(series, title="Episode Returns", save_path: Path | None = None):
    plt.figure()
    for label, data in series:
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved plot] {save_path}")
    else:
        plt.show()


def with_timestamp(path: Path, ts: str) -> Path:
    """Append _<ts> before suffix, preserving parent/stem/suffix."""
    return path.with_name(path.stem + f"_{ts}" + path.suffix)


def extract_ts_from_csv(csv_path: Path) -> str | None:
    """
    Try to extract YYYYMMDD-HHMMSS from a CSV named like sarsa_train_log_YYYYMMDD-HHMMSS.csv.
    """
    m = re.search(r'(\d{8}-\d{6})', csv_path.name)
    return m.group(1) if m else None


def latest_csv_in_dir(dir_path: Path) -> Path | None:
    if not dir_path.exists():
        return None
    csvs = sorted(dir_path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


# ---------- main ----------
def main():
    # sensible defaults to your requested folder
    default_fig = Path("trace_results/figs/compare_returns.png")
    default_logs_dir = Path("../../../trace_results/logs/")

    p = argparse.ArgumentParser(description="Evaluate Random, Fixed, and SARSA on PedagoReLearnEnv.")
    p.add_argument("--N", type=int, default=3, help="Number of cultural rules.")
    p.add_argument("--episodes_random", type=int, default=200)
    p.add_argument("--episodes_fixed", type=int, default=200)
    p.add_argument("--episodes_sarsa", type=int, default=600)
    p.add_argument("--seed_random", type=int, default=123)
    p.add_argument("--seed_fixed", type=int, default=7)
    p.add_argument("--seed_sarsa", type=int, default=0)

    # outputs (now default to trace_results/*)
    p.add_argument("--no_plots", action="store_true", help="Disable plots.")
    p.add_argument("--save_plot", type=str, default=str(default_fig),
                   help=f"Save figure path (default: {default_fig}).")
    p.add_argument("--smooth", type=int, default=25,  # default smoothing
                   help="Moving-average window; 0 disables.")
    p.add_argument("--save_csv_sarsa", type=str, default=str(default_logs_dir),
                   help=f"Folder or CSV path for SARSA logs (default: {default_logs_dir}).")
    p.add_argument("--soft_mastery", type=float, default=2.5,
                   help="Avg mastery threshold for 'soft mastered'.")
    args = p.parse_args()

    # Baselines
    rr, rsteps, rm = run_random_policy(num_episodes=args.episodes_random, seed=args.seed_random, N=args.N)
    fr, fsteps, fm = run_fixed_curriculum(num_episodes=args.episodes_fixed, seed=args.seed_fixed, N=args.N)

    # SARSA (with CSV logging + soft mastery)
    sr, ssteps, sm = train_sarsa(
        num_episodes=args.episodes_sarsa,
        seed=args.seed_sarsa,
        N=args.N,
        log_csv_path=args.save_csv_sarsa if args.save_csv_sarsa else None,
        soft_mastery_threshold=args.soft_mastery,
    )

    # Table
    rows = [
        summarize("Random Policy", rr, rsteps, rm),
        summarize("Fixed Curriculum", fr, fsteps, fm),
        summarize("SARSA (Trained)", sr, ssteps, sm),
    ]
    print_table(rows)

    # Determine timestamp to pair PNG with matching CSV (best-effort)
    save_plot_path = Path(args.save_plot) if args.save_plot else None
    ts = None
    # If save_csv_sarsa is a directory (or looks like one), use its latest CSV's timestamp
    logs_path = Path(args.save_csv_sarsa)
    if logs_path.suffix == "" or str(logs_path).endswith("/") or logs_path.is_dir():
        latest = latest_csv_in_dir(logs_path)
        if latest:
            ts = extract_ts_from_csv(latest)

    if ts is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plots (with smoothing + timestamped save path)
    if not args.no_plots:
        w = args.smooth if args.smooth and args.smooth > 1 else 0
        series = [
            ("Random", moving_average(rr, w) if w else rr),
            ("Fixed",  moving_average(fr, w) if w else fr),
            ("SARSA",  moving_average(sr, w) if w else sr),
        ]
        if save_plot_path is not None:
            save_plot_path = with_timestamp(save_plot_path, ts)
        plot_curves(series, title=f"Returns by Episode (N={args.N})", save_path=save_plot_path)


if __name__ == "__main__":
    main()