import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tutor_baselines import run_random_policy, run_fixed_curriculum
from tutor_train_sarsa import train_sarsa


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
    if len(x) < w:
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


def main():
    p = argparse.ArgumentParser(description="Evaluate Random, Fixed, and SARSA on PedagoReLearnEnv.")
    p.add_argument("--N", type=int, default=3, help="Number of cultural rules.")
    p.add_argument("--episodes_random", type=int, default=200)
    p.add_argument("--episodes_fixed", type=int, default=200)
    p.add_argument("--episodes_sarsa", type=int, default=600)
    p.add_argument("--seed_random", type=int, default=123)
    p.add_argument("--seed_fixed", type=int, default=7)
    p.add_argument("--seed_sarsa", type=int, default=0)
    p.add_argument("--no_plots", action="store_true", help="Disable plots.")
    p.add_argument("--save_plot", type=str, default="", help="Save figure to this path.")
    p.add_argument("--smooth", type=int, default=0, help="Moving-average window; 0 disables.")
    p.add_argument("--save_csv_sarsa", type=str, default="", help="Folder or CSV path to save SARSA episode logs.")
    p.add_argument("--soft_mastery", type=float, default=2.5, help="Avg mastery threshold for 'soft mastered'.")
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

    rows = [
        summarize("Random Policy", rr, rsteps, rm),
        summarize("Fixed Curriculum", fr, fsteps, fm),
        summarize("SARSA (Trained)", sr, ssteps, sm),
    ]
    print_table(rows)

    if not args.no_plots:
        save_path = Path(args.save_plot) if args.save_plot else None
        w = args.smooth if args.smooth and args.smooth > 1 else 0
        series = [
            ("Random", moving_average(rr, w) if w else rr),
            ("Fixed",  moving_average(fr, w) if w else fr),
            ("SARSA",  moving_average(sr, w) if w else sr),
        ]
        plot_curves(series, title=f"Returns by Episode (N={args.N})", save_path=save_path)


if __name__ == "__main__":
    main()