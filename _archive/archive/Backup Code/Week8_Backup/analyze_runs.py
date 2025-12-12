#!/usr/bin/env python3
"""
Analyze multiple PedagoReLearn runs by reading runs/run_*/summary.json.

Outputs:
- Pretty console leaderboard (Rich)
- Optional CSV export (--out-csv)
- Optional Markdown report (--out-md)
- Optional comparison chart (--plot)

Usage examples:
  python analyze_runs.py
  python analyze_runs.py --roots runs          # default
  python analyze_runs.py --roots runs,other/dir --out-csv compare.csv --plot
  python analyze_runs.py --sort success_rate_pct --desc --top 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ---------- helpers ----------
def _find_summaries(roots: List[Path]) -> List[Path]:
    out = []
    for root in roots:
        if not root.exists():
            continue
        # Accept either the standard runs layout or any folder that contains summary.json files
        for p in root.rglob("summary.json"):
            out.append(p)
    # stable sort by mtime newest → oldest
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out

def _load_summary(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r") as f:
            j = json.load(f)
        j["_summary_path"] = str(p)
        j["_run_dir"] = str(p.parent)  # the run folder
        return j
    except Exception as e:
        console.print(f"[yellow]Warning:[/] failed to read {p}: {e}")
        return None

def _as_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    # pick key fields; tolerate missing
    rows = []
    for j in items:
        rows.append({
            "timestamp": j.get("timestamp"),
            "policy": j.get("policy"),
            "seed": j.get("seed"),
            "mode": j.get("mode"),
            "episodes_total": j.get("episodes_total", 0),
            "steps_cap_per_episode": j.get("steps_cap_per_episode", 0),
            "successes": j.get("successes", 0),
            "success_rate_pct": j.get("success_rate_pct", 0.0),
            "mean_reward": j.get("mean_reward", np.nan),
            "mean_steps": j.get("mean_steps", np.nan),
            "reward_trend": j.get("reward_trend", "n/a"),
            "success_trend": j.get("success_trend", "n/a"),
            "maturity": j.get("maturity", "n/a"),
            "auto_stop_target_pct": j.get("auto_stop_target_pct", None),
            "auto_stop_achieved": j.get("auto_stop_achieved", None),
            "run_dir": j.get("_run_dir", ""),
            "summary_path": j.get("_summary_path", ""),
        })
    df = pd.DataFrame(rows)
    # Useful stable sort: success desc, reward desc, steps asc, episodes desc
    df = df.sort_values(
        by=["success_rate_pct", "mean_reward", "episodes_total"],
        ascending=[False, False, False],
        kind="mergesort",
        ignore_index=True,
    )
    return df

def _print_leaderboard(df: pd.DataFrame, top: int = 20):
    if df.empty:
        console.print("[red]No summaries found.[/]")
        return
    n = min(top, len(df))
    view = df.head(n).copy()

    tbl = Table(title=f"Run Leaderboard (top {n}/{len(df)})", box=box.SIMPLE_HEAVY)
    for col, just in [
        ("#", "right"),
        ("timestamp", "left"),
        ("mode", "left"),
        ("policy", "left"),
        ("seed", "right"),
        ("episodes_total", "right"),
        ("success_rate_pct", "right"),
        ("mean_reward", "right"),
        ("mean_steps", "right"),
        ("maturity", "left"),
        ("success_trend", "left"),
        ("reward_trend", "left"),
        ("run_dir", "left"),
    ]:
        tbl.add_column(col, justify=just)

    for i, row in view.iterrows():
        tbl.add_row(
            str(i+1),
            str(row.get("timestamp", "")),
            str(row.get("mode", "")),
            str(row.get("policy", "")),
            str(row.get("seed", "")),
            str(row.get("episodes_total", "")),
            f"{float(row.get('success_rate_pct', 0.0)):.1f}",
            f"{float(row.get('mean_reward', 0.0)):.1f}",
            f"{float(row.get('mean_steps', 0.0)):.1f}",
            str(row.get("maturity", "")),
            str(row.get("success_trend", "")),
            str(row.get("reward_trend", "")),
            str(row.get("run_dir", "")),
        )
    console.print(tbl)

def _plot_compare(df: pd.DataFrame, out: Optional[Path] = None, top: int = 12):
    if df.empty:
        console.print("[yellow]Skip plot: no data.[/]")
        return
    view = df.head(min(top, len(df))).copy()
    # Build simple bar chart comparing success rate (and annotate mean reward)
    labels = [f"{i+1}" for i in range(len(view))]
    sr = view["success_rate_pct"].astype(float).values
    mr = view["mean_reward"].astype(float).values

    plt.figure()
    bars = plt.bar(labels, sr)
    # annotate with mean reward
    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()+1,
                 f"R≈{mr[i]:.0f}", ha="center", va="bottom", fontsize=9, rotation=0)
    plt.ylim(0, max(100, np.nanmax(sr)*1.1))
    plt.xlabel("Run rank (see console table)")
    plt.ylabel("Success rate (%)")
    plt.title("Top Runs — Success Rate (annotated with mean reward)")
    plt.tight_layout()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        console.print(f"[green]📈 Saved comparison plot:[/] {out}")
        plt.close()
    else:
        plt.show()

def _write_markdown(df: pd.DataFrame, out_md: Path, top: int = 20):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    n = min(top, len(df))
    view = df.head(n).copy()

    lines = []
    lines.append(f"# PedagoReLearn — Run Leaderboard (top {n}/{len(df)})\n")
    lines.append("| # | timestamp | mode | policy | seed | episodes | success% | mean reward | mean steps | maturity | succ trend | rew trend | run dir |")
    lines.append("|---:|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|:--|:--|")
    for i, row in view.iterrows():
        lines.append(
            "| {rank} | {ts} | {mode} | {pol} | {seed} | {eps} | {sr:.1f} | {mr:.1f} | {ms:.1f} | {mat} | {strend} | {rtrend} | {rd} |".format(
                rank=i+1,
                ts=row.get("timestamp",""),
                mode=row.get("mode",""),
                pol=row.get("policy",""),
                seed=int(row.get("seed",0)),
                eps=int(row.get("episodes_total",0)),
                sr=float(row.get("success_rate_pct",0.0)),
                mr=float(row.get("mean_reward",0.0)),
                ms=float(row.get("mean_steps",0.0)),
                mat=row.get("maturity",""),
                strend=row.get("success_trend",""),
                rtrend=row.get("reward_trend",""),
                rd=row.get("run_dir",""),
            )
        )
    out_md.write_text("\n".join(lines))
    console.print(f"[green]📝 Saved Markdown report:[/] {out_md}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Analyze PedagoReLearn runs from summary.json files.")
    ap.add_argument("--roots", default="runs", help="Comma-separated root folders to scan (default: runs)")
    ap.add_argument("--sort", default="success_rate_pct", help="Column to sort by (default: success_rate_pct)")
    ap.add_argument("--desc", action="store_true", help="Sort descending (default off)")
    ap.add_argument("--top", type=int, default=20, help="Rows to display/export (default: 20)")
    ap.add_argument("--out-csv", default="", help="Optional CSV path to export leaderboard")
    ap.add_argument("--out-md", default="", help="Optional Markdown path to export leaderboard")
    ap.add_argument("--plot", action="store_true", help="Show or save a comparison bar chart")
    ap.add_argument("--plot-out", default="", help="Optional PNG path; if empty and --plot set, shows interactively")
    args = ap.parse_args()

    roots = [Path(x.strip()) for x in args.roots.split(",") if x.strip()]
    summaries = [_load_summary(p) for p in _find_summaries(roots)]
    summaries = [s for s in summaries if s is not None]

    if not summaries:
        console.print("[red]No summary.json files found. Run the trainer with --group-run or JSON summary enabled.[/]")
        return

    df = _as_df(summaries)

    # Optional sort override
    if args.sort in df.columns:
        df = df.sort_values(by=args.sort, ascending=(not args.desc), kind="mergesort", ignore_index=True)
    else:
        console.print(f"[yellow]Note:[/] sort column '{args.sort}' not found; using default order.")

    _print_leaderboard(df, top=args.top)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.head(args.top).to_csv(out_csv, index=False)
        console.print(f"[green]💾 Saved CSV leaderboard:[/] {out_csv}")

    if args.out_md:
        _write_markdown(df, Path(args.out_md), top=args.top)

    if args.plot:
        _plot_compare(df, Path(args.plot_out) if args.plot_out else None, top=args.top)

if __name__ == "__main__":
    main()