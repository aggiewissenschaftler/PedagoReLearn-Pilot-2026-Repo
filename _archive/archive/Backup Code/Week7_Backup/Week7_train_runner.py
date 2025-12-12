#!/usr/bin/env python3
"""
PedagoReLearn — Full Runner (training SR + eval SR + maturity + scheduler + auto-stop)

Features
- Policies: SARSA | Fixed | Random
- Env auto-load from pedagogorelearn_env_rewarded.py (class *Env or factory make_env/create_env/build_env)
- Per-episode logging, per-phase training summary (SR, mean steps, mean reward)
- Greedy evaluation after each phase (ε=0) with STRICT env-based success + optional reward fallback
- Maturity tracking: "Achieved" when eval_sr ≥ 80, eval_reward ≥ 800, mean_steps ≤ 30 (defaults)
- Sustained maturity stopping (e.g., 2 consecutive phases) to avoid spikes
- Auto-stop by training SR OR maturity (whichever occurs first), or use fixed --schedule
- CSV logs, checkpoints, trend plots, JSON summary, rich-rendered tables
- SARSA hyperparam overrides from CLI

Usage examples
  Auto-stop to SR or maturity (whichever hits first):
    python train_runner.py --policy sarsa --train --quiet-steps \
      --phase-episodes 200 --max-episodes 10000 \
      --until-success-rate 80 --until-maturity --sustain-maturity 2 --group-run

  Fixed schedule with permissive eval fallback threshold:
    python train_runner.py --policy sarsa --train --quiet-steps \
      --schedule 100,200,500 --eval-not-strict --eval-success-reward 700
"""

from __future__ import annotations
import argparse, csv, importlib, inspect, json, os, pathlib, pickle, random, sys, time
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

# --------------------------------------------------------------------------------------
# Console + path
# --------------------------------------------------------------------------------------
console = Console()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------
def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _moving_avg(x, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def _trend_label(series, w=5) -> str:
    if len(series) < max(2*w, 10):
        return "insufficient-data"
    ma = _moving_avg(series, w)
    early = float(np.mean(ma[:w]))
    late  = float(np.mean(ma[-w:]))
    slope = float(np.polyfit(np.arange(len(ma[-w:])), ma[-w:], 1)[0]) if len(ma) >= w else 0.0
    if late > early * 1.02 or slope > 0.05:
        return "improving"
    if late < early * 0.98 or slope < -0.05:
        return "regressing"
    return "plateauing"

def _maturity_label(eval_sr_pct: float, eval_mean_reward: float, train_mean_steps: float) -> str:
    if eval_sr_pct >= 80.0 and eval_mean_reward >= 800.0 and train_mean_steps <= 30.0:
        return "Achieved"
    if eval_sr_pct >= 60.0 and eval_mean_reward >= 700.0 and train_mean_steps <= 35.0:
        return "Developing"
    if eval_sr_pct >= 30.0 and eval_mean_reward >= 600.0:
        return "Early learning"
    return "Struggling"

# --------------------------------------------------------------------------------------
# Environment auto-loader
# --------------------------------------------------------------------------------------
def _load_env_ctor() -> Tuple[Callable[..., Any], bool]:
    mod = importlib.import_module("pedagorelearn_env_rewarded")
    for cand in ("make_env", "create_env", "build_env"):
        fn = getattr(mod, cand, None)
        if callable(fn):
            return fn, True
    # Look for classes defined in that module with 'Env' in name
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if ("Env" in name or "Environment" in name) and obj.__module__ == mod.__name__:
            return obj, False
    raise ImportError("No environment class/factory found in pedagorelearn_env_rewarded.py")

_EnvCtor, _IS_FACTORY = _load_env_ctor()
def make_env_instance():
    return _EnvCtor() if _IS_FACTORY else _EnvCtor()

# --------------------------------------------------------------------------------------
# Agents / Baselines
# --------------------------------------------------------------------------------------
def _import_agent_and_baselines():
    Agent = RandomPolicy = FixedCurriculumPolicy = None
    try:
        mb_agent = importlib.import_module("tutor_train_sarsa_rewarded")
        Agent = getattr(mb_agent, "SarsaAgent", None)
    except Exception as e:
        console.print(f"[yellow]Note:[/] Could not import SarsaAgent: {e}")
    try:
        mb = importlib.import_module("tutor_baselines")
        console.print(f"[cyan]tutor_baselines loaded from:[/] {getattr(mb, '__file__', '?')}")
        RandomPolicy = getattr(mb, "RandomPolicy", None)
        FixedCurriculumPolicy = getattr(mb, "FixedCurriculumPolicy", None)
        if FixedCurriculumPolicy is None:
            # discover Fixed*Policy if name differs slightly
            for name, obj in inspect.getmembers(mb, inspect.isclass):
                if name.lower().startswith("fixed") and "policy" in name.lower():
                    FixedCurriculumPolicy = obj
                    console.print(f"[green]Discovered fixed baseline class:[/] {name}")
                    break
    except Exception as e:
        console.print(f"[yellow]Note:[/] Could not import tutor_baselines: {e}")
    return Agent, RandomPolicy, FixedCurriculumPolicy

SarsaAgent, RandomPolicy, FixedCurriculumPolicy = _import_agent_and_baselines()

def make_policy(name: str, env, sarsa_kwargs=None):
    name = name.lower()
    if name == "random":
        if RandomPolicy is None:
            class _Rand:
                def __init__(self, action_space): self.action_space = action_space
                def select_action(self, _): return self.action_space.sample()
            return _Rand(env.action_space)
        return RandomPolicy(env.action_space)
    if name == "fixed":
        if FixedCurriculumPolicy is None:
            raise ImportError("FixedCurriculumPolicy not found in tutor_baselines.")
        return FixedCurriculumPolicy(env.action_space)
    if name == "sarsa":
        if SarsaAgent is None:
            raise ImportError("SarsaAgent not found in tutor_train_sarsa_rewarded.")
        n_actions = getattr(env.action_space, "n", None)
        if n_actions is None:
            raise TypeError("SarsaAgent requires Discrete action space with `.n`.")
        sarsa_kwargs = sarsa_kwargs or {}
        try:
            return SarsaAgent(int(n_actions), **sarsa_kwargs)
        except TypeError:
            return SarsaAgent(action_space_size=int(n_actions), **sarsa_kwargs)
    raise ValueError("Use policy: random | fixed | sarsa")

# --------------------------------------------------------------------------------------
# Env normalization
# --------------------------------------------------------------------------------------
def _reset(env):
    out = env.reset()
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})

def _step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    raise RuntimeError("Unexpected env.step output")

# --------------------------------------------------------------------------------------
# Learning helpers & success detection
# --------------------------------------------------------------------------------------
def maybe_learn_sarsa(agent, s, a, r, s_next, a_next, done):
    for method in ("update", "learn", "observe"):
        if hasattr(agent, method):
            try: getattr(agent, method)(s, a, r, s_next, a_next, done); return
            except TypeError:
                try: getattr(agent, method)(s, a, r, s_next, a_next); return
                except Exception: pass

def _strict_env_success(info, env) -> bool:
    """Unified STRICT success check used by both training and eval."""
    try:
        if isinstance(info, dict) and info.get("all_mastered", False):
            return True
        if isinstance(info, dict):
            rs = info.get("r_sums", {})
            if isinstance(rs, dict) and float(rs.get("terminal", 0.0)) > 0.0:
                return True
        if hasattr(env, "all_mastered") and bool(getattr(env, "all_mastered")):
            return True
    except Exception:
        pass
    return False

# --------------------------------------------------------------------------------------
# Single episode (training)
# --------------------------------------------------------------------------------------
def run_episode(env, policy_name: str, policy, max_steps: int, train: bool, quiet_steps: bool):
    state, _ = _reset(env)
    total_reward = 0.0
    steps = 0
    success = False
    success_step: Optional[int] = None
    info: Dict[str, Any] = {}
    t = -1

    # Episode start hook
    if hasattr(policy, "start_episode"):
        try: policy.start_episode(state)
        except TypeError: pass

    if policy_name == "sarsa" and train and hasattr(policy, "select_action"):
        action = policy.select_action(state)
        for t in range(max_steps):
            next_state, reward, done, info = _step(env, action)
            total_reward += reward
            next_action = policy.select_action(next_state)

            # STRICT success (env-native)
            if _strict_env_success(info, env) and not success:
                success = True
                success_step = t

            if not quiet_steps:
                console.print(f"[dim]t={t:02d}[/] | a={action} | r={reward:.3f}{' | ✅' if success and success_step == t else ''}")

            maybe_learn_sarsa(policy, state, action, reward, next_state, next_action, done)
            steps += 1
            state, action = next_state, next_action
            if done:
                # final strict check
                success = success or _strict_env_success(info, env)
                break
    else:
        for t in range(max_steps):
            action = policy.select_action(state) if hasattr(policy, "select_action") else \
                     (policy.act(state) if hasattr(policy, "act") else env.action_space.sample())
            next_state, reward, done, info = _step(env, action)
            total_reward += reward
            if _strict_env_success(info, env) and not success:
                success = True
                success_step = t
            if not quiet_steps:
                console.print(f"[dim]t={t:02d}[/] | a={action} | r={reward:.3f}{' | ✅' if success and success_step == t else ''}")
            steps += 1
            state = next_state
            if done:
                success = success or _strict_env_success(info, env)
                break

    # Episode end hook
    if hasattr(policy, "end_episode"):
        try: policy.end_episode()
        except TypeError: pass

    return steps, total_reward, success, success_step, info

def _print_episode_summary(ep_idx: int, st: int, R: float, ok: bool, dt: float):
    tbl = Table(title=f"Episode {ep_idx:02d} Summary", box=box.SIMPLE_HEAVY)
    for c in ("steps","reward","success","time (s)"): tbl.add_column(c, justify="right")
    tbl.add_row(str(st), f"{R:.3f}", "✅" if ok else "—", f"{dt:.2f}")
    console.print(tbl)

# --------------------------------------------------------------------------------------
# Greedy evaluation (ε=0) with layered success
# --------------------------------------------------------------------------------------
def evaluate_policy(env_ctor,
                    policy_obj,
                    steps_per_ep=40,
                    episodes=50,
                    strict_eval=True,
                    reward_success_threshold=700.0):
    env = env_ctor() if callable(env_ctor) else env_ctor

    # Force greedy mode
    if hasattr(policy_obj, "set_eval_mode"):
        try: policy_obj.set_eval_mode(True)
        except Exception: pass
    if hasattr(policy_obj, "epsilon"):
        try: policy_obj.epsilon = 0.0
        except Exception: pass

    successes, rewards = 0, []
    saw_info_keys = False

    for _ in range(episodes):
        obs, _ = _reset(env)
        total = 0.0
        info: Dict[str, Any] = {}
        for t in range(steps_per_ep):
            a = policy_obj.select_action(obs) if hasattr(policy_obj, "select_action") else env.action_space.sample()
            obs, r, done, info = _step(env, a)
            total += r
            if done:
                # STRICT: env flags; FALLBACK: reward spike + ended early
                ok = _strict_env_success(info, env) or (not strict_eval and (t + 1 < steps_per_ep and total >= reward_success_threshold))
                if ok: successes += 1
                break
        if isinstance(info, dict) and ("all_mastered" in info or "r_sums" in info):
            saw_info_keys = True
        rewards.append(total)

    if strict_eval and not saw_info_keys:
        console.print("[yellow]Eval warning:[/] env did not set info['all_mastered'] or info['r_sums']; "
                      "strict eval may undercount successes. Use --eval-not-strict or add those fields in env.")

    eval_sr = 100.0 * successes / max(1, episodes)
    eval_reward = float(np.mean(rewards)) if rewards else 0.0
    return eval_sr, eval_reward

# --------------------------------------------------------------------------------------
# Phase training
# --------------------------------------------------------------------------------------
def _save_checkpoint(policy, path: pathlib.Path) -> bool:
    _ensure_dir(path.parent)
    # If agent has custom save, use it
    if hasattr(policy, "save"):
        try:
            policy.save(str(path)); return True
        except Exception:
            pass
    # Otherwise try to pickle Q or the object
    try:
        with open(path, "wb") as f:
            pickle.dump(getattr(policy, "Q", policy), f)
        return True
    except Exception:
        return False

def _save_phase_plot(rew, succ, path: pathlib.Path, w=25, title="Training Trend"):
    _ensure_dir(path.parent)
    plt.figure()
    plt.plot(_moving_avg(rew, w), label="Reward (MA)")
    ax2 = plt.gca().twinx()
    ax2.plot(_moving_avg(succ, w), '--', label="Success (MA)")
    plt.title(title); plt.xlabel("Episode"); plt.tight_layout(); plt.savefig(path); plt.close()

def train_phase(env, policy, args, phase_eps, ep_offset, log_path: pathlib.Path, quiet_steps: bool):
    rewards: List[float] = []
    steps_list: List[int] = []
    successes: List[int] = []

    n_succ = 0
    first_succ: Optional[int] = None
    total_time = 0.0

    for ep in range(phase_eps):
        t0 = time.time()
        st, R, ok, ok_step, info = run_episode(env, args.policy, policy, args.steps, args.train, quiet_steps)
        dt = time.time() - t0
        total_time += dt

        rewards.append(R)
        steps_list.append(st)
        successes.append(1 if ok else 0)
        if ok:
            n_succ += 1
            if first_succ is None:
                first_succ = ep_offset + ep

        _print_episode_summary(ep_offset + ep, st, R, ok, dt)

        # CSV log
        _ensure_dir(log_path.parent)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["policy","seed","episode","phase","steps","reward","success","time_s"])
            w.writerow([args.policy, args.seed, ep_offset + ep, "", st, f"{R:.6f}", int(ok), f"{dt:.3f}"])

        # Optional epsilon decay if agent supports it
        if hasattr(policy, "decay_epsilon") and args.policy == "sarsa" and args.train:
            try: policy.decay_epsilon()
            except TypeError: pass

    # Training stats
    sr_train = 100.0 * n_succ / max(1, phase_eps)
    mr_train = float(np.mean(rewards)) if rewards else 0.0
    ms_train = float(np.mean(steps_list)) if steps_list else 0.0

    # Pretty phase training table
    phase_tbl = Table(title=f"Phase Summary (training) — {phase_eps} eps", box=box.SIMPLE_HEAVY)
    for c in ("episodes","successes","success_rate(%)","first_success_ep","mean_steps","mean_reward","total_time(s)"):
        phase_tbl.add_column(c, justify="right")
    phase_tbl.add_row(str(phase_eps), str(n_succ), f"{sr_train:.1f}", str(first_succ or "—"),
                      f"{ms_train:.2f}", f"{mr_train:.3f}", f"{total_time:.2f}")
    console.print(phase_tbl)

    # Greedy evaluation (ε=0)
    eval_sr, eval_reward = evaluate_policy(
        make_env_instance, policy,
        steps_per_ep=args.steps,
        episodes=max(20, args.eval_episodes),
        strict_eval=not args.eval_not_strict,
        reward_success_threshold=args.eval_success_reward,
    )
    console.print(f"[bold]Eval (greedy) after phase[/]: success={eval_sr:.1f}%  reward≈{eval_reward:.1f}\n")

    # Learning progress report (train series only)
    trend_r = _trend_label(rewards, w=max(5, args.trend_window))
    trend_s = _trend_label(successes, w=max(5, args.trend_window))
    maturity = _maturity_label(eval_sr, eval_reward, ms_train)
    console.print(f"[bold]Learning Progress Report[/]")
    console.print(f"• Reward trend  (train): [blue]{trend_r}[/]")
    console.print(f"• Success trend (train): [red]{trend_s}[/]")
    console.print(f"• Maturity (eval-based): [green]{maturity}[/]\n")

    return rewards, successes, steps_list, sr_train, eval_sr, eval_reward, ms_train, maturity

# --------------------------------------------------------------------------------------
# JSON summary
# --------------------------------------------------------------------------------------
def write_run_summary(
    out_path: pathlib.Path,
    policy: str,
    seed: int,
    steps_per_ep: int,
    episodes_total: int,
    rewards_all: List[float],
    successes_all: List[int],
    steps_all: List[int],
    phases: List[Dict],
    trend_window: int,
    mode: str,
    auto_target: Optional[float] = None,
    target_achieved: Optional[bool] = None,
    episodes_at_target: Optional[int] = None,
):
    successes = int(np.sum(successes_all)) if successes_all else 0
    sr_overall = 100.0 * successes / max(1, episodes_total)
    mean_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
    mean_steps = float(np.mean(steps_all)) if steps_all else 0.0

    # Prefer last eval metrics for final maturity label
    last_eval_sr = phases[-1]["eval_success_rate_pct"] if phases else None
    last_eval_reward = phases[-1]["eval_mean_reward"] if phases else None
    maturity_eval = _maturity_label(
        float(last_eval_sr if last_eval_sr is not None else sr_overall),
        float(last_eval_reward if last_eval_reward is not None else mean_reward),
        mean_steps,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "policy": policy,
        "seed": seed,
        "mode": mode,  # auto-stop | schedule | single
        "episodes_total": episodes_total,
        "steps_cap_per_episode": steps_per_ep,
        "successes": successes,
        "success_rate_pct": round(sr_overall, 2),
        "mean_reward": round(mean_reward, 3),
        "mean_steps": round(mean_steps, 3),
        "reward_trend": _trend_label(rewards_all, w=max(5, trend_window)) if rewards_all else "n/a",
        "success_trend": _trend_label(successes_all, w=max(5, trend_window)) if successes_all else "n/a",
        "maturity_eval_based": maturity_eval,
        "auto_stop_target_pct": auto_target,
        "auto_stop_achieved": target_achieved,
        "episodes_at_target": episodes_at_target,
        "phases": phases,
    }

    _ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]🧾 Saved run summary:[/] {out_path}")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PedagoReLearn runner (SR + eval + maturity + scheduler + auto-stop)")
    # Core
    p.add_argument("--policy", default="sarsa", choices=["random","fixed","sarsa"])
    p.add_argument("--episodes", type=int, default=100, help="Used in single-run mode")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", action="store_true")
    # Output
    p.add_argument("--log", default="trace_results/run_log.csv")
    p.add_argument("--log-stamp", action="store_true")
    p.add_argument("--group-run", action="store_true",
                   help="Group outputs under runs/run_<timestamp>/{logs,checkpoints,figs}")
    # Scheduler
    p.add_argument("--schedule", default="", help='Comma list of episode counts (e.g. "100,200,500")')
    # Auto-stop by training SR
    p.add_argument("--until-success-rate", type=float, default=0.0,
                   help="If >0, stop when last phase's TRAINING success rate ≥ this percent")
    p.add_argument("--phase-episodes", type=int, default=200)
    p.add_argument("--max-episodes", type=int, default=10000)
    # Auto-stop by maturity (eval-based)
    p.add_argument("--until-maturity", action="store_true",
                   help="Stop when eval-based maturity thresholds are met")
    p.add_argument("--maturity-thresholds", default="80,800,30",
                   help="Eval success %, eval mean reward, train mean steps; e.g. '80,800,30'")
    p.add_argument("--sustain-maturity", type=int, default=2,
                   help="Require maturity to be achieved for N consecutive phases to stop")
    # Eval controls
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-not-strict", action="store_true",
                   help="Enable reward fallback for eval success")
    p.add_argument("--eval-success-reward", type=float, default=700.0,
                   help="Eval fallback threshold (episode must finish early AND total reward ≥ this)")
    # SARSA overrides
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--alpha",   type=float, default=None)
    p.add_argument("--gamma",   type=float, default=None)
    p.add_argument("--epsilon-min",   type=float, default=None)
    p.add_argument("--epsilon-decay", type=float, default=None)
    # Misc
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--figs", default="trace_results/figs")
    p.add_argument("--quiet-steps", action="store_true")
    p.add_argument("--trend-window", type=int, default=5)
    args = p.parse_args()

    # Parse maturity thresholds
    m_sr, m_reward, m_steps = 80.0, 800.0, 30.0
    try:
        parts = [x.strip() for x in args.maturity_thresholds.split(",")]
        if len(parts) == 3:
            m_sr, m_reward, m_steps = float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass

    random.seed(args.seed)

    console.print("[bold]🔧 Spinning up environment…[/]")
    env = make_env_instance()
    try: env.reset(seed=args.seed)
    except TypeError: pass

    console.print(f"[bold]🧠 Building policy:[/] {args.policy}")
    policy = make_policy(args.policy, env)

    # Optional SARSA hyperparam overrides
    if args.policy == "sarsa":
        if args.epsilon is not None and hasattr(policy, "epsilon"):
            policy.epsilon = float(args.epsilon)
        if args.alpha is not None and hasattr(policy, "alpha"):
            policy.alpha = float(args.alpha)
        if args.gamma is not None and hasattr(policy, "gamma"):
            policy.gamma = float(args.gamma)
        if hasattr(policy, "epsilon_min") and args.epsilon_min is not None:
            policy.epsilon_min = float(args.epsilon_min)
        if hasattr(policy, "epsilon_decay") and args.epsilon_decay is not None:
            policy.epsilon_decay = float(args.epsilon_decay)

    # Output destinations
    stamp = _stamp()
    if args.group_run:
        run_root = pathlib.Path("runs") / f"run_{stamp}"
        logs_dir = _ensure_dir(run_root / "logs")
        ckpt_dir = _ensure_dir(run_root / "checkpoints")
        figs_dir = _ensure_dir(run_root / "figs")
        log_path = logs_dir / "run_log.csv"
        summary_path = run_root / "summary.json"
    else:
        base_log_path = pathlib.Path(args.log)
        log_path = base_log_path.with_name(f"{base_log_path.stem}_{stamp}.csv") if args.log_stamp else base_log_path
        ckpt_dir = _ensure_dir(pathlib.Path(args.checkpoints))
        figs_dir = _ensure_dir(pathlib.Path(args.figs))
        summary_path = log_path.with_name(f"{log_path.stem}_summary.json")
    _ensure_dir(log_path.parent)

    auto_stop = args.until_success_rate and args.until_success_rate > 0.0
    use_schedule = bool(args.schedule.strip())

    all_rewards: List[float] = []
    all_success: List[int] = []
    all_steps: List[int] = []
    phase_summaries: List[Dict] = []

    # -------------------------------- Auto-stop mode --------------------------------
    if auto_stop or args.until_maturity:
        target_sr = float(args.until_success_rate) if auto_stop else 0.0
        goal_bits = []
        if auto_stop: goal_bits.append(f"train SR ≥ {target_sr:.1f}%")
        if args.until_maturity: goal_bits.append(f"maturity (eval_sr≥{m_sr}%, eval_reward≥{m_reward}, mean_steps≤{m_steps}) x{args.sustain_maturity}")
        console.rule(f"[bold]Auto-stop Mode[/] — target: " + " OR ".join(goal_bits))

        total_eps = 0
        phase_idx = 0
        achieved = False
        episodes_at_target = None
        maturity_streak = 0

        while total_eps < args.max_episodes:
            phase_idx += 1
            n = min(args.phase_episodes, args.max_episodes - total_eps)
            console.rule(f"[bold]Phase {phase_idx} — {n} episodes (total so far: {total_eps})[/]")
            R, S, ST, sr_train, eval_sr, eval_reward, ms_train, maturity = train_phase(
                env, policy, args, n, total_eps, log_path, args.quiet_steps
            )

            all_rewards += R; all_success += S; all_steps += ST
            total_eps += n

            # Save checkpoint + plot
            ckpt_path = ckpt_dir / f"{args.policy}_ep{total_eps}.pkl"
            _save_checkpoint(policy, ckpt_path)
            fig_path = figs_dir / f"trend_phase{phase_idx}_ep{n}.png"
            _save_phase_plot(R, S, fig_path, w=max(5, args.trend_window), title=f"Phase {phase_idx} ({n} eps)")
            console.print(f"[green]💾 Checkpoint:[/] {ckpt_path}")
            console.print(f"[green]📈 Chart:[/] {fig_path}\n")

            phase_summaries.append({
                "phase_index": phase_idx,
                "episodes": n,
                "success_rate_pct": round(sr_train, 2),      # training SR
                "mean_reward": round(float(np.mean(R)), 3) if R else 0.0,
                "mean_steps": round(float(np.mean(ST)), 3) if ST else 0.0,
                "eval_success_rate_pct": round(eval_sr, 2),  # eval SR
                "eval_mean_reward": round(eval_reward, 1),   # eval reward
                "checkpoint": str(ckpt_path),
                "trend_fig": str(fig_path),
            })

            # Stopping conditions:
            met_sr = auto_stop and (sr_train >= target_sr)
            if args.until_maturity:
                if maturity == "Achieved" and eval_sr >= m_sr and eval_reward >= m_reward and ms_train <= m_steps:
                    maturity_streak += 1
                else:
                    maturity_streak = 0
            met_maturity = args.until_maturity and (maturity_streak >= max(1, args.sustain_maturity))

            if met_sr or met_maturity:
                achieved = True
                episodes_at_target = total_eps
                label = "🎯 Train SR target met" if met_sr else f"🏁 Maturity sustained x{args.sustain_maturity}"
                console.rule(f"[bold green]{label}[/]")
                break

        console.rule(f"[bold green]✅ Auto-stop complete[/] (episodes run: {total_eps})")

        # Final Run Summary table
        successes_total = int(np.sum(all_success)) if all_success else 0
        sr_overall = 100.0 * successes_total / max(1, len(all_success))
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        mean_steps = float(np.mean(all_steps)) if all_steps else 0.0
        run_tbl = Table(title="Run Summary", box=box.SIMPLE_HEAVY)
        for c in ("policy","seed","episodes","successes","success_rate(%)","mean_steps","mean_reward"):
            run_tbl.add_column(c, justify="right")
        run_tbl.add_row(args.policy, str(args.seed), str(len(all_success)), str(successes_total),
                        f"{sr_overall:.1f}", f"{mean_steps:.2f}", f"{mean_reward:.3f}")
        console.print(run_tbl)

        write_run_summary(
            summary_path, args.policy, args.seed, args.steps, len(all_success),
            all_rewards, all_success, all_steps, phase_summaries, args.trend_window,
            mode="auto-stop", auto_target=target_sr if auto_stop else None,
            target_achieved=achieved, episodes_at_target=episodes_at_target
        )

    # -------------------------------- Scheduler mode --------------------------------
    elif use_schedule:
        phases = [int(x) for x in args.schedule.split(",") if x.strip()]
        offset = 0
        for i, n in enumerate(phases, start=1):
            console.rule(f"[bold]Phase {i}/{len(phases)} — {n} episodes[/]")
            R, S, ST, sr_train, eval_sr, eval_reward, ms_train, maturity = train_phase(
                env, policy, args, n, offset, log_path, args.quiet_steps
            )
            all_rewards += R; all_success += S; all_steps += ST
            offset += n

            ckpt_path = ckpt_dir / f"{args.policy}_ep{offset}.pkl"
            _save_checkpoint(policy, ckpt_path)
            fig_path = figs_dir / f"trend_phase{i}_ep{n}.png"
            _save_phase_plot(R, S, fig_path, w=max(5, args.trend_window), title=f"Phase {i} ({n} eps)")
            console.print(f"[green]💾 Checkpoint:[/] {ckpt_path}")
            console.print(f"[green]📈 Chart:[/] {fig_path}\n")

            phase_summaries.append({
                "phase_index": i,
                "episodes": n,
                "success_rate_pct": round(sr_train, 2),
                "mean_reward": round(float(np.mean(R)), 3) if R else 0.0,
                "mean_steps": round(float(np.mean(ST)), 3) if ST else 0.0,
                "eval_success_rate_pct": round(eval_sr, 2),
                "eval_mean_reward": round(eval_reward, 1),
                "checkpoint": str(ckpt_path),
                "trend_fig": str(fig_path),
            })

        total_eps = sum(phases)
        console.rule("[bold green]✅ Scheduler complete[/]")

        # Final Run Summary table
        successes_total = int(np.sum(all_success)) if all_success else 0
        sr_overall = 100.0 * successes_total / max(1, len(all_success))
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        mean_steps = float(np.mean(all_steps)) if all_steps else 0.0
        run_tbl = Table(title="Run Summary", box=box.SIMPLE_HEAVY)
        for c in ("policy","seed","episodes","successes","success_rate(%)","mean_steps","mean_reward"):
            run_tbl.add_column(c, justify="right")
        run_tbl.add_row(args.policy, str(args.seed), str(len(all_success)), str(successes_total),
                        f"{sr_overall:.1f}", f"{mean_steps:.2f}", f"{mean_reward:.3f}")
        console.print(run_tbl)

        write_run_summary(
            summary_path, args.policy, args.seed, args.steps, total_eps,
            all_rewards, all_success, all_steps, phase_summaries, args.trend_window,
            mode="schedule"
        )

    # -------------------------------- Single-run mode --------------------------------
    else:
        console.rule(f"[bold]Single Run[/] — {args.episodes} episodes")
        R, S, ST, sr_train, eval_sr, eval_reward, ms_train, maturity = train_phase(
            env, policy, args, args.episodes, 0, log_path, args.quiet_steps
        )
        all_rewards += R; all_success += S; all_steps += ST
        total_eps = args.episodes

        ckpt_path = ckpt_dir / f"{args.policy}_ep{total_eps}.pkl"
        _save_checkpoint(policy, ckpt_path)
        fig_path = figs_dir / f"trend_phase1_ep{args.episodes}.png"
        _save_phase_plot(R, S, fig_path, w=max(5, args.trend_window), title=f"Phase 1 ({args.episodes} eps)")
        console.print(f"[green]💾 Checkpoint:[/] {ckpt_path}")
        console.print(f"[green]📈 Chart:[/] {fig_path}\n")

        # Final Run Summary table
        successes_total = int(np.sum(all_success)) if all_success else 0
        sr_overall = 100.0 * successes_total / max(1, len(all_success))
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        mean_steps = float(np.mean(all_steps)) if all_steps else 0.0
        run_tbl = Table(title="Run Summary", box=box.SIMPLE_HEAVY)
        for c in ("policy","seed","episodes","successes","success_rate(%)","mean_steps","mean_reward"):
            run_tbl.add_column(c, justify="right")
        run_tbl.add_row(args.policy, str(args.seed), str(len(all_success)), str(successes_total),
                        f"{sr_overall:.1f}", f"{mean_steps:.2f}", f"{mean_reward:.3f}")
        console.print(run_tbl)

        write_run_summary(
            summary_path, args.policy, args.seed, args.steps, total_eps,
            all_rewards, all_success, all_steps, phase_summaries, args.trend_window,
            mode="single"
        )

if __name__ == "__main__":
    main()