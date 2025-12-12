from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import random, inspect

from pedagorelearn_env import PedagoReLearnEnv
from agents.sarsa_agent import SarsaAgent

SEEDS    = list(range(10))
EPISODES = 500
MAX_STEPS= 1000
RESULTS  = Path("results"); RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS    = Path("plots");   PLOTS.mkdir(parents=True, exist_ok=True)

ALPHA=0.10; GAMMA=0.95; EPS_START=0.30; EPS_END=0.05; EPS_DECAY_EPIS=400

def set_seed(s:int):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def epsilon(ep:int)->float:
    frac = min(1.0, max(0.0, ep/float(EPS_DECAY_EPIS)))
    return EPS_START + (EPS_END - EPS_START) * frac

def obs_to_tuple(x):
    if isinstance(x, np.ndarray): return tuple(int(v) for v in x.ravel().tolist())
    if isinstance(x, (list,tuple)): return tuple(int(v) for v in x)
    try: return (int(x),)
    except Exception: return tuple(x)

def _n_states(env):
    sp = getattr(env, "observation_space", None)
    if sp is None: return None
    if hasattr(sp, "n"): return int(sp.n)             # Discrete
    nvec = getattr(sp, "nvec", None)                  # MultiDiscrete
    if nvec is not None:
        n = 1
        for v in nvec: n *= int(v)
        return int(n)
    return None

def create_agent(env):
    """
    Build SarsaAgent safely:
    1) Prefer keyword 'environment=env'
    2) Otherwise pass n_actions (and n_states when supported)
    3) Avoid passing the env positionally to prevent misbinding to n_actions
    """
    n_actions = getattr(getattr(env, "action_space", None), "n", None)
    n_states  = _n_states(env)

    # Inspect constructor
    try:
        params = list(inspect.signature(SarsaAgent.__init__).parameters.keys())
    except Exception:
        params = []

    # 1) If it supports environment=, use it
    if "environment" in params:
        for kw in (
            dict(environment=env, alpha=ALPHA, gamma=GAMMA,
                 epsilon_start=EPS_START, epsilon_end=EPS_END,
                 epsilon_decay_episodes=EPS_DECAY_EPIS),
            dict(environment=env, alpha=ALPHA, gamma=GAMMA),
            dict(environment=env),
        ):
            try:
                return SarsaAgent(**kw)
            except TypeError:
                pass

    # 2) Try explicit n_actions(/n_states) keyword patterns
    if n_actions is not None:
        for akey in ("n_actions","action_size","num_actions"):
            if akey in params:
                cand = {akey: n_actions}
                if "n_states" in params and n_states is not None:
                    cand["n_states"] = n_states
                for extra in (
                    dict(alpha=ALPHA, gamma=GAMMA, epsilon_start=EPS_START, epsilon_end=EPS_END),
                    dict(alpha=ALPHA, gamma=GAMMA),
                    dict(),
                ):
                    kw = cand | extra
                    try:
                        return SarsaAgent(**kw)
                    except TypeError:
                        pass

        # 2b) Positional patterns (only when we’re confident)
        try:
            if "n_states" in params and n_states is not None:
                return SarsaAgent(n_states, n_actions, ALPHA, GAMMA, EPS_START, EPS_END)
        except TypeError:
            pass
        for args in (
            (n_actions, ALPHA, GAMMA, EPS_START, EPS_END),
            (n_actions,)
        ):
            try:
                return SarsaAgent(*args)
            except TypeError:
                pass

    # 3) Last-chance: try passing env BUT only as a keyword not to clobber n_actions
    for kw in (dict(environment=env), dict()):
        try:
            return SarsaAgent(**kw)
        except TypeError:
            pass

    sig = "unknown"
    try: sig = str(inspect.signature(SarsaAgent.__init__))
    except Exception: pass
    raise RuntimeError(f"Couldn't construct SarsaAgent. __init__ signature: {sig}")

def resolve_action(agent):
    for name in ("select_action","choose_action","act","policy"):
        f = getattr(agent, name, None)
        if callable(f):
            # use epsilon if the method supports it
            try:
                if "epsilon" in inspect.signature(f).parameters:
                    return f
            except Exception:
                pass
            return lambda s, eps: f(s)
    raise RuntimeError("Need agent.select_action/choose_action/act/policy")

def resolve_update(agent):
    """
    Return (fn, needs_next_action)
    - SARSA style: fn(s,a,r,s2,a2, alpha,gamma) => True
    - Q-learn style: fn(s,a,r,s2, alpha,gamma)   => False
    """
    for name in ("update","learn","step","update_q","sarsa_update","q_update"):
        f = getattr(agent, name, None)
        if not callable(f): continue
        try:
            n = len(inspect.signature(f).parameters)
            if n >= 7: return f, True    # s,a,r,s2,a2,alpha,gamma
            if n >= 6: return f, False   # s,a,r,s2,alpha,gamma
        except Exception:
            pass
    f = getattr(agent, "learn", None)
    if callable(f):
        try:
            if len(inspect.signature(f).parameters) >= 5:
                return f, True
        except Exception:
            pass
    raise RuntimeError("Need agent.update/learn with SARSA or Q-learning signature")

def run_one(mode:str, seed:int, episodes:int=EPISODES, max_steps:int=MAX_STEPS):
    set_seed(seed)
    env = PedagoReLearnEnv(aggregation=(mode=="aggregated"))
    agent = create_agent(env)
    act = resolve_action(agent)
    upd, needs_a2 = resolve_update(agent)

    returns=[]; steps=[]; accs=[]
    for ep in range(episodes):
        eps = epsilon(ep)
        obs, _ = env.reset(seed=seed+ep)
        s = obs_to_tuple(obs)
        a = act(s, eps)
        tot=0.0; n=0; trials=0; corrects=0

        for _ in range(max_steps):
            # Gymnasium-style env: (obs, reward, terminated, truncated, info)
            obs2, r, terminated, truncated, info = env.step(a)
            s2 = obs_to_tuple(obs2)
            tot += float(r); n += 1
            if "correct" in info:
                trials += 1
                if info["correct"]: corrects += 1

            if terminated or truncated:
                try:
                    if needs_a2: upd(s, a, r, s2, a, ALPHA, GAMMA)
                    else:        upd(s, a, r, s2, ALPHA, GAMMA)
                except TypeError:
                    try:
                        if needs_a2: upd(s, a, r, s2, a)
                        else:        upd(s, a, r, s2)
                    except TypeError:
                        pass
                break

            a2 = act(s2, eps)
            try:
                if needs_a2: upd(s, a, r, s2, a2, ALPHA, GAMMA)
                else:        upd(s, a, r, s2, ALPHA, GAMMA)
            except TypeError:
                try:
                    if needs_a2: upd(s, a, r, s2, a2)
                    else:        upd(s, a, r, s2)
                except TypeError:
                    pass

            s, a = s2, a2

        returns.append(tot)
        steps.append(n)
        accs.append((corrects / trials) if trials > 0 else np.nan)

    # Save per-seed curves
    L = max(len(returns), len(steps), len(accs))
    r = np.array(returns,float); r = np.pad(r,(0,L-len(r)),constant_values=np.nan)
    st= np.array(steps,float);   st= np.pad(st,(0,L-len(st)),constant_values=np.nan)
    ac= np.array(accs,float);    ac= np.pad(ac,(0,L-len(ac)),constant_values=np.nan)
    pd.DataFrame({"episode":np.arange(1,L+1),
                  "reward":r,"steps":st,"acc":ac,
                  "mode":mode,"seed":seed}).to_csv(
        RESULTS/f"curves_{mode}_seed{seed}.csv", index=False)

    return float(np.nanmean(r)), float(np.nanmean(st)), float(np.nanmean(ac))

def main():
    rows=[]
    for mode in ("full","aggregated"):
        for seed in SEEDS:
            print(f"[Week9] mode={mode} seed={seed}")
            rm, sm, am = run_one(mode, seed)
            rows.append({"mode":mode,"seed":seed,
                         "reward_mean":rm,"steps_mean":sm,"acc_mean":am})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS/"aggregation_comparison.csv", index=False)

    summary = df.groupby("mode")[["reward_mean","steps_mean","acc_mean"]].agg(["mean","std"]).reset_index()
    summary.to_csv(RESULTS/"summary_week9.csv", index=False)

    # Plots
    def bar(metric, title, outpng):
        plt.figure(figsize=(7,5))
        means = df.groupby("mode")[metric].mean()
        stds  = df.groupby("mode")[metric].std()
        plt.bar(means.index, means.values, yerr=stds.values, capsize=6)
        plt.ylabel(metric); plt.title(title); plt.tight_layout()
        plt.savefig(PLOTS/outpng, dpi=150); plt.close()

    bar("reward_mean","Mean Cumulative Reward","week9_bar_reward_mean.png")
    bar("steps_mean","Mean Steps to Mastery (lower better)","week9_bar_steps_mean.png")
    bar("acc_mean","Mean Quiz Accuracy","week9_bar_acc_mean.png")

    def mean_curve(metric, label, outpng):
        frames=[]
        for mode in ("full","aggregated"):
            mode_paths=sorted(Path("results").glob(f"curves_{mode}_seed*.csv"))
            mode_frames=[pd.read_csv(p) for p in mode_paths if p.exists()]
            if not mode_frames: continue
            big=pd.concat(mode_frames, ignore_index=True)
            grp=big.groupby("episode")[metric].mean().reset_index()
            grp["mode"]=mode; frames.append(grp)
        if not frames: return
        allc=pd.concat(frames, ignore_index=True)
        plt.figure(figsize=(7,5))
        for mode in ("full","aggregated"):
            sub=allc[allc["mode"]==mode]
            if not sub.empty: plt.plot(sub["episode"], sub[metric], label=mode)
        plt.xlabel("Episode"); plt.ylabel(label); plt.title(f"{label} — mean across seeds")
        plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/outpng, dpi=150); plt.close()

    mean_curve("reward","Cumulative Reward","week9_curve_reward.png")
    mean_curve("steps","Steps per Episode","week9_curve_steps.png")
    mean_curve("acc","Quiz Accuracy","week9_curve_acc.png")

    print("[Week9] DONE → results/: aggregation_comparison.csv, summary_week9.csv; plots/: week9_*")

if __name__ == "__main__":
    main()