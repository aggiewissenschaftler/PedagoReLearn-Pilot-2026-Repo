from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

from pedagorelearn_env import PedagoReLearnEnv
from agents.random_agent import RandomAgent
from agents.sarsa_agent import SarsaAgent


def run_episode(env: PedagoReLearnEnv, agent, seed: int) -> Tuple[int, float, str]:
    state, _ = env.reset(seed=seed)
    a = agent.act(state)
    total_r = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while True:
        s_next, r, done, info = env.step(a)
        last_info = info
        total_r += r
        steps += 1

        learner = getattr(agent, "learn", None)
        if callable(learner):
            a_next = agent.act(s_next)
            learner(state, a, r, s_next, a_next, done)
            a = a_next
        else:
            a = agent.act(s_next)

        state = s_next
        if done:
            break

    return steps, total_r, last_info.get("done_reason", "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rules_dir", type=str, default=str(Path(__file__).resolve().parent / "rules"))
    parser.add_argument("--culture", type=str, default="DE")
    parser.add_argument("--tag", type=str, default="baseline", help="Short label to tag this run (e.g., a, b, fewer_steps)")
    parser.add_argument("--no_summary", action="store_true")

    # SARSA hyperparams & decay
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--eps0", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.96)
    parser.add_argument("--min_eps", type=float, default=0.02)

    # Env reward knobs + terminal incentives
    parser.add_argument("--max_steps", type=int, default=35)
    parser.add_argument("--correct_bonus", type=float, default=10.0)
    parser.add_argument("--wrong_penalty", type=float, default=-2.0)
    parser.add_argument("--step_cost", type=float, default=-0.15)
    parser.add_argument("--mastery_bonus", type=float, default=3.0)
    parser.add_argument("--pragmatics_bonus", type=float, default=5.0)
    parser.add_argument("--end_bonus_mastery", type=float, default=20.0)
    parser.add_argument("--timeout_penalty", type=float, default=-10.0)
    args = parser.parse_args()

    # Paths
    trace_dir = Path("trace_results"); trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = args.tag.replace(" ", "_")
    fname = f"trace_results_{args.culture}_ep{args.episodes}_seeds{len(args.seeds)}_{tag}_{timestamp}"
    out_csv = trace_dir / f"{fname}.csv"
    out_params = trace_dir / f"{fname}.params.json"

    # Env + agents
    env = PedagoReLearnEnv(
        rules_dir=args.rules_dir, culture=args.culture, seed=0,
        max_steps=args.max_steps, verbose_rules=True,
        correct_bonus=args.correct_bonus, wrong_penalty=args.wrong_penalty,
        step_cost=args.step_cost, mastery_bonus=args.mastery_bonus,
        pragmatics_bonus=args.pragmatics_bonus,
        end_bonus_mastery=args.end_bonus_mastery,
        timeout_penalty=args.timeout_penalty,
    )
    random_agent = RandomAgent(seed=0)
    sarsa_agent = SarsaAgent(seed=0, alpha=args.alpha, gamma=args.gamma, eps=args.eps0)

    # Save run parameters for CI/repro
    with out_params.open("w") as jf:
        json.dump(vars(args), jf, indent=2)

    # Running means per (seed, algo)
    def init_stats(): return {"sum_steps": 0.0, "sum_reward": 0.0, "count": 0}
    stats: Dict[tuple, Dict[str, float]] = {}

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag","seed","algo","episode","is_final",
            "steps_to_mastery","total_reward","done_reason",
            "running_mean_steps","running_mean_reward"
        ])

        for seed in args.seeds:
            stats[("Random", seed)] = init_stats()
            stats[("SARSA",  seed)] = init_stats()

            # Random
            for ep in range(args.episodes):
                steps, reward, why = run_episode(env, random_agent, seed + ep*101)
                s = stats[("Random", seed)]
                s["sum_steps"]+=steps; s["sum_reward"]+=reward; s["count"]+=1
                writer.writerow([
                    tag, seed, "Random", ep, int(ep==args.episodes-1),
                    steps, round(reward,3), why,
                    round(s["sum_steps"]/s["count"],3), round(s["sum_reward"]/s["count"],3),
                ])

            # SARSA (decay)
            for ep in range(args.episodes):
                if hasattr(sarsa_agent, "decay"):
                    sarsa_agent.decay(factor=args.decay, min_eps=args.min_eps)
                steps, reward, why = run_episode(env, sarsa_agent, seed + ep*101 + 999)
                s = stats[("SARSA", seed)]
                s["sum_steps"]+=steps; s["sum_reward"]+=reward; s["count"]+=1
                writer.writerow([
                    tag, seed, "SARSA", ep, int(ep==args.episodes-1),
                    steps, round(reward,3), why,
                    round(s["sum_steps"]/s["count"],3), round(s["sum_reward"]/s["count"],3),
                ])

    print(f"✅ Wrote trace results to: {out_csv.resolve()}")
    print(f"📝 Params saved to:      {out_params.resolve()}")

    if not args.no_summary:
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        for algo in ("Random","SARSA"):
            sub = [r for r in rows if r["algo"]==algo]
            if not sub: continue
            steps = [int(r["steps_to_mastery"]) for r in sub]
            rews  = [float(r["total_reward"]) for r in sub]
            print(f"{algo}: mean steps={sum(steps)/len(steps):.1f}, mean reward={sum(rews)/len(rews):.1f}")


if __name__ == "__main__":
    main()