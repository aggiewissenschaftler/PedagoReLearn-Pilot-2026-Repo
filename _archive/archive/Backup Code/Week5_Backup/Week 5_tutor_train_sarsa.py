import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from pedagorelearn_env import PedagoReLearnEnv
from student_model_sarsa import SarsaAgent, encode_obs


def run_episode(env: PedagoReLearnEnv, agent: SarsaAgent, ep_seed: int):
    obs, _ = env.reset(seed=ep_seed)
    s = encode_obs(obs)
    a = agent.act(s)
    G, steps = 0.0, 0
    while True:
        obs2, r, term, trunc, _ = env.step(a)
        G += r; steps += 1
        s2 = encode_obs(obs2)
        a2 = agent.act(s2)
        agent.update(s, a, r, s2, a2)
        s, a = s2, a2
        if term or trunc:
            # return final obs so we can compute mastery metrics
            return G, steps, bool(term), obs2


def train_sarsa(
    num_episodes: int = 600,
    seed: int = 0,
    N: int = 3,
    log_csv_path: str | None = None,
    soft_mastery_threshold: float = 2.5,
    epsilon: float = 0.20,
    alpha: float = 0.5,
    gamma: float = 0.97,
    eps_decay_rate: float = 0.997,
    eps_floor: float = 0.05,
):
    env = PedagoReLearnEnv(N=N, seed=seed)
    agent = SarsaAgent(n_actions=env.action_space.n, epsilon=epsilon, alpha=alpha, gamma=gamma, seed=seed)

    returns, ep_steps, mastered_hard = [], [], []
    rows = []

    for ep in range(num_episodes):
        G, steps, success_hard, final_obs = run_episode(env, agent, ep_seed=ep + seed)
        returns.append(G); ep_steps.append(steps); mastered_hard.append(success_hard)

        m = final_obs["mastery"]
        avg_mastery = float(np.mean(m))
        frac_at_3 = float(np.mean(m == 3))
        success_soft = int(avg_mastery >= soft_mastery_threshold)

        rows.append({
            "episode": ep,
            "return": G,
            "steps": steps,
            "epsilon": agent.epsilon,
            "avg_mastery": avg_mastery,
            "frac_mastery3": frac_at_3,
            "mastered_hard": int(success_hard),
            "mastered_soft": success_soft,
        })

        agent.decay_epsilon(floor=eps_floor, rate=eps_decay_rate)

        if (ep + 1) % 50 == 0:
            avg = np.mean(returns[max(0, ep - 49):ep + 1])
            print(f"[Ep {ep+1:4d}] avg_return(last50)={avg:7.2f}  eps={agent.epsilon:0.3f}  mastered_hard%={100*np.mean(mastered_hard):.1f}")

    # Save CSV if requested
    if log_csv_path:
        p = Path(log_csv_path)
        if p.is_dir() or str(log_csv_path).endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            p = p / f"sarsa_train_log_{ts}.csv"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"[Saved episode log] {p}")

    return returns, ep_steps, mastered_hard


if __name__ == "__main__":
    r, s, m = train_sarsa(num_episodes=200, seed=0, N=3, log_csv_path="docs/results")
    print("avg_return:", np.mean(r), "mastered%:", 100*np.mean(m))