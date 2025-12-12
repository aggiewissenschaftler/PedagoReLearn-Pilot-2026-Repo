# tutor_train_sarsa_rewarded.py
# Week 6 trainer for reward-shaped environment (fixed n_actions + epsilon decay)
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

from archive.pedagorelearn_env_rewarded import PedagoReLearnEnvRewarded as PedagoReLearnEnv
from student_model_sarsa import SarsaAgent, encode_obs


def run_episode(env, agent, ep_seed: int):
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
            return G, steps, bool(term), obs2


def train_sarsa_rewarded(
    num_episodes=600,
    seed=0,
    N=3,
    alpha=0.5,
    gamma=0.97,
    epsilon=0.20,
    epsilon_decay=0.997,
    min_epsilon=0.05,
    log_csv_path="trace_results/logs/",
):
    np.random.seed(seed)
    env = PedagoReLearnEnv(N=N, seed=seed)

    # ✅ FIX: pass action space size, not env, to SarsaAgent
    agent = SarsaAgent(
        n_actions=env.action_space.n,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        seed=seed,
    )

    records = []
    for ep in range(num_episodes):
        G, steps, done, last_obs = run_episode(env, agent, ep_seed=ep + seed)

        # simple mastery summaries for the log
        mastered_soft = int(float(np.mean(last_obs["mastery"])) >= 2.5)
        mastered_hard = int(np.all(last_obs["mastery"] == 3))
        avg_mastery = float(np.mean(last_obs["mastery"]))

        # ✅ FIX: decay_epsilon expects (floor, rate)
        agent.decay_epsilon(floor=min_epsilon, rate=epsilon_decay)

        records.append({
            "episode": ep,
            "return": G,
            "steps": steps,
            "avg_mastery": avg_mastery,
            "mastered_soft": mastered_soft,
            "mastered_hard": mastered_hard,
            "epsilon": agent.epsilon,
        })

        if (ep + 1) % 50 == 0:
            print(f"[Ep {ep+1:4d}] avg_return(last1)={G:7.2f}  eps={agent.epsilon:.3f}  avgM={avg_mastery:.2f}")

    # save log (timestamped)
    log_dir = Path(log_csv_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = log_dir / f"sarsa_rewarded_train_log_{ts}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[Saved SARSA(Rewarded) log] {out_path}")

    env.close()
    return out_path


if __name__ == "__main__":
    train_sarsa_rewarded()