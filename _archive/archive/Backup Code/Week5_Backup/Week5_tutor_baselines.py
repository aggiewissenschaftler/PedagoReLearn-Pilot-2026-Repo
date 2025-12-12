import numpy as np
from typing import List, Tuple
from pedagorelearn_env import PedagoReLearnEnv


def action_id(verb: str, idx: int, N: int) -> int:
    assert verb in {"teach", "quiz", "review"}
    base = {"teach": 0, "quiz": 1, "review": 2}[verb]
    return idx * 3 + base  # noop is 3N


def run_random_policy(num_episodes: int = 200, seed: int = 123, N: int = 3) -> Tuple[List[float], List[int], List[bool]]:
    env = PedagoReLearnEnv(N=N, seed=seed)
    returns, ep_steps, mastered = [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + seed)
        G, steps = 0.0, 0
        while True:
            a = env.action_space.sample()
            obs2, r, term, trunc, _ = env.step(a)
            G += r; steps += 1
            if term or trunc:
                returns.append(G); ep_steps.append(steps); mastered.append(bool(term))
                break
    return returns, ep_steps, mastered


def run_fixed_curriculum(num_episodes: int = 200, seed: int = 7, N: int = 3) -> Tuple[List[float], List[int], List[bool]]:
    """
    Deterministic curriculum:
      1) Two passes of TEACH over rules [0..N-1]
      2) Then loop QUIZ over [0..N-1] until episode ends
    """
    env = PedagoReLearnEnv(N=N, seed=seed)
    returns, ep_steps, mastered = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + seed)
        G, steps = 0.0, 0

        # Phase 1: teach passes
        done = False
        for _pass in range(2):
            for i in range(N):
                obs, r, term, trunc, _ = env.step(action_id("teach", i, N))
                G += r; steps += 1
                if term or trunc:
                    returns.append(G); ep_steps.append(steps); mastered.append(bool(term))
                    done = True
                    break
            if done:
                break

        # Phase 2: cycle quizzes until end
        if not done:
            i = 0
            while True:
                obs, r, term, trunc, _ = env.step(action_id("quiz", i, N))
                G += r; steps += 1
                if term or trunc:
                    returns.append(G); ep_steps.append(steps); mastered.append(bool(term))
                    break
                i = (i + 1) % N

    return returns, ep_steps, mastered


if __name__ == "__main__":
    rr, _, rm = run_random_policy()
    fr, _, fm = run_fixed_curriculum()
    print("Random:", np.mean(rr), np.mean(rm))
    print("Fixed:", np.mean(fr), np.mean(fm))