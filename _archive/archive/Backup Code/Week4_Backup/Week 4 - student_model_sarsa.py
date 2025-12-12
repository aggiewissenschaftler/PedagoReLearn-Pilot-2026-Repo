from __future__ import annotations
from collections import defaultdict
import random
from typing import Dict, Tuple, Any
import numpy as np


def encode_obs(obs: Dict[str, np.ndarray], step_bins: Tuple[int, ...] = (0, 10, 25, 50, 100)) -> Tuple:
    """
    Turn the environment observation dict into a hashable state key.
    State = ("m", *mastery, "r", *recency, "t", step_bin)
    """
    mastery = tuple(int(x) for x in obs["mastery"].tolist())
    recency = tuple(int(x) for x in obs["recency"].tolist())

    step = int(obs["step"])
    for k, cut in enumerate(step_bins):
        if step <= cut:
            bin_idx = k
            break
    else:
        bin_idx = len(step_bins)

    return ("m",) + mastery + ("r",) + recency + ("t", bin_idx)


class SarsaAgent:
    """
    Tabular SARSA(0) agent with epsilon-greedy policy.
    """
    def __init__(self, n_actions: int, epsilon: float = 0.20, alpha: float = 0.5, gamma: float = 0.97, seed: int | None = None):
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        # Q-table: Q[state][action] = value
        self.Q: Dict[Any, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def act(self, state) -> int:
        # epsilon-greedy
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        qsa = self.Q[state]
        best_a, best_q = 0, -float("inf")
        for a in range(self.n_actions):
            v = qsa.get(a, 0.0)
            if v > best_q or (v == best_q and a < best_a):
                best_q, best_a = v, a
        return best_a

    def update(self, s, a, r, s_next, a_next):
        # SARSA(0): Q[s,a] ← Q[s,a] + α * (r + γ*Q[s',a'] − Q[s,a])
        qsa = self.Q[s].get(a, 0.0)
        next_q = self.Q[s_next].get(a_next, 0.0)
        target = r + self.gamma * next_q
        self.Q[s][a] = qsa + self.alpha * (target - qsa)

    def decay_epsilon(self, floor: float = 0.05, rate: float = 0.997):
        self.epsilon = max(floor, self.epsilon * rate)