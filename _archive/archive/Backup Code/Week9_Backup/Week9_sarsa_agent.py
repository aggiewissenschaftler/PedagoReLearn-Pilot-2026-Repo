from __future__ import annotations
import random
from collections import defaultdict
from typing import Tuple


class SarsaAgent:
    def __init__(
        self,
        n_actions: int = 5,
        alpha: float = 0.4,
        gamma: float = 0.95,
        eps: float = 0.2,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = random.Random(seed)
        self.Q = defaultdict(float)

    def _key(self, s: Tuple[int, int, int], a: int) -> Tuple[int, int, int, int]:
        return (s[0], s[1], s[2], a)

    def act(self, state: Tuple[int, int, int]) -> int:
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        qs = [self.Q[self._key(state, a)] for a in range(self.n_actions)]
        max_q = max(qs)
        best = [a for a, q in enumerate(qs) if q == max_q]
        return self.rng.choice(best)

    def learn(self, s, a, r, s_next, a_next, done):
        key = self._key(s, a)
        target = r + (0.0 if done else self.gamma * self.Q[self._key(s_next, a_next)])
        self.Q[key] += self.alpha * (target - self.Q[key])

    # --- exploration decay (Week-8 improvement) ---
    def decay(self, factor: float = 0.99, min_eps: float = 0.05):
        self.eps = max(min_eps, self.eps * factor)