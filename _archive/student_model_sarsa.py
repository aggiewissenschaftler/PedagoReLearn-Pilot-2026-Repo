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
    mastery = tuple(int(x) for x in np.asarray(obs["mastery"]).tolist())
    recency = tuple(int(x) for x in np.asarray(obs["recency"]).tolist())

    step = int(np.asarray(obs["step"]))
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
    Q is stored as a dict: state_key -> np.ndarray(shape=(n_actions,))
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.20,
        alpha: float = 0.5,
        gamma: float = 0.97,
        seed: int | None = None,
    ):
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)

        # Q-table: state_key -> np.array of action-values
        self.Q: Dict[Any, np.ndarray] = {}

    # --------- state key helpers ---------
    def _state_key(self, s: Any) -> Any:
        """
        Convert unhashable observations (dict, list, etc.) into a stable key.
        - If it's the PedagoReLearn dict obs with arrays, use encode_obs (preferred).
        - Otherwise, fall back to a sorted tuple of items / tuple(list).
        """
        if isinstance(s, dict) and {"mastery", "recency", "step"}.issubset(s.keys()):
            try:
                return encode_obs(s)
            except Exception:
                pass

        if isinstance(s, dict):
            return tuple(sorted((k, self._coerce_val(v)) for k, v in s.items()))
        if isinstance(s, (list, tuple)):
            return tuple(self._coerce_val(v) for v in s)
        return s

    @staticmethod
    def _coerce_val(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return tuple(v.tolist())
        if isinstance(v, (list, tuple)):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted((k, SarsaAgent._coerce_val(x)) for k, x in v.items()))
        return v

    def _ensure_state(self, key: Any) -> np.ndarray:
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions, dtype=float)
        return self.Q[key]

    # --------- policy ---------
    def select_action(self, state: Any) -> int:
        """ε-greedy over Q(state, ·)."""
        key = self._state_key(state)
        qvals = self._ensure_state(key)

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        max_q = np.max(qvals)
        best_actions = np.flatnonzero(qvals == max_q)
        if len(best_actions) == 0:
            return self.rng.randrange(self.n_actions)
        return int(self.rng.choice(best_actions))

    # alias for compatibility with the runner
    act = select_action

    # --------- learning ---------
    def update(self, s, a: int, r: float, s_next, a_next: int, done: bool | None = None):
        """
        SARSA(0): Q[s,a] ← Q[s,a] + α * (r + γ * Q[s',a'] − Q[s,a])
        If `done` is True, bootstrap term is omitted (target = r).
        """
        s_key = self._state_key(s)
        sp_key = self._state_key(s_next)

        q_s = self._ensure_state(s_key)
        q_sp = self._ensure_state(sp_key)

        qsa = q_s[a]
        next_q = 0.0 if done else q_sp[a_next]
        target = r + self.gamma * next_q

        q_s[a] = qsa + self.alpha * (target - qsa)

    # friendly aliases
    def learn(self, s, a, r, s_next, a_next, done: bool | None = None):
        self.update(s, a, r, s_next, a_next, done)

    def observe(self, *args, **kwargs):
        """Accepts either (s,a,r,s',a',done) or (s,a,r,s',a') and forwards to update."""
        if len(args) == 6:
            s, a, r, s_next, a_next, done = args
            self.update(s, a, r, s_next, a_next, done)
        elif len(args) == 5:
            s, a, r, s_next, a_next = args
            self.update(s, a, r, s_next, a_next, None)
        else:
            pass

    # --------- utilities ---------
    def decay_epsilon(self, floor: float = 0.05, rate: float = 0.997):
        self.epsilon = max(floor, self.epsilon * rate)

    def start_episode(self, *_args, **_kwargs):  # optional hooks
        pass

    def end_episode(self, *_args, **_kwargs):
        pass