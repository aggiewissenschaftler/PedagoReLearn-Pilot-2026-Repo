from collections import defaultdict
import gymnasium as gym
import numpy as np

# Abstract base tutor shared by specific policies.
class AbstractTutor:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 0.3,
        epsilon_decay: float = 0.0,
        final_epsilon: float = 0.05,
        discount_factor: float = 0.95,
    ):
        self.env = env

        self.lr = float(learning_rate)
        self.discount_factor = float(discount_factor)

        # Exploration parameters
        self.epsilon = float(initial_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.final_epsilon = float(final_epsilon)

    def get_action(self, obs) -> int:
        """Return an action for the given observation."""
        raise NotImplementedError

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Update internal policy/value estimates from a transition."""
        raise NotImplementedError

    def decay_epsilon(self):
        """Reduce exploration rate (linear schedule)."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class RandomPolicyTutor(AbstractTutor):
    """Tutor that selects actions uniformly at random."""

    def get_action(self, obs) -> int:  
        return int(self.env.action_space.sample())

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        # Random policy does not learn.
        return None
    
class FixedPolicyTutor(AbstractTutor):
    """Tutor that selects action in rotation."""
    def __init__(self, env):
        super().__init__(env)
        self._i = 0

    def get_action(self, obs):
        # If discrete, cycle deterministically; otherwise fall back to random
        n = getattr(self.env.action_space, "n", None)
        if n is not None:
            a = self._i % n
            self._i += 1
            return a
        return self.env.action_space.sample()
    
    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        # Fixed policy does not learn.
        return None


class SARSATutor(AbstractTutor):
    """Tabular SARSA(0) with eps-greedy policy over discrete actions."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 0.3,
        epsilon_decay: float = 0.0,
        final_epsilon: float = 0.05,
        discount_factor: float = 0.97,
        seed: int | None = None,
    ):
        super().__init__(
            env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        self._rng = np.random.default_rng(seed)
        self._n_actions = int(env.action_space.n)
        self.Q: dict[object, np.ndarray] = defaultdict(
            lambda : np.full(self._n_actions, 0, dtype=float)
        )

    # ---------- state encoding helpers ----------
    @staticmethod
    def _encode_obs(obs: dict) -> tuple:
        """Encode dict observation into a compact, hashable key.

        Expected obs keys: 'mastery' (array), 'recency' (array)
        Step is bucketed to stabilize tabular state space size.
        """
        mastery = tuple(int(x) for x in np.asarray(obs["mastery"]).tolist())
        recency = tuple(int(x) for x in np.asarray(obs["recency"]).tolist())

        return ("m",) + mastery + ("r",) + recency

    def _state_key(self, s) -> object:
        if isinstance(s, dict) and {"mastery", "recency"}.issubset(s.keys()):
            try:
                return self._encode_obs(s)
            except Exception:
                pass
        # Fallbacks for other potential obs types
        if isinstance(s, dict):
            return tuple(sorted((k, self._coerce_val(v)) for k, v in s.items()))
        if isinstance(s, (list, tuple)):
            return tuple(self._coerce_val(v) for v in s)
        return s

    @staticmethod
    def _coerce_val(v):
        if isinstance(v, np.ndarray):
            return tuple(v.tolist())
        if isinstance(v, (list, tuple)):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted((k, SARSATutor._coerce_val(x)) for k, x in v.items()))
        return v

    # ---------- policy ----------
    def get_action(self, obs) -> int:
        key = self._state_key(obs)
        q_vals = self.Q[key]

        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self._n_actions))

        max_q = float(np.max(q_vals))
        best = np.flatnonzero(q_vals == max_q)
        if best.size == 0:
            return int(self._rng.integers(self._n_actions))
        choice_idx = int(self._rng.integers(best.size))
        return int(best[choice_idx])

    # ---------- learning ----------
    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        s_key = self._state_key(obs)
        sp_key = self._state_key(next_obs)

        q_s = self.Q[s_key]
        q_sp = self.Q[sp_key]

        # On-policy bootstrap action a'
        next_action = self.get_action(next_obs)

        target = reward
        if not terminated:
            target += self.discount_factor * q_sp[next_action]

        q_old = q_s[action]
        q_s[action] = q_old + self.lr * (target - q_old)

        return next_action
