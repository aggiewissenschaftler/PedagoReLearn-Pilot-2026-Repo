# pedagorelearn_env_rewarded.py
# Week 6: Reward-shaped environment for PedagoReLearn
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PedagoReLearnEnvRewarded(gym.Env):
    """Reward-shaped tutoring environment.
    Adds progress, maintenance, and spacing-based rewards.
    """

    metadata = {"render_modes": []}

    # --- Tunable reward parameters ---
    STEP_COST = -0.10
    PROGRESS_BONUS = +1.00
    MAINTAIN_L3_BONUS = +0.20
    QUIZ_CORRECT_BONUS = +4.00
    QUIZ_WRONG_PENALTY = -2.00
    SPACING_IDEAL = 3
    SPACING_TOL = 2
    SPACING_PENALTY = -0.20
    SPACING_BONUS = +0.20
    TEACH_SUCCESS = 0.65
    REVIEW_REINFORCE = 0.30
    QUIZ_BASE_P = (0.45, 0.70, 0.88)
    FORGET_RECENCY_THRESHOLD = 6
    FORGET_PROBS = (0.30, 0.20, 0.10)
    TERMINAL_BONUS = +10.0
    MAX_STEPS = 120

    def __init__(self, N: int = 3, seed: int | None = None, R_max: int = 30):
        super().__init__()
        self.N = int(N)
        self.R_max = int(R_max)
        self.max_steps = self.MAX_STEPS

        self.action_space = spaces.Discrete(3 * self.N + 1)
        self.observation_space = spaces.Dict({
            "mastery": spaces.Box(low=0, high=3, shape=(self.N,), dtype=np.int32),
            "recency": spaces.Box(low=0, high=self.R_max, shape=(self.N,), dtype=np.int32),
            "step": spaces.Box(low=0, high=self.max_steps, shape=(), dtype=np.int32),
        })

        self._rng = np.random.default_rng(seed)
        self._mastery = None
        self._recency = None
        self._step = 0

    # ---------- core Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mastery = np.zeros(self.N, dtype=np.int32)
        self._recency = np.full(self.N, 5, dtype=np.int32)
        self._step = 0
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step += 1
        m_before = self._mastery.copy()

        reward = 0.0
        verb, idx = self._decode_action(action)

        if verb == "teach":
            reward += self._apply_teach(idx)
            reward += self._spacing_reward(idx)
        elif verb == "quiz":
            r, _ = self._apply_quiz(idx)
            reward += r + self._spacing_reward(idx)
        elif verb == "review":
            reward += self._apply_review(idx)
            reward += self._spacing_reward(idx)

        # reward shaping terms
        reward += self._progress_bonus(m_before, self._mastery)
        reward += self.MAINTAIN_L3_BONUS * float(np.mean(self._mastery == 3))
        self._apply_forgetting()
        self._recency = np.minimum(self._recency + 1, self.R_max)
        reward += self.STEP_COST

        done = bool(np.all(self._mastery == 3))
        truncated = bool(self._step >= self.max_steps)
        if done:
            reward += self.TERMINAL_BONUS
        return self._obs(), float(reward), done, truncated, {}

    # ---------- internal helpers ----------
    def _obs(self):
        return {
            "mastery": self._mastery.copy(),
            "recency": self._recency.copy(),
            "step": np.int32(self._step),
        }

    def _decode_action(self, a: int):
        if a == 3 * self.N:
            return "noop", None
        i, t = divmod(a, 3)
        return ("teach", i) if t == 0 else ("quiz", i) if t == 1 else ("review", i)

    def _spacing_reward(self, i: int):
        if i is None:
            return 0.0
        rec = int(self._recency[i])
        if rec < self.SPACING_IDEAL - self.SPACING_TOL:
            return self.SPACING_PENALTY
        if abs(rec - self.SPACING_IDEAL) <= self.SPACING_TOL:
            return self.SPACING_BONUS
        return 0.0

    def _progress_bonus(self, before, after):
        gains = (after - before)
        return self.PROGRESS_BONUS * float(np.sum(gains > 0))

    def _apply_teach(self, i: int):
        if self._mastery[i] < 3 and self._rng.random() < self.TEACH_SUCCESS:
            self._mastery[i] += 1
        self._recency[i] = 0
        return 0.0

    def _apply_review(self, i: int):
        if self._mastery[i] < 3 and self._rng.random() < self.REVIEW_REINFORCE:
            self._mastery[i] += 1
        self._recency[i] = 0
        return 0.0

    def _apply_quiz(self, i: int):
        m = int(self._mastery[i])
        if m >= 3:
            return self.QUIZ_CORRECT_BONUS, True
        p = self.QUIZ_BASE_P[m]
        correct = bool(self._rng.random() < p)
        if correct:
            if self._mastery[i] < 3 and self._rng.random() < 0.60:
                self._mastery[i] += 1
            r = self.QUIZ_CORRECT_BONUS
        else:
            if self._mastery[i] > 0 and self._rng.random() < 0.35:
                self._mastery[i] -= 1
            r = self.QUIZ_WRONG_PENALTY
        self._recency[i] = 0
        return r, correct

    def _apply_forgetting(self):
        mask = self._recency >= self.FORGET_RECENCY_THRESHOLD
        for i in np.where(mask)[0]:
            m = int(self._mastery[i])
            if m <= 0:
                continue
            drop_p = self.FORGET_PROBS[min(m - 1, len(self.FORGET_PROBS) - 1)]
            if self._rng.random() < drop_p:
                self._mastery[i] = m - 1

    def render(self):
        pass

    def close(self):
        pass