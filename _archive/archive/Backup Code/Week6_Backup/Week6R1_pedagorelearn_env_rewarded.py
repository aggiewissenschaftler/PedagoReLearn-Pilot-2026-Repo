# pedagorelearn_env_rewarded.py
# Week 6: Reward-shaped environment for PedagoReLearn
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PedagoReLearnEnvRewarded(gym.Env):
    """
    Reward-shaped tutoring environment for PedagoReLearn.
    - Global REWARD_SCALE scales the *major positive* rewards at init.
    - Spacing bonus fires in a sweet spot; lateness penalized (early = neutral).
    - First-time L3 bonus encourages breadth.
    - r_sums diagnostics print at episode end for quick reward forensics.
    """

    metadata = {"render_modes": []}

    # -------------------------------
    # Global control knobs
    # -------------------------------
    REWARD_SCALE: float = 8.0   # Try 8.0 / 10.0 / 12.0 to tune overall signal strength
    SCALE_PENALTIES: bool = False  # If True, also scales penalties & step cost

    # -------------------------------
    # Base (unscaled) tunables by subsystem
    # -------------------------------

    # --- Reward shaping (what the agent feels each step/action) ---
    PROGRESS_BONUS_BASE      = 2.0    # + for level ups (e.g., L1->L2 or L2->L3)
    MAINTAIN_L3_BONUS        = 0.30   # + small drip each step (see per-item calc below)
    SPACING_BONUS            = 0.35   # + when practice at ideal spacing band
    STEP_COST_BASE           = -0.01  # - small per-step drag
    SPACING_PENALTY_BASE     = -0.03  # - if practice far past ideal spacing
    QUIZ_CORRECT_BONUS_BASE  = 7.0    # + when a quiz is answered correctly
    QUIZ_WRONG_PENALTY_BASE  = -0.20  # - when a quiz is answered incorrectly
    TERMINAL_BONUS_BASE      = 20.0   # + once when all items reach L3
    FIRST_TIME_L3_BONUS_BASE = 2.0    # + once per item the first time it hits L3

    # --- Spacing policy (sweet spot definition) ---
    SPACING_IDEAL = 3              # ideal gap (in steps) since last practice
    SPACING_TOL   = 1              # tolerance around the ideal (|gap-ideal| <= TOL => bonus)

    # --- Student model / transition dynamics ---
    TEACH_SUCCESS             = 0.70                    # P(level up) on TEACH (if < L3)
    REVIEW_REINFORCE          = 0.30                    # P(level up) on REVIEW (if < L3)
    QUIZ_BASE_P               = (0.45, 0.70, 0.88)      # P(correct) at L1, L2, L3 (pre-spacing)
    QUIZ_LVLUP_ON_CORRECT_P   = 0.60                    # P(level up) when quiz is correct (if < L3)

    # --- Forgetting model (decay with disuse) ---
    FORGET_RECENCY_THRESHOLD  = 6                       # eligible to forget if gap > threshold
    FORGET_PROBS              = (0.30, 0.20, 0.10)      # P(drop 1 level) from L1, L2, L3

    # --- Episode control ---
    MAX_STEPS = 100                                     # step cap per episode (can use 90–120)

    # -------------------------------
    # Gym API
    # -------------------------------
    def __init__(self, N: int = 3, seed: int | None = None, R_max: int = 30):
        super().__init__()
        self.N = int(N)
        self.R_max = int(R_max)
        self.max_steps = self.MAX_STEPS

        # --- Apply global reward scaling (copy class constants to instance, scaled) ---
        # Scale MAJOR positive rewards; penalties optionally scaled for stricter environments.
        self.PROGRESS_BONUS      = self.PROGRESS_BONUS_BASE      * self.REWARD_SCALE
        self.QUIZ_CORRECT_BONUS  = self.QUIZ_CORRECT_BONUS_BASE  * self.REWARD_SCALE
        self.TERMINAL_BONUS      = self.TERMINAL_BONUS_BASE      * self.REWARD_SCALE
        self.FIRST_TIME_L3_BONUS = self.FIRST_TIME_L3_BONUS_BASE * self.REWARD_SCALE

        if self.SCALE_PENALTIES:
            self.QUIZ_WRONG_PENALTY = self.QUIZ_WRONG_PENALTY_BASE * self.REWARD_SCALE
            self.STEP_COST          = self.STEP_COST_BASE          * self.REWARD_SCALE
            self.SPACING_PENALTY    = self.SPACING_PENALTY_BASE    * self.REWARD_SCALE
        else:
            self.QUIZ_WRONG_PENALTY = self.QUIZ_WRONG_PENALTY_BASE
            self.STEP_COST          = self.STEP_COST_BASE
            self.SPACING_PENALTY    = self.SPACING_PENALTY_BASE

        # Unscaled “nudges” left as-is (intentionally small):
        self.MAINTAIN_L3_BONUS = self.MAINTAIN_L3_BONUS
        self.SPACING_BONUS     = self.SPACING_BONUS

        # Action/observation spaces
        self.action_space = spaces.Discrete(3 * self.N + 1)  # teach/quiz/review for each item + noop
        self.observation_space = spaces.Dict({
            "mastery": spaces.Box(low=0, high=3, shape=(self.N,), dtype=np.int32),
            "recency": spaces.Box(low=0, high=self.R_max, shape=(self.N,), dtype=np.int32),
            "step":    spaces.Box(low=0, high=self.max_steps, shape=(), dtype=np.int32),
        })

        # State
        self._rng = np.random.default_rng(seed)
        self._mastery = None
        self._recency = None
        self._step = 0
        self.hit_L3_once = None  # per-item flag for FIRST_TIME_L3_BONUS

        # Diagnostics
        self.r_sums = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._mastery = np.zeros(self.N, dtype=np.int32)
        self._recency = np.full(self.N, 5, dtype=np.int32)  # start with moderate gaps
        self._step = 0
        self.hit_L3_once = {i: False for i in range(self.N)}

        # Reward diagnostics (per-episode running sums)
        self.r_sums = dict(
            step=0.0, progress=0.0, l3=0.0,
            spacing_pos=0.0, spacing_neg=0.0,
            quiz_pos=0.0, quiz_neg=0.0,
            first_L3=0.0, terminal=0.0,
        )
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step += 1
        m_before = self._mastery.copy()
        reward = 0.0

        verb, idx = self._decode_action(action)

        # --- Action effects + spacing reward tied to chosen item (if any) ---
        if verb == "teach" and idx is not None:
            reward += self._apply_teach(idx)
            sr = self._spacing_reward(idx); reward += sr; self._accum_spacing(sr)

        elif verb == "quiz" and idx is not None:
            rq, _ok = self._apply_quiz(idx); reward += rq; self._accum_quiz(rq)
            sr = self._spacing_reward(idx); reward += sr; self._accum_spacing(sr)

        elif verb == "review" and idx is not None:
            reward += self._apply_review(idx)
            sr = self._spacing_reward(idx); reward += sr; self._accum_spacing(sr)

        # noop does nothing but time still advances below

        # --- Progress bonus (for any level ups this step) ---
        pr = self._progress_bonus(m_before, self._mastery)
        reward += pr; self.r_sums["progress"] += pr

        # --- First-time L3 bonus (one-off per item) ---
        newly_L3 = np.where((self._mastery == 3) & (m_before < 3))[0]
        for i in newly_L3:
            if not self.hit_L3_once[i]:
                reward += self.FIRST_TIME_L3_BONUS
                self.r_sums["first_L3"] += self.FIRST_TIME_L3_BONUS
                self.hit_L3_once[i] = True

        # --- Maintenance drip while at L3 (per-L3-item, per-step) ---
        l3_count = int(np.sum(self._mastery == 3))
        l3_r = self.MAINTAIN_L3_BONUS * l3_count
        reward += l3_r; self.r_sums["l3"] += l3_r

        # --- Forgetting + time passing ---
        self._apply_forgetting()
        self._recency = np.minimum(self._recency + 1, self.R_max)

        # --- Per-step drag ---
        reward += self.STEP_COST
        self.r_sums["step"] += self.STEP_COST

        # --- Episode termination ---
        done = bool(np.all(self._mastery == 3))
        truncated = bool(self._step >= self.max_steps)
        if done:
            reward += self.TERMINAL_BONUS
            self.r_sums["terminal"] += self.TERMINAL_BONUS

        # Diagnostics print
        if done or truncated:
            print(f"[EP_END] r_sums={self.r_sums}, steps={self._step}, all_mastered={done}")

        return self._obs(), float(reward), done, truncated, {}

    # -------------------------------
    # Helpers
    # -------------------------------
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

    def _spacing_reward(self, i: int) -> float:
        """Reward based on time since last practice for item i.
        Bonus in sweet spot; penalty only when LATE; early is neutral.
        """
        if i is None:
            return 0.0
        gap = int(self._recency[i])
        if abs(gap - self.SPACING_IDEAL) <= self.SPACING_TOL:
            return self.SPACING_BONUS
        if gap > self.SPACING_IDEAL + self.SPACING_TOL:
            return self.SPACING_PENALTY
        return 0.0

    def _progress_bonus(self, before: np.ndarray, after: np.ndarray) -> float:
        gains = (after - before)
        return self.PROGRESS_BONUS * float(np.sum(gains > 0))

    def _apply_teach(self, i: int) -> float:
        if self._mastery[i] < 3 and self._rng.random() < self.TEACH_SUCCESS:
            self._mastery[i] += 1
        self._recency[i] = 0
        return 0.0

    def _apply_review(self, i: int) -> float:
        if self._mastery[i] < 3 and self._rng.random() < self.REVIEW_REINFORCE:
            self._mastery[i] += 1
        self._recency[i] = 0
        return 0.0

    def _apply_quiz(self, i: int):
        m = int(self._mastery[i])
        if m >= 3:
            r = self.QUIZ_CORRECT_BONUS
            self._recency[i] = 0
            return r, True

        p = self.QUIZ_BASE_P[m]  # (optionally: apply spacing modifier here)
        correct = bool(self._rng.random() < p)

        if correct:
            if self._mastery[i] < 3 and self._rng.random() < self.QUIZ_LVLUP_ON_CORRECT_P:
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
            drop_idx = min(m - 1, len(self.FORGET_PROBS) - 1)
            if self._rng.random() < self.FORGET_PROBS[drop_idx]:
                self._mastery[i] = m - 1

    # --- Diagnostics helpers ---
    def _accum_spacing(self, r: float):
        if r > 0:   self.r_sums["spacing_pos"] += r
        elif r < 0: self.r_sums["spacing_neg"] += r

    def _accum_quiz(self, r: float):
        if r > 0:   self.r_sums["quiz_pos"] += r
        elif r < 0: self.r_sums["quiz_neg"] += r

    def render(self): pass
    def close(self): pass