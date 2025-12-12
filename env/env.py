import numpy as np
from gymnasium import spaces
from students.student import Student

class Env:

    # PROGRESS_BONUS_BASE = 2.0
    STEP_COST = -1
    QUIZ_CORRECT_BONUS = 1.0
    QUIZ_WRONG_PENALTY = -0.20
    TERMINAL_BONUS = 500.0

    MAX_STEPS = 100

    def __init__(self, N: int = 3, seed: int | None = None, R_max: int = 30):
        self.N = int(N)
        self.R_max = int(R_max)
        self.max_steps = self.MAX_STEPS

        self._rng = np.random.default_rng(seed)
        
        # Student 
        self.student = Student(self.N, R_max=self.R_max, seed=seed)
        
        # Action space. Student owns the observation space
        self.action_space = spaces.Discrete(3 * self.N, self._rng)
        
        # Diagnostics
        self.r_sums = None
        self.hit_L3_once = None

        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options=None):
        # reset rng
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # reset space
        self.action_space = spaces.Discrete(3 * self.N, self._rng)
        # reset student
        self.student.reset(seed=seed)
        
        # reset diagnostics
        self.hit_L3_once = {i: False for i in range(self.N)}
        self.r_sums = dict(
            step=0.0, progress=0.0, l3=0.0,
            spacing_pos=0.0, spacing_neg=0.0,
            quiz_pos=0.0, quiz_neg=0.0,
            first_L3=0.0, terminal=0.0,
        )
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        # get state and action
        state = self._obs()
        mastery = state["mastery"].copy()
        i_type, rule_id = self._decode_action(action)

        # Instruct student
        ok = self.student.instruct(i_type,rule_id)

        # Calculate reward
        reward = 0.0
        # quiz-reward 
        if i_type == 'quiz':
            if ok:
                if mastery[rule_id] >= self.student.MASTERY_MAX:
                    rq = self.QUIZ_CORRECT_BONUS
                else:
                    rq = self.QUIZ_CORRECT_BONUS / 2
            else:
                rq = self.QUIZ_WRONG_PENALTY
            reward += rq
            self._accum_quiz(rq)
        
        # Step cost
        reward += self.STEP_COST; self.r_sums["step"] += self.STEP_COST

        # Termination
        done = bool(np.all(mastery == 3))
        truncated = bool(self.student.get_state()["step"] >= self.max_steps)
        if done:
            reward += self.TERMINAL_BONUS
            self.r_sums["terminal"] += self.TERMINAL_BONUS

        info = {"all_mastered": done, "r_sums": dict(self.r_sums)}
        return self._obs(), float(reward), done, truncated, info

    # -------------------------------
    # Helpers
    # -------------------------------
    def _obs(self):
        return self.student.get_state()

    def _decode_action(self, a: int):
        if a == 3 * self.N: # noop is not used
            return "noop", None
        i, t = divmod(int(a), 3)
        return ("teach", i) if t == 0 else ("quiz", i) if t == 1 else ("review", i)

    def _accum_quiz(self, r: float):
        if r > 0:   self.r_sums["quiz_pos"] += r
        elif r < 0: self.r_sums["quiz_neg"] += r

    def render(self):
        pass
