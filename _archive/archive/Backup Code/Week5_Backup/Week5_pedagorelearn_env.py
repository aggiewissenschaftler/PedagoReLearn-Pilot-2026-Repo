
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from typing import Dict, Tuple, Optional, List

# -----------------------------
# Utility: load N rule names from YAML files (optional)
# -----------------------------
def load_rule_names_from_yaml(yaml_paths: List[str], N: int) -> List[str]:
    """
    Try to pull REQUIRED rule keys from provided YAMLs (DE -> REQUIRED -> keys).
    Falls back to generic rule names if not enough are found.
    """
    names = []
    for p in yaml_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            req = (data.get("DE") or {}).get("REQUIRED") or {}
            for key in req.keys():
                names.append(key)
                if len(names) >= N:
                    break
            if len(names) >= N:
                break
        except Exception:
            continue
    # Fallback
    while len(names) < N:
        names.append(f"rule_{len(names)}")
    return names[:N]


class PedagoReLearnEnv(gym.Env):
    """
    PedagoReLearn: minimal Gymnasium environment for RL tutoring over cultural rules.

    Observation (Dict):
      - mastery: MultiDiscrete([4]*N)  values in {0,1,2,3}
      - recency: MultiDiscrete([H+1]*N) values in {0..H}
      - step:    Discrete(max_steps+1)

    Action (Discrete):
      index -> (verb, i), where verb in {teach, quiz, review} for i in [0..N-1]; plus a single no-op.
      So |A| = 3N + 1.

    Dynamics (simplified from proposal):
      - teach(i): raises mastery with probability depending on current m_i; recency[i]=0.
      - quiz(i):  correct prob depends on m_i; reward based on correct/incorrect; may change mastery; recency[i]=0.
      - review(i): small chance to increase mastery if 1<=m_i<=2; recency[i]=0.
      - no-op: do nothing except time passes.
      - forgetting: if recency >= 4 then mastery may decrease (smaller prob when at level 3).
      - per-step cost -0.1; terminal +50 when all mastered (m_i==3 for all i).

    Episode ends when:
      - all rules mastered, or
      - step reaches max_steps, or
      - agent takes 3 consecutive no-ops.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        N: int = 3,
        H: int = 5,
        max_steps: int = 100,
        seed: Optional[int] = None,
        yaml_rule_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.N = int(N)
        self.H = int(H)
        self.max_steps = int(max_steps)

        # Action mapping: 0..3N-1 -> teach/quiz/review, 3N -> no-op
        self.num_actions = 3 * self.N + 1
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space
        self.observation_space = spaces.Dict({
            "mastery": spaces.MultiDiscrete([4] * self.N),
            "recency": spaces.MultiDiscrete([self.H + 1] * self.N),
            "step": spaces.Discrete(self.max_steps + 1),
        })

        # Probabilities (from proposal baseline; tweak as needed)
        self.teach_upgrade_probs = {0: 0.9, 1: 0.6, 2: 0.4, 3: 0.0}
        self.quiz_success_probs  = {0: 0.1, 1: 0.4, 2: 0.7, 3: 0.95}
        self.quiz_up_on_success = 0.3
        self.quiz_down_on_fail  = 0.2
        self.review_upgrade_prob = 0.2

        # Forgetting parameters
        self.forget_recency_threshold = 4
        self.forget_probs = {1: 0.15, 2: 0.15, 3: 0.05}  # only applies when recency >= threshold

        # Rewards
        self.step_cost = -0.1
        self.quiz_reward_correct_lo = +5
        self.quiz_reward_correct_hi = +10  # when already at mastery (m==3)
        self.quiz_reward_incorrect  = -2
        self.terminal_reward = +50

        # Book-keeping
        self._rng = np.random.default_rng(seed)
        self._rule_names = load_rule_names_from_yaml(yaml_rule_paths or [], self.N)

        self._mastery = None
        self._recency = None
        self._step = 0
        self._noop_streak = 0

        self.reset(seed=seed)


    # -----------------------------
    # Gymnasium API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mastery = np.zeros(self.N, dtype=np.int64)  # all unseen
        self._recency = np.zeros(self.N, dtype=np.int64)  # fresh
        self._step = 0
        self._noop_streak = 0
        obs = self._get_obs()
        info = {"rule_names": self._rule_names}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Per-step efficiency cost
        reward += self.step_cost

        verb, idx = self._decode_action(action)

        # Apply chosen action
        if verb == "noop":
            self._noop_streak += 1
        else:
            self._noop_streak = 0  # reset streak on any non-noop

        if verb == "teach":
            reward += self._apply_teach(idx)
        elif verb == "quiz":
            rew_quiz, correct = self._apply_quiz(idx)
            reward += rew_quiz
            info["quiz_correct"] = bool(correct)
        elif verb == "review":
            reward += self._apply_review(idx)
        elif verb == "noop":
            pass

        # Time passes: update recency (chosen rule recency was already reset inside action handlers)
        self._tick_time(idx if verb in {"teach", "quiz", "review"} else None)

        # Passive forgetting after time passes
        self._apply_forgetting()

        # Check termination conditions
        if np.all(self._mastery == 3):
            reward += self.terminal_reward
            terminated = True
        elif self._step >= self.max_steps:
            truncated = True
        elif self._noop_streak >= 3:
            truncated = True

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    # -----------------------------
    # Action helpers
    # -----------------------------
    def _decode_action(self, a: int) -> Tuple[str, Optional[int]]:
        last = 3 * self.N
        if a == last:
            return "noop", None
        i = a // 3
        t = a % 3
        return (["teach", "quiz", "review"][t], i)

    def _apply_teach(self, i: int) -> float:
        m = int(self._mastery[i])
        p = self.teach_upgrade_probs[m]
        if self._rng.random() < p and m < 3:
            self._mastery[i] = m + 1
        # Teaching has no immediate reward in this shaping
        # Reset recency for acted-on rule
        self._recency[i] = 0
        return 0.0

    def _apply_quiz(self, i: int) -> Tuple[float, bool]:
        m = int(self._mastery[i])
        p_succ = self.quiz_success_probs[m]
        correct = self._rng.random() < p_succ

        if correct:
            r = self.quiz_reward_correct_hi if m == 3 else self.quiz_reward_correct_lo
            # chance to upgrade on success
            if m < 3 and self._rng.random() < self.quiz_up_on_success:
                self._mastery[i] = m + 1
        else:
            r = self.quiz_reward_incorrect
            # chance to downgrade on failure
            if m > 0 and self._rng.random() < self.quiz_down_on_fail:
                self._mastery[i] = m - 1

        self._recency[i] = 0
        return r, correct

    def _apply_review(self, i: int) -> float:
        m = int(self._mastery[i])
        if 1 <= m <= 2:
            if self._rng.random() < self.review_upgrade_prob:
                self._mastery[i] = m + 1
        self._recency[i] = 0
        return 0.0

    def _tick_time(self, acted_index: Optional[int]):
        # Advance global step
        self._step += 1
        # Increase recency for all except possibly the one just acted upon (already reset)
        for j in range(self.N):
            if acted_index is not None and j == acted_index:
                continue
            self._recency[j] = min(self.H, self._recency[j] + 1)

    def _apply_forgetting(self):
        for i in range(self.N):
            if self._recency[i] >= self.forget_recency_threshold:
                m = int(self._mastery[i])
                if m in self.forget_probs and self._rng.random() < self.forget_probs[m]:
                    self._mastery[i] = m - 1

    # -----------------------------
    # Observation
    # -----------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "mastery": self._mastery.copy(),
            "recency": self._recency.copy(),
            "step": np.array(self._step, dtype=np.int64),
        }

    # -----------------------------
    # Convenience
    # -----------------------------
    @property
    def rule_names(self) -> List[str]:
        return list(self._rule_names)

    def action_meanings(self) -> List[str]:
        names = []
        for a in range(self.num_actions):
            verb, idx = self._decode_action(a)
            if verb == "noop":
                names.append("noop")
            else:
                names.append(f"{verb}({idx}:{self._rule_names[idx]})")
        return names
