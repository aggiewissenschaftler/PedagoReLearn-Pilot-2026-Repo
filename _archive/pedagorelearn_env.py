# pedagorelearn_env.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import random

# Week 9: Gymnasium compliance
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rules_loader import merge_rule_files, flatten_required_rules
from student_model_complete import StudentModel

PathLike = Union[str, Path]


class PedagoReLearnEnv(gym.Env):
    """
    Gymnasium-compliant environment for PedagoReLearn.

    Observations: (skill 0..2, fatigue 0..2, pragmatics 0..2)
    Actions:      0=easy, 1=hard, 2=hint, 3=repeat, 4=pragmatics
    Episode ends when (skill + pragmatics) ≥ 4 OR max_steps reached.

    Notes
    -----
    • Cultural rules are loaded from YAML via rules_loader.merge_rule_files.
    • StudentModel implements time-based forgetting and mastery dynamics.
    • Reward shaping knobs remain identical to your Week 8 design.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rules_dir: PathLike = "rules",
        culture: str = "DE",
        seed: Optional[int] = None,
        max_steps: int = 40,
        verbose_rules: bool = True,
        # --- Shaping / alignment knobs ---
        correct_bonus: float = 10.0,
        wrong_penalty: float = -2.0,
        step_cost: float = -0.1,
        mastery_bonus: float = 2.0,
        pragmatics_bonus: float = 5.0,
        # --- Terminal incentives ---
        end_bonus_mastery: float = 20.0,   # large bonus at episode end if mastered
        timeout_penalty: float = -10.0,    # penalty if episode ends via timeout
        # (Optional) Week 9 flag; ignored by default but accepted for compatibility
        aggregation: bool = False,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self._ext_seed = seed
        self.max_steps = int(max_steps)
        self.culture = culture
        self.aggregation = bool(aggregation)

        # Reward parameters
        self.correct_bonus = float(correct_bonus)
        self.wrong_penalty = float(wrong_penalty)
        self.step_cost = float(step_cost)
        self.mastery_bonus = float(mastery_bonus)
        self.pragmatics_bonus = float(pragmatics_bonus)
        self.end_bonus_mastery = float(end_bonus_mastery)
        self.timeout_penalty = float(timeout_penalty)

        # Load YAML rules (prints which files were loaded)
        self.rules_dir = Path(rules_dir)
        self.rules = merge_rule_files(self.rules_dir, verbose=verbose_rules)
        self.required_bag = flatten_required_rules(self.rules, culture)

        # Student model with time-based forgetting
        self.student = StudentModel(culture_profile=culture, seed=seed)
        self.steps = 0

        # -------- Gymnasium spaces --------
        # Observation is a 3-tuple: skill, fatigue, pragmatics in {0,1,2}
        self.observation_space = spaces.MultiDiscrete([3, 3, 3])
        # Five discrete pedagogical actions
        self.action_space = spaces.Discrete(5)

    # -------- Gymnasium API --------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Seed RNGs
        if seed is not None:
            self._ext_seed = seed
        if self._ext_seed is not None:
            self.rng.seed(self._ext_seed)
        # fresh student with a sub-seed for stochasticity
        self.student = StudentModel(
            culture_profile=self.culture,
            seed=self.rng.randint(0, 10**9)
        )
        self.steps = 0
        obs = self._obs_to_ndarray(self.student.get_state())
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}; expected in [0..{self.action_space.n-1}]."
            )

        self.steps += 1
        s_next, correct, prag_ok = self.student.step(action, self._cultural_ok_prob())

        # Base reward shaping
        reward = 0.0
        if action in (0, 1, 2, 3):
            reward += self.correct_bonus if correct else self.wrong_penalty
        if action == 4 and prag_ok:
            reward += self.pragmatics_bonus
        reward += self.step_cost
        if self.student.mastery_improved():
            reward += self.mastery_bonus

        # Termination
        mastered = (self.student.skill + self.student.pragmatics) >= 4
        timed_out = self.steps >= self.max_steps
        terminated = bool(mastered)     # task success
        truncated = bool(timed_out and not mastered)  # time limit

        # Terminal incentives to encourage finishing
        if terminated:
            reward += self.end_bonus_mastery
        elif truncated:
            reward += self.timeout_penalty

        info: Dict[str, Any] = {
            "correct": correct,
            "prag_ok": prag_ok,
            "steps": self.steps,
            "done_reason": "mastery" if mastered else ("timeout" if timed_out else None),
        }

        obs = self._obs_to_ndarray(s_next)
        return obs, float(reward), terminated, truncated, info

    # -------- Helpers --------
    def _cultural_ok_prob(self) -> float:
        bag = max(1, len(self.required_bag))
        return min(0.9, 0.5 + 0.4 * (bag / (bag + 10)))

    @staticmethod
    def _obs_to_ndarray(obs_tuple: Tuple[int, int, int]) -> np.ndarray:
        # Ensure obs matches MultiDiscrete([3,3,3])
        return np.asarray(obs_tuple, dtype=np.int64)

    # (Optional stubs to satisfy some Gym tooling)
    def render(self) -> None:  # noqa: D401
        """No-op (text rendering not implemented)."""
        return None

    def close(self) -> None:
        return None