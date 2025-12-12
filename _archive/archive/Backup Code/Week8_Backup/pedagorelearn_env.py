from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import random

from rules_loader import merge_rule_files, flatten_required_rules
from student_model_complete import StudentModel

PathLike = Union[str, Path]


class PedagoReLearnEnv:
    """
    Week 8 environment (finalized):
      • Observations: (skill 0..2, fatigue 0..2, pragmatics 0..2)
      • Actions: 0=easy, 1=hard, 2=hint, 3=repeat, 4=pragmatics
      • Episode ends when (skill + pragmatics) ≥ 4 OR max_steps reached
      • Cultural rules loaded from YAML via rules_loader.merge_rule_files (verbose)
      • Time-based forgetting implemented in StudentModel
      • Reward knobs exposed for alignment
    """
    def __init__(
        self,
        rules_dir: PathLike,
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
        # --- NEW: terminal incentives ---
        end_bonus_mastery: float = 20.0,   # large bonus at episode end if mastered
        timeout_penalty: float = -10.0,    # penalty if episode ends via timeout
    ) -> None:
        self.rng = random.Random(seed)
        self.max_steps = int(max_steps)
        self.culture = culture

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

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.student = StudentModel(culture_profile=self.culture, seed=self.rng.randint(0, 10**9))
        self.steps = 0
        return self.student.get_state(), {}

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        if action not in (0, 1, 2, 3, 4):
            raise ValueError(f"Invalid action {action}; expected one of {{0,1,2,3,4}}.")

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
        done = mastered or timed_out

        # Terminal incentives to encourage finishing
        if done:
            if mastered:
                reward += self.end_bonus_mastery
            elif timed_out:
                reward += self.timeout_penalty

        info: Dict[str, Any] = {
            "correct": correct,
            "prag_ok": prag_ok,
            "steps": self.steps,
            "done_reason": "mastery" if mastered else ("timeout" if timed_out else None),
        }
        return s_next, float(reward), bool(done), info

    def _cultural_ok_prob(self) -> float:
        bag = max(1, len(self.required_bag))
        return min(0.9, 0.5 + 0.4 * (bag / (bag + 10)))