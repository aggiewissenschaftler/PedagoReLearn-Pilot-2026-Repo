
from __future__ import annotations
import math
import random
from typing import Tuple, Optional, Dict, Any, List

class StudentModel:
    """
    Simple student model with time-based forgetting and a culture-sensitive pragmatics channel.

    State (discrete 0..2 each):
      - skill: cognitive mastery of the target concept
      - fatigue: increases with hard tasks, decreases with rest/repeat/hints
      - pragmatics: cultural/etiquette mastery (independent but correlated with practice)

    Time-based forgetting:
      - Each step we track "since_practice_skill" and "since_practice_prag".
      - If not practiced for T_f steps, we decay with probability p_f per step.
    """
    def __init__(self, culture_profile: str = "DE", seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.culture_profile = culture_profile

        self.skill = 0
        self.fatigue = 0
        self.pragmatics = 0

        self.prev_mastery = self.skill + self.pragmatics

        self.since_practice_skill = 0
        self.since_practice_prag = 0

        # Forgetting hyperparams (tunable):
        self.T_f = 4            # grace steps without practice
        self.p_f = 0.25         # per-step decay probability after T_f

    def get_state(self) -> Tuple[int, int, int]:
        return (self.skill, self.fatigue, self.pragmatics)

    def _apply_forgetting(self) -> None:
        # Skill decay
        if self.since_practice_skill > self.T_f and self.skill > 0:
            if self.rng.random() < self.p_f:
                self.skill -= 1
        # Pragmatics decay
        if self.since_practice_prag > self.T_f and self.pragmatics > 0:
            if self.rng.random() < self.p_f:
                self.pragmatics -= 1

    def step(self, action: int, cultural_ok_prob: float) -> Tuple[Tuple[int,int,int], bool, bool]:
        """
        Apply the tutor's action and evolve the student.
        action: 0=easy,1=hard,2=hint,3=repeat,4=pragmatics
        cultural_ok_prob: probability that a pragmatics interaction aligns with cultural rules.
        Returns: (state, correctness, pragmatics_ok)
        """
        # Base correctness probability depends on skill & fatigue and task difficulty
        base_correct = [0.3, 0.6, 0.85][self.skill]
        fatigue_penalty = [0.0, 0.1, 0.25][self.fatigue]
        if action == 0:   # easy
            task_diff = -0.1
            fatigue_delta = 0
            practice_skill = True
        elif action == 1: # hard
            task_diff = +0.15
            fatigue_delta = +1
            practice_skill = True
        elif action == 2: # hint
            task_diff = -0.15
            fatigue_delta = -1
            practice_skill = True
        elif action == 3: # repeat (light practice, rest)
            task_diff = -0.05
            fatigue_delta = -1
            practice_skill = False  # repetition consolidates less; model as non-practice for decay timer
        elif action == 4: # pragmatics focus
            task_diff = -0.05
            fatigue_delta = 0
            practice_skill = False
        else:
            task_diff = 0.0
            fatigue_delta = 0
            practice_skill = False

        # Clamp fatigue to 0..2
        self.fatigue = max(0, min(2, self.fatigue + fatigue_delta))

        # Correctness outcome (only if not pure pragmatics)
        if action in (0,1,2,3):
            p_correct = max(0.0, min(1.0, base_correct - fatigue_penalty - max(0.0, task_diff)))
            correctness = (self.rng.random() < p_correct)
            if correctness and self.skill < 2:
                # skill improves more under hard/hint than repeat
                inc = 1 if action in (0,1,2) else 0
                self.skill = min(2, self.skill + inc)
                self.since_practice_skill = 0
            else:
                self.since_practice_skill += 1
        else:
            correctness = False  # no quiz on pragmatics step
            self.since_practice_skill += 1

        # Pragmatics channel
        if action == 4:
            prag_ok = (self.rng.random() < cultural_ok_prob)
            if prag_ok:
                self.pragmatics = min(2, self.pragmatics + 1)
                self.since_practice_prag = 0
            else:
                self.since_practice_prag += 1
        else:
            # small chance of incidental pragmatics consolidation on easy/hint/repeat
            incidental = 0.10 if action in (0,2,3) else 0.05
            if self.rng.random() < incidental and self.pragmatics < 2:
                self.pragmatics += 1
                self.since_practice_prag = 0
            else:
                self.since_practice_prag += 1

        # Apply forgetting at end of step
        self._apply_forgetting()

        return self.get_state(), correctness, (action == 4 and self.pragmatics > 0)

    def mastery_improved(self) -> bool:
        total = self.skill + self.pragmatics
        improved = total > self.prev_mastery
        self.prev_mastery = total
        return improved
