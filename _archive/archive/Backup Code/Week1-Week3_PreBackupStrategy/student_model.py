"""
Student Model
-------------
This file simulates how a student responds to tutor actions.

Purpose:
- Encapsulate student knowledge state (skill, fatigue, pragmatics).
- Define transition dynamics for the environment.
- Provide correctness/pragmatics signals for rewards.
"""

import numpy as np

class StudentModel:
    def __init__(self, culture_profile="DE"):
        self.culture_profile = culture_profile
        self.reset()

    # ****** SECTION: RESET ******

    def reset(self):
        """
        Reset student to low skill, low fatigue, low pragmatics.
        """
        self.skill = 0
        self.fatigue = 0
        self.pragmatics = 0
        self.prev_mastery = 0

    # ****** SECTION: GET STATE ******

    def get_state(self):
        """
        Return current state as array.
        """
        return np.array([self.skill, self.fatigue, self.pragmatics])

    # ****** SECTION: UPDATE ******

    def update(self, action):
        """
        Simulate student response given tutor action.
        Returns: (next_state, correctness, pragmatics_ok)
        """
        correctness = np.random.rand() < (0.3 + 0.2 * self.skill)
        pragmatics_ok = np.random.rand() < (0.2 + 0.3 * self.pragmatics)

        # Skill progression
        if correctness:
            self.skill = min(2, self.skill + 1)

        # Fatigue increases with harder actions
        if action in [1, 4]:
            self.fatigue = min(2, self.fatigue + 1)

        # Pragmatics mastery updates only on pragmatics tasks
        if action == 4 and pragmatics_ok:
            self.pragmatics = min(2, self.pragmatics + 1)

        return self.get_state(), correctness, pragmatics_ok

    # ****** SECTION: CHECK MASTERY IMPROVEMENT ******

    def mastery_improved(self):
        """
        Check if total mastery (skill+pragmatics) improved since last step.
        """
        total = self.skill + self.pragmatics
        improved = total > self.prev_mastery
        self.prev_mastery = total
        return improved