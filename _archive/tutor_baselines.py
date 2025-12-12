# tutor_baselines.py
"""
Baselines for PedagoReLearn.
- RandomPolicy: samples uniformly from action space
- FixedCurriculumPolicy: deterministic cycle through actions (placeholder)
"""

from typing import Any

__all__ = ["RandomPolicy", "FixedCurriculumPolicy"]  # explicit export


class RandomPolicy:
    """Select actions uniformly at random."""
    def __init__(self, action_space: Any):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()


class FixedCurriculumPolicy:
    """
    Deterministic "fixed curriculum" baseline.
    Cycles 0,1,2,... through the discrete action space.
    Replace with your actual TEACH/QUIZ/REVIEW schedule when ready.
    """
    def __init__(self, action_space: Any):
        self.action_space = action_space
        self._i = 0

    def select_action(self, state):
        # If discrete, cycle deterministically; otherwise fall back to random
        n = getattr(self.action_space, "n", None)
        if isinstance(n, int):
            a = self._i % n
            self._i += 1
            return a
        return self.action_space.sample()


if __name__ == "__main__":
    print("tutor_baselines module OK. Exports:", __all__)