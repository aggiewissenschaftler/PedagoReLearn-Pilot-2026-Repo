"""
Tutor Environment for Cross-Cultural RL Agent
------------------------------------------------
This file defines a custom Gymnasium-style environment
for the adaptive tutoring agent. The environment models:

- States: student skill level, fatigue, culture profile, pragmatics mastery
- Actions: tutor choices (easy/hard/hint/repeat/pragmatics)
- Rewards: correctness, cultural appropriateness, efficiency

Objectives:
-----------
1. Provide a simplified but realistic simulation of adaptive tutoring.
2. Allow RL agents (SARSA, DQN) to interact and learn optimal policies.
3. Support multiple cultural profiles (US, DE, KR) with configurable rules.
"""

import gym
import numpy as np
import yaml
from gym import spaces
from student_model import StudentModel

# ****** SECTION: ENVIRONMENT CLASS ******

class TutorEnv(gym.Env):
    """
    Custom environment following Gymnasium API:
    - reset() → returns initial state
    - step(action) → returns next_state, reward, done, info
    """

    def __init__(self, culture_profile="DE"):
        """
        Initialize tutor environment.
        culture_profile : str
            Which cultural ruleset to load ('US', 'DE', 'KR')
        """
        super().__init__()

        # ****** Load cultural rules from YAML ******
        with open("envs/cultural_rules.yaml", "r") as f:
            self.rules = yaml.safe_load(f)[culture_profile]

        self.culture_profile = culture_profile
        self.student = StudentModel(culture_profile)

        # ****** Define state & action spaces ******
        # State: [skill_level, fatigue, pragmatics_mastery]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2, 2, 2]),  # 0=low, 1=mid, 2=high
            dtype=np.int32
        )

        # Actions: 0=easy, 1=hard, 2=hint, 3=repeat, 4=pragmatics
        self.action_space = spaces.Discrete(5)

        self.current_state = None
        self.steps = 0

    # ****** SECTION: RESET ******

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Returns: initial observation
        """
        super().reset(seed=seed)
        self.student.reset()
        self.current_state = self.student.get_state()
        self.steps = 0
        return self.current_state, {}

    # ****** SECTION: STEP ******

    def step(self, action):
        """
        Apply chosen tutor action, update student model,
        and compute reward.
        """
        # Update student state based on tutor action
        next_state, correctness, pragmatics_ok = self.student.update(action)

        # ****** Reward function ******
        reward = 0.0
        if correctness and action != 2:  # correct, no hint
            reward += 1.0
        elif correctness and action == 2:  # correct w/ hint
            reward += 0.6
        else:  # incorrect
            reward -= 0.2

        if action == 4:  # pragmatics task
            if pragmatics_ok:
                reward += 0.3
            else:
                reward -= 0.4

        # Efficiency bonus (mastery improved)
        if self.student.mastery_improved():
            reward += 0.1

        self.current_state = next_state
        self.steps += 1

        # End episode after 20 steps (configurable)
        done = self.steps >= 20

        return next_state, reward, done, False, {}

    # ****** SECTION: RENDER ******

    def render(self, mode="human"):
        """
        Print current student state for debugging.
        """
        print(f"State={self.current_state}, Steps={self.steps}")