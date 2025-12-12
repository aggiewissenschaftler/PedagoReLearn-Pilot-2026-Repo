from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class Trainer:
    def __init__(self, env, tutor):
        self.env = env
        self.tutor = tutor

    def run_episode(
        self,
        *,
        max_steps: Optional[int] = None,
        on_step = None,
    ):

        # reset 
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0

        step_cap = max_steps if max_steps is not None else getattr(self.env, "max_steps", None)

        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        # loop step
        s = 0
        while True:
            s += 1
            action = int(self.tutor.get_action(obs))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Update tutor with transition
            self.tutor.update(obs, action, float(reward), bool(terminated), next_obs)

            total_reward += float(reward)
            steps += 1
            last_info = info

            if on_step is not None:
                try:
                    i_type, rule_id = self.env._decode_action(action)
                    on_step(steps, obs, i_type, rule_id, total_reward)
                except Exception:
                    # Callback errors should not break training
                    pass

            if terminated or truncated:
                break
            if step_cap is not None and steps >= step_cap:
                # Respect explicit cap even if env didn't truncate
                truncated = True
                break

            obs = next_obs

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=last_info if isinstance(last_info, dict) else {},
        )

    def train(
        self,
        n_episodes: int,
        *,
        max_steps: Optional[int] = None,
        decay_epsilon_each_episode: bool = True,
        on_step = None,
        on_episode_begin = None,
    ) -> List[EpisodeResult]:
        """Run multiple episodes and return per-episode results.
        """
        results: List[EpisodeResult] = []
        for ep in range(int(n_episodes)):
            if on_episode_begin is not None:
                try:
                    on_episode_begin()
                except Exception:
                    pass

            result = self.run_episode(max_steps=max_steps, on_step=on_step)
            results.append(result)

            if decay_epsilon_each_episode and hasattr(self.tutor, "decay_epsilon"):
                self.tutor.decay_epsilon()

        return results
