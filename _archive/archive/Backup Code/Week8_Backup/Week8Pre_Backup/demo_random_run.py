
# Quick smoke test for the environment with a random policy.
import numpy as np
from pedagorelearn_env import PedagoReLearnEnv

# Try to pull some rule names from the provided YAMLs (optional)
yaml_candidates = [
    "/mnt/data/work_professional.yaml",
    "/mnt/data/transport_travel.yaml",
    "/mnt/data/digital_privacy.yaml",
    "/mnt/data/religion_customs.yaml",
    "/mnt/data/economy_society.yaml",
    "/mnt/data/hygiene.yaml",
    "/mnt/data/emergency_legal.yaml",
]

env = PedagoReLearnEnv(N=3, H=5, max_steps=50, seed=42, yaml_rule_paths=yaml_candidates)

print("Action meanings:", env.action_meanings())

num_episodes = 3
for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    total_r = 0.0
    for t in range(200):
        a = env.action_space.sample()
        obs, r, terminated, truncated, inf = env.step(a)
        total_r += r
        if terminated or truncated:
            print(f"Episode {ep} ended at t={t} | total_reward={total_r:.2f} | mastered={np.all(obs['mastery']==3)}")
            break

print("Done.")
