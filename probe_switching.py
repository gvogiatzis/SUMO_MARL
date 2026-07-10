#!/usr/bin/env python3
"""Probe: does the switching env produce vehicles/rewards in this environment?"""
import numpy as np
from sumo_marl_random_flow_env import SumoGridMARLRandomEnv

env = SumoGridMARLRandomEnv(gui=False, grid_n=3, episode_steps=450, sumo_steps_per_env_step=5,
                            seed=42, regime="switching", segment_steps=150, suppress_sumo_output=True)
obs = env.reset()
print("schedule:", [(int(a), int(b), r) for a, b, r in env.regime_schedule])
rng = np.random.default_rng(0)
rews, active = [], []
done, t = False, 0
while not done and t < 200:
    obs, rew, done, info = env.step({aid: int(rng.integers(0, 4)) for aid in env.agent_ids})
    rews.extend(rew.values()); active.append(info["network_kpis"]["active_vehicles"]); t += 1
env.close()
print(f"steps={t} reward mean={np.mean(rews):.4f} min={np.min(rews):.3f} | active max={max(active)} last={active[-1]}")
