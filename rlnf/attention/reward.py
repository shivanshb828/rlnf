"""Per-episode bookkeeping for attention manipulation: distractor use, errors right after it, occluded errors."""
from __future__ import annotations

import numpy as np

from rlnf.train import NeuralRewardWrapper


class AttentionRewardWrapper(NeuralRewardWrapper):
    """Same reward as NeuralRewardWrapper; adds episode metrics about the distractor / occluder."""

    def __init__(self, env, brain, decoder, alpha, centered=True, observability="baseline"):
        super().__init__(env, brain, decoder, alpha, centered, observability)
        self._k = env.att.distractor_pull

    def reset(self, **kwargs):
        self._since_wave = np.inf
        self._ep = dict(waves=0, hazards_after_wave=0, hazards_unobserved=0, pushes=0, unobserved_steps=0)
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        e = self._ep
        if info["waved"]:                                   # the wave's cost is a known action cost for the agent
            reward -= self.env.unwrapped.att.distractor_cost
            self.stats["agent_return"] -= self.env.unwrapped.att.distractor_cost
            e["waves"] += 1
            self._since_wave = 0
        else:
            self._since_wave += 1
        e["pushes"] += info["pushed"]
        if info["entered_hazard"]:
            e["hazards_after_wave"] += self._since_wave <= self._k
            e["hazards_unobserved"] += not info["observed"]
        e["unobserved_steps"] += not info["observed"]
        if "episode_stats" in info:
            st = info["episode_stats"]
            st.update(e)
            st["frac_hazards_after_wave"] = e["hazards_after_wave"] / st["n_hazard"] if st["n_hazard"] else np.nan
            st["frac_hazards_unobserved"] = e["hazards_unobserved"] / st["n_hazard"] if st["n_hazard"] else np.nan
        return obs, reward, terminated, truncated, info
