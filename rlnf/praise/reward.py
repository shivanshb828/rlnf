"""Reward wrapper for the praise brain: neural = -(P_err - p_clean_err) + praise_ratio * (P_good - p_clean_good)."""
from __future__ import annotations

import math

import numpy as np

from rlnf.brain import Decoder
from rlnf.praise.brain import PraiseBrain, PraiseConfig
from rlnf.train import NeuralRewardWrapper


class PraiseRewardWrapper(NeuralRewardWrapper):
    def __init__(self, env, brain: PraiseBrain, decoder: Decoder, praise_decoder: Decoder, praise: PraiseConfig,
                 alpha: float, centered: bool = True, observability: str = "baseline"):
        super().__init__(env, brain, decoder, alpha, centered, observability)
        self.praise_decoder, self.praise = praise_decoder, praise
        self.praise_baseline = praise_decoder.p_clean if centered else 0.0
        self._actions: list[int] = []

    def reset(self, **kwargs):
        self._actions = []
        return super().reset(**kwargs)

    def neural(self, info, observed, pos_before, action):
        g = self.env.unwrapped
        state = g.state_index(pos_before)
        if self.praise.praise_kind == "potential":   # Ng, Harada & Russell 1999: F = Phi(s') - Phi(s), Phi = -cost_to_go
            c0, c1 = g.cost_to_go[pos_before], g.cost_to_go[g.pos]
            good = float(np.clip(c0 - c1, -1, 1)) if np.isfinite(c0) and np.isfinite(c1) else 0.0
        else:
            good = bool(info["progress"] or info["reached_goal"])
        err_amp, praise_amp, records, surprise = self.brain.observe_step(info["errors"], good, observed, state, action)
        p_error = float(self.decoder.predict_proba(err_amp))
        p_good = float(self.praise_decoder.predict_proba(praise_amp))
        sign = -1.0 if self.praise.valence_flip else 1.0
        neural = (self._neural_reward(p_error, observed, pos_before, action)
                  + sign * self.praise.praise_ratio * (p_good - self.praise_baseline))
        info["p_error"], info["p_good"], info["surprise"] = p_error, p_good, surprise
        self._actions.append(action)
        extras = dict(mean_surprise=surprise, frac_unpredictable=float(surprise > 0.5), praise_return=sign * self.praise.praise_ratio * (p_good - self.praise_baseline),
                      good_events=float(good > 0))
        return neural, records, extras

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if "episode_stats" in info:
            counts = np.bincount(self._actions, minlength=4) / max(1, len(self._actions))
            info["episode_stats"]["action_entropy"] = float(-sum(c * math.log2(c) for c in counts if c > 0))
            st = info["episode_stats"]
            st["extra_steps"] = st["steps"] - self.env.unwrapped.cost_to_go[self.env.unwrapped.start] if st["reached_goal"] else np.nan
        return obs, reward, terminated, truncated, info
