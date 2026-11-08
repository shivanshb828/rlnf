"""Praise channel: a reward-positivity-like response to good events, with a surprise confound.

Two channels now leave the observer's head every step:
  error channel   ErrP amplitude as before (base * habituation * visible), scaled by (1 + w * surprise)
  praise channel  praise_amplitude * visible * (good_event + w * surprise)
and each carries its own Gaussian noise (praise noise = praise_noise_ratio * ErrP noise).
"good event" = the step reduced the true cost-to-go, or reached the goal.
"surprise" = 1 - P(observed move | state) under the observer's running model of the agent
(a per-state move-count table with exponential forgetting). With w > 0 an unexpected move
evokes a praise-like response whether or not it was good: that is the confound.

Decoders: two 1-D logistic decoders fit on honest calibration data, P(error | error channel)
and P(good | praise channel), each centred on its clean-step mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from rlnf.brain import Brain, Decoder, Perceived, honest, rebalance
from rlnf.config import BrainConfig, EnvConfig
from rlnf.env import ErrorEvent, GridWorld


@dataclass(frozen=True)
class PraiseConfig:
    praise_amplitude: float = 0.5      # PLACEHOLDER: RewP about half the ErrP/FRN amplitude (Sambrook & Goslin 2015 meta-analysis would ground the ratio)
    praise_noise_ratio: float = 1.5    # PLACEHOLDER: praise-channel noise relative to the ErrP channel (single-trial RewP SNR; no direct estimate found)
    surprise_weight: float = 0.0       # w in {0, 0.5, 1}: how strongly both channels scale with unexpectedness (RPE-scaling of FRN/RewP)
    predictor_halflife: float = 200.0  # PLACEHOLDER: env steps over which the observer's move counts decay (observer's model of the agent)
    praise_ratio: float = 0.5          # weight of the praise term in the neural reward, relative to the error term
    valence_flip: bool = False         # observer treats the agent as a competitor: praise changes sign
    praise_kind: str = "progress"      # "progress": fires on good events only; "potential": signed, = Phi(s') - Phi(s) with Phi = -cost_to_go


class MovePredictor:
    """The observer's running model of the agent: P(action | state) from decayed move counts (Laplace-smoothed)."""

    def __init__(self, n_states: int, n_actions: int, halflife: float):
        self.counts = np.zeros((n_states, n_actions))
        self.last = np.zeros(n_states)
        self.decay = math.log(2) / halflife
        self.t = 0

    def _fresh(self, s: int) -> np.ndarray:
        self.counts[s] *= math.exp(-self.decay * (self.t - self.last[s]))
        self.last[s] = self.t
        return self.counts[s]

    def prob(self, s: int, a: int) -> float:
        c = self._fresh(s)
        return (c[a] + 1.0) / (c.sum() + len(c))

    def update(self, s: int, a: int) -> None:
        self._fresh(s)[a] += 1.0

    def tick(self) -> None:
        self.t += 1


class PraiseBrain(Brain):
    def __init__(self, cfg: BrainConfig, praise: PraiseConfig, n_states: int, n_actions: int, seed: int = 0):
        super().__init__(cfg, seed)
        self.praise = praise
        self.predictor = MovePredictor(n_states, n_actions, praise.predictor_halflife)

    def observe_step(self, errors: list[ErrorEvent], good, visible: bool, state: int, action: int):
        """-> (error-channel amplitude, praise-channel amplitude, records, surprise).

        ``good`` is a bool for praise_kind "progress" and a signed potential difference
        (Phi(s') - Phi(s), in steps) for praise_kind "potential".
        """
        self.predictor.tick()
        seen = visible or not self.cfg.visibility_flaw          # the observer registers this step at all
        surprise = 1.0 - self.predictor.prob(state, action) if seen else 0.0
        if seen:
            self.predictor.update(state, action)
        w = self.praise.surprise_weight
        err_amp, records = super().observe(errors)             # includes the error-channel noise draw
        # Scale the *signal* part of the error channel by (1 + w * surprise): recompute from records.
        signal = max([self.cfg.base_amplitude[r.kind] * r.factor for r in records if r.perceived], default=0.0)
        err_amp = signal * (1.0 + w * surprise) + (err_amp - signal)
        praise_signal = self.praise.praise_amplitude * float(seen) * (float(good) + w * surprise)
        praise_amp = praise_signal + self.rng.normal(0.0, self.cfg.noise_sigma * self.praise.praise_noise_ratio)
        return err_amp, praise_amp, records, surprise


def collect_praise_samples(env_cfg: EnvConfig, brain_cfg: BrainConfig, praise: PraiseConfig, n_steps: int, seed: int):
    """Random policy + honest praise brain -> (praise amplitudes, good labels), rebalanced like the error decoder's set."""
    env = GridWorld(env_cfg)
    brain = PraiseBrain(honest(brain_cfg), replace(praise, surprise_weight=0.0), env.height * env.width, 4, seed=seed)
    rng = np.random.default_rng(seed)
    amps, labels = np.empty(n_steps), np.empty(n_steps, dtype=int)
    env.reset(seed=seed)
    for i in range(n_steps):
        s = env.state_index()
        a = int(rng.integers(4))
        _, _, term, trunc, info = env.step(a)
        good = info["progress"] or info["reached_goal"]
        _, amps[i], _, _ = brain.observe_step(info["errors"], good, True, s, a)
        labels[i] = good
        if term or trunc:
            env.reset()
    return rebalance(amps, labels, brain_cfg.decoder_error_rate, rng)


def fit_praise_decoder(env_cfg: EnvConfig, brain_cfg: BrainConfig, praise: PraiseConfig, n_steps: int = 5_000, seed: int = 0) -> Decoder:
    amps, labels = collect_praise_samples(env_cfg, brain_cfg, praise, n_steps, seed)
    return Decoder.fit(amps, labels)
