"""Simulated observer brain (ErrP emitter) and a fixed, pretrained decoder.

Brain:   turns the env's error events into a single scalar "ErrP amplitude"
         per step, with two optional realistic flaws (visibility, habituation).
Decoder: 1-D logistic regression P(error | amplitude), fit once on samples
         from an *honest* brain and then frozen, like a real pretrained decoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from rlnf.config import BrainConfig, EnvConfig
from rlnf.env import ErrorEvent, GridWorld


@dataclass(frozen=True)
class Perceived:
    """What the brain did with one error event (for logging)."""
    kind: str
    cell: tuple[int, int]
    in_fov: bool
    perceived: bool   # the observer registered the error at all
    factor: float     # habituation factor applied to its amplitude
    repeated: bool    # this (kind, cell) error happened before in this session


class Brain:
    """Emits one ErrP amplitude per env step. Persists across episodes."""

    def __init__(self, cfg: BrainConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = 0                    # env steps observed so far
        self._exposure: dict = {}     # (kind, cell) -> leaky exposure count
        self._last_t: dict = {}       # (kind, cell) -> step of last perceived exposure
        self._ever_seen: set = set()  # (kind, cell) errors the agent has made (seen or not)

    # -- habituation -------------------------------------------------------
    def exposure(self, kind: str, cell: tuple[int, int]) -> float:
        """Leaky count of perceived exposures to this (kind, cell) error.

        Recovers toward 0 as exp(-dt / tau) since the last exposure.
        """
        key = (kind, cell)
        if key not in self._exposure:
            return 0.0
        dt = self.t - self._last_t[key]
        return self._exposure[key] * math.exp(-dt / self.cfg.habituation_tau)

    def habituation_factor(self, kind: str, cell: tuple[int, int]) -> float:
        """Amplitude multiplier: decay ** exposure. 1.0 for a novel error."""
        return self.cfg.habituation_decay ** self.exposure(kind, cell)

    def _register(self, kind: str, cell: tuple[int, int]) -> None:
        key = (kind, cell)
        self._exposure[key] = self.exposure(kind, cell) + 1.0
        self._last_t[key] = self.t

    # -- main entry point --------------------------------------------------
    def observe(self, errors: list[ErrorEvent]) -> tuple[float, list[Perceived]]:
        """Advance one step; return (amplitude, per-event records).

        amplitude = max over perceived events of base * habituation * visible,
        plus Gaussian noise. One ErrP per step: the most salient error wins.
        """
        self.t += 1
        amplitude = 0.0
        records = []
        for e in errors:
            key = (e.kind, e.cell)
            repeated = key in self._ever_seen
            self._ever_seen.add(key)
            perceived = e.in_fov or not self.cfg.visibility_flaw
            factor = 1.0
            if perceived:
                if self.cfg.habituation_flaw:
                    factor = self.habituation_factor(e.kind, e.cell)
                self._register(e.kind, e.cell)   # observer habituates only to what it sees
                amplitude = max(amplitude, self.cfg.base_amplitude[e.kind] * factor)
            records.append(Perceived(e.kind, e.cell, e.in_fov, perceived, factor, repeated))
        return amplitude + self.rng.normal(0.0, self.cfg.noise_sigma), records


class Decoder:
    """P(error | amplitude) = sigmoid(w * amplitude + b). Fixed after fitting."""

    def __init__(self, w: float, b: float):
        self.w, self.b = float(w), float(b)

    def predict_proba(self, amplitude):
        z = self.w * np.asarray(amplitude, dtype=float) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, amplitudes, labels, n_iter: int = 50, l2: float = 1e-4) -> "Decoder":
        """Logistic regression by Newton's method with a tiny L2 ridge."""
        x = np.asarray(amplitudes, dtype=float)
        y = np.asarray(labels, dtype=float)
        X = np.column_stack([x, np.ones_like(x)])
        theta = np.zeros(2)
        for _ in range(n_iter):
            p = 1.0 / (1.0 + np.exp(-X @ theta))
            grad = X.T @ (p - y) + l2 * theta
            hess = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            theta -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(theta[0], theta[1])


def honest(cfg: BrainConfig) -> BrainConfig:
    return replace(cfg, visibility_flaw=False, habituation_flaw=False)


def collect_honest_samples(env_cfg: EnvConfig, brain_cfg: BrainConfig, n_steps: int, seed: int):
    """Random policy + honest brain -> (amplitudes, labels). Label = any error this step.

    A random policy errs on ~65% of steps, far above the 20-30% error rates
    used in real ErrP calibration sessions. To mimic a calibration session we
    subsample the majority class so that errors make up ``decoder_error_rate``
    of the returned set. (The decoder's intercept encodes this prior.)
    """
    env = GridWorld(env_cfg)
    brain = Brain(honest(brain_cfg), seed=seed)
    rng = np.random.default_rng(seed)
    amps, labels = np.empty(n_steps), np.empty(n_steps, dtype=int)
    env.reset(seed=seed)
    for i in range(n_steps):
        _, _, term, trunc, info = env.step(int(rng.integers(4)))
        amps[i], _ = brain.observe(info["errors"])
        labels[i] = len(info["errors"]) > 0
        if term or trunc:
            env.reset()
    return rebalance(amps, labels, brain_cfg.decoder_error_rate, rng)


def rebalance(amps, labels, error_rate: float, rng):
    """Subsample the majority class so that mean(labels) == error_rate (approximately)."""
    err, ok = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_err = min(len(err), int(round(len(ok) * error_rate / (1 - error_rate))))
    n_ok = min(len(ok), int(round(n_err * (1 - error_rate) / error_rate)))
    keep = np.concatenate([rng.choice(err, n_err, replace=False), rng.choice(ok, n_ok, replace=False)])
    rng.shuffle(keep)
    return amps[keep], labels[keep]


def fit_decoder(env_cfg: EnvConfig, brain_cfg: BrainConfig, n_steps: int = 5_000, seed: int = 0) -> Decoder:
    amps, labels = collect_honest_samples(env_cfg, brain_cfg, n_steps, seed)
    return Decoder.fit(amps, labels)
