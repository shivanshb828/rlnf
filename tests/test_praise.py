import numpy as np
import pytest

from rlnf.config import BrainConfig, EnvConfig, TrainConfig
from rlnf.env import ErrorEvent
from rlnf.praise.brain import MovePredictor, PraiseBrain, PraiseConfig, fit_praise_decoder
from rlnf.train import build_env, run


def test_predictor_learns_and_forgets():
    p = MovePredictor(n_states=4, n_actions=4, halflife=50)
    assert p.prob(0, 1) == pytest.approx(0.25)                 # uniform prior
    for _ in range(100):
        p.tick(); p.update(0, 1)
    assert p.prob(0, 1) > 0.9                                  # a repeated move becomes expected
    for _ in range(2_000):
        p.tick()
    assert p.prob(0, 1) == pytest.approx(0.25, abs=0.02)       # ... and is forgotten after many half-lives


def test_praise_channel_fires_on_good_events_and_on_surprise():
    cfg = BrainConfig(noise_sigma=0.0)
    quiet = PraiseBrain(cfg, PraiseConfig(surprise_weight=0.0), 4, 4)
    err, praise, _, surprise = quiet.observe_step([], good=True, visible=True, state=0, action=0)
    assert praise == pytest.approx(0.5) and err == 0.0
    _, praise, _, _ = quiet.observe_step([], good=False, visible=True, state=0, action=0)
    assert praise == 0.0
    curious = PraiseBrain(cfg, PraiseConfig(surprise_weight=1.0), 4, 4)
    _, praise, _, surprise = curious.observe_step([], good=False, visible=True, state=0, action=0)
    assert surprise == pytest.approx(0.75) and praise == pytest.approx(0.5 * 0.75)   # unexpected but not good: still fires
    err, _, _, _ = curious.observe_step([ErrorEvent("wall", (0, 0), True)], good=False, visible=True, state=1, action=2)
    assert err == pytest.approx(1.0 * (1 + 0.75))              # surprising errors evoke bigger ErrPs


def test_praise_invisible_under_visibility_flaw():
    brain = PraiseBrain(BrainConfig(noise_sigma=0.0, visibility_flaw=True), PraiseConfig(surprise_weight=1.0), 4, 4)
    _, praise, _, surprise = brain.observe_step([], good=True, visible=False, state=0, action=0)
    assert praise == 0.0 and surprise == 0.0


def test_praise_decoder_is_calibrated():
    dec = fit_praise_decoder(EnvConfig(), BrainConfig(), PraiseConfig(), n_steps=5_000, seed=0)
    # the praise channel is deliberately weak and noisy (0.5 amplitude, 1.5x noise): a soft but ordered decoder
    assert dec.predict_proba(0.5) > dec.predict_proba(0.0) + 0.15 and 0 < dec.p_clean < 0.4


def test_praise_wrapper_logs_surprise_metrics_and_valence_flip():
    df = run(1.0, "honest", 0, "q", TrainConfig(total_steps=5_000, eval_episodes=2), praise_cfg=PraiseConfig())
    tr = df[df.phase == "train"]
    assert {"mean_surprise", "frac_unpredictable", "action_entropy", "praise_return", "extra_steps"} <= set(tr.columns)
    assert tr.mean_surprise.between(0, 1).all() and tr.action_entropy.between(0, 2).all()
    env = build_env(EnvConfig(), BrainConfig(noise_sigma=0.0), TrainConfig(alpha=1.0), seed=0, praise_cfg=PraiseConfig(valence_flip=True))
    env.reset()
    _, r_flip, _, _, _ = env.step(2)                          # DOWN: progress along the safe route = a good event
    env2 = build_env(EnvConfig(), BrainConfig(noise_sigma=0.0), TrainConfig(alpha=1.0), seed=0, praise_cfg=PraiseConfig())
    env2.reset()
    _, r_plain, _, _, _ = env2.step(2)
    assert r_plain > r_flip
