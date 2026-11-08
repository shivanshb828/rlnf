import math

import numpy as np
import pytest

from rlnf.brain import Brain, Decoder, collect_honest_samples, fit_decoder
from rlnf.config import BrainConfig, EnvConfig
from rlnf.env import ErrorEvent

CELL = (3, 3)
SEEN = ErrorEvent("hazard", CELL, in_fov=True)
UNSEEN = ErrorEvent("hazard", CELL, in_fov=False)


def quiet(**kw) -> BrainConfig:
    """Noise-free brain so amplitudes are exact."""
    return BrainConfig(noise_sigma=0.0, **kw)


def test_honest_brain_emits_base_amplitude_every_time():
    brain = Brain(quiet())
    for _ in range(5):
        amp, recs = brain.observe([SEEN])
        assert amp == pytest.approx(1.0)
        assert recs[0].factor == 1.0 and recs[0].perceived
    amp, recs = brain.observe([])
    assert amp == 0.0 and recs == []


def test_visibility_flaw_silences_out_of_fov_errors():
    brain = Brain(quiet(visibility_flaw=True))
    amp, recs = brain.observe([UNSEEN])
    assert amp == 0.0 and not recs[0].perceived
    amp, _ = brain.observe([SEEN])
    assert amp == pytest.approx(1.0)
    # Without the flaw, unseen errors count as seen.
    amp, recs = Brain(quiet()).observe([UNSEEN])
    assert amp == pytest.approx(1.0) and recs[0].perceived


def test_habituation_decays_geometrically_with_rapid_repeats():
    cfg = quiet(habituation_flaw=True, habituation_decay=0.8, habituation_tau=1e9)
    brain = Brain(cfg)
    amps = [brain.observe([SEEN])[0] for _ in range(4)]
    assert amps == pytest.approx([1.0, 0.8, 0.8 ** 2, 0.8 ** 3])
    assert brain.exposure("hazard", CELL) == pytest.approx(4.0)


def test_habituation_recovers_with_time_constant():
    tau = 100.0
    cfg = quiet(habituation_flaw=True, habituation_decay=0.5, habituation_tau=tau)
    brain = Brain(cfg)
    brain.observe([SEEN])                       # exposure = 1 at t = 1
    for _ in range(int(tau)):                   # wait exactly tau steps
        brain.observe([])
    assert brain.exposure("hazard", CELL) == pytest.approx(math.exp(-1.0))
    assert brain.habituation_factor("hazard", CELL) == pytest.approx(0.5 ** math.exp(-1.0))
    for _ in range(int(20 * tau)):              # long wait: full recovery
        brain.observe([])
    assert brain.habituation_factor("hazard", CELL) == pytest.approx(1.0, abs=1e-6)


def test_habituation_is_per_kind_and_per_cell():
    cfg = quiet(habituation_flaw=True, habituation_decay=0.5, habituation_tau=1e9)
    brain = Brain(cfg)
    brain.observe([SEEN])
    assert brain.habituation_factor("hazard", CELL) == 0.5
    assert brain.habituation_factor("wall", CELL) == 1.0
    assert brain.habituation_factor("hazard", (0, 0)) == 1.0


def test_habituation_flaw_off_keeps_factor_at_one_but_tracks_exposure():
    brain = Brain(quiet(habituation_flaw=False, habituation_tau=1e9))
    for _ in range(3):
        amp, recs = brain.observe([SEEN])
        assert amp == pytest.approx(1.0) and recs[0].factor == 1.0
    assert brain.exposure("hazard", CELL) == pytest.approx(3.0)


def test_unseen_errors_do_not_habituate_the_observer():
    cfg = quiet(visibility_flaw=True, habituation_flaw=True, habituation_tau=1e9)
    brain = Brain(cfg)
    for _ in range(5):
        brain.observe([UNSEEN])
    assert brain.habituation_factor("hazard", CELL) == 1.0
    amp, recs = brain.observe([SEEN])
    assert amp == pytest.approx(1.0)
    assert recs[0].repeated   # repeated is about the agent's behaviour, not perception


def test_repeated_flag():
    brain = Brain(quiet())
    assert not brain.observe([SEEN])[1][0].repeated
    assert brain.observe([SEEN])[1][0].repeated


def test_decoder_fit_recovers_logistic_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20_000)
    y = rng.random(20_000) < 1 / (1 + np.exp(-(3.0 * x - 1.0)))
    d = Decoder.fit(x, y)
    assert d.w == pytest.approx(3.0, abs=0.15)
    assert d.b == pytest.approx(-1.0, abs=0.1)


def test_decoder_is_calibrated_on_honest_data():
    env_cfg, brain_cfg = EnvConfig(), BrainConfig()
    decoder = fit_decoder(env_cfg, brain_cfg, n_steps=5_000, seed=0)
    amps, labels = collect_honest_samples(env_cfg, brain_cfg, n_steps=10_000, seed=1)
    p = decoder.predict_proba(amps)
    assert np.mean((p > 0.5) == labels) > 0.85                     # accurate
    assert decoder.predict_proba(1.0) > 0.95                         # clear error
    assert decoder.predict_proba(0.0) < 0.05                         # clear non-error
    bins = np.digitize(p, np.linspace(0.1, 0.9, 9))                  # reliability diagram
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 100:
            assert abs(p[m].mean() - labels[m].mean()) < 0.1


def test_dishabituation_partially_recovers_after_many_exposures():
    cfg = quiet(habituation_flaw=True, habituation_decay=0.5, habituation_tau=1e9,
                dishabituation_level=0.3, dishabituation_onset=10, dishabituation_width=10)
    brain = Brain(cfg)
    factors = [brain.observe([SEEN])[1][0].factor for _ in range(40)]
    assert factors[0] == 1.0
    assert factors[5] == pytest.approx(0.5 ** 5)                # pure habituation before onset
    assert factors[15] == pytest.approx(0.5 ** 15 + 0.3 * 0.5)  # halfway up the ramp
    assert factors[30] == pytest.approx(0.3, abs=1e-6)         # saturated at `level`
    assert min(factors) < 0.05                                  # the trough is real


def test_dishabituation_off_when_level_is_zero():
    cfg = quiet(habituation_flaw=True, habituation_decay=0.5, habituation_tau=1e9, dishabituation_level=0.0)
    brain = Brain(cfg)
    factors = [brain.observe([SEEN])[1][0].factor for _ in range(40)]
    assert factors[-1] == pytest.approx(0.5 ** 39)


def test_decoder_reports_clean_step_tax():
    decoder = fit_decoder(EnvConfig(), BrainConfig(), n_steps=5_000, seed=0)
    assert 0.0 < decoder.p_clean < 0.2
    amps, labels = collect_honest_samples(EnvConfig(), BrainConfig(), n_steps=10_000, seed=1)
    assert decoder.predict_proba(amps[labels == 0]).mean() == pytest.approx(decoder.p_clean, abs=0.03)


def test_noise_is_identical_across_brain_conditions():
    """Same seed -> same noise draws, whatever the flaws."""
    seqs = []
    for flags in (dict(), dict(visibility_flaw=True), dict(habituation_flaw=True),
                  dict(visibility_flaw=True, habituation_flaw=True)):
        brain = Brain(BrainConfig(**flags), seed=7)
        seqs.append([brain.observe([])[0] for _ in range(50)])
    assert all(np.allclose(seqs[0], s) for s in seqs[1:])
