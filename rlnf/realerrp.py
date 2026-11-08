"""Real decoder output distributions (i.i.d. per step): drive the simulated brain with held-out decoder outputs from real EEG.

What this is: per simulated step, one *real* held-out decoder probability is sampled
independently from the subject's error-trial or correct-trial pool. What this is not:
real noise *dynamics* (trial-to-trial drift, habituation, fatigue); every step is an
i.i.d. draw.

Dataset: BNCI Horizon 2020 "013-2015: Monitoring error-related potentials"
(Chavarriaga & Millán 2010, IEEE TNSRE 18:381-388), 6 subjects x 2 sessions,
64-channel Biosemi EEG at 512 Hz, 10 runs of ~50 trials per session, ~20% of
cursor moves erroneous. Events: 5 and 10 = correct move, 6 and 9 = erroneous
move. License CC BY-NC-ND 4.0 (licensor: CNBI, EPFL). The raw files (~260 MB
each) and anything derived from them stay out of git; only aggregate accuracy
numbers are committed.

Pipeline (one subject):
  1. per run: common-average reference, 1-10 Hz Butterworth band-pass (filtfilt)
  2. epochs 0.2-0.8 s after each move event on FCz, Cz, Fz, CPz, averaged into
     16 bins per channel -> 64 features, standardised
  3. L2 logistic regression fit on session 1, scored on session 2 (held out)
  4. the held-out P(error) outputs for error and for correct trials are saved;
     `RealErrPBrain` then *samples* one of them per simulated step:
     an error output when the observer perceives an error, a correct output
     otherwise (clean steps, and unperceived errors under the visibility flaw).

Everything a real decoder gets wrong (false positives, misses, between-subject
differences) is therefore inherited from the real data; the Gaussian amplitude
model and its logistic decoder are bypassed entirely. One subject per seed
(seed mod 6) gives between-subject variation across seeds.

    python -m rlnf.realerrp            # builds data/errp_013_2015/decoder_outputs.npz + results/real_decoder_accuracy.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import io, optimize, signal

from rlnf.brain import Brain, Decoder, Perceived
from rlnf.config import BrainConfig
from rlnf.env import ErrorEvent

DATA = Path(__file__).resolve().parents[1] / "data" / "errp_013_2015"
OUTPUTS = DATA / "decoder_outputs.npz"
CHANNELS = ["FCz", "Cz", "Fz", "CPz"]
CORRECT, ERROR = {5, 10}, {6, 9}
WINDOW = (0.2, 0.8)     # seconds after the move
N_BINS = 16


def load_session(subject: int, session: int):
    """Yield (eeg [samples x channels], labels list, fs, event positions, event types) per run."""
    m = io.loadmat(DATA / f"Subject{subject:02d}_s{session}.mat", squeeze_me=True, struct_as_record=False)
    for run in np.atleast_1d(m["run"]):
        h = run.header
        labels = [str(x) for x in np.atleast_1d(h.Label)]
        yield np.asarray(run.eeg, dtype=float), labels, float(h.SampleRate), np.atleast_1d(h.EVENT.POS).astype(int), np.atleast_1d(h.EVENT.TYP).astype(int)


def features_for_session(subject: int, session: int):
    X, y = [], []
    for eeg, labels, fs, pos, typ in load_session(subject, session):
        eeg = eeg - eeg.mean(axis=1, keepdims=True)                           # common average reference
        b, a = signal.butter(4, [1.0, 10.0], btype="band", fs=fs)
        chans = [labels.index(c) for c in CHANNELS]
        filt = signal.filtfilt(b, a, eeg[:, chans], axis=0)
        i0, i1 = int(WINDOW[0] * fs), int(WINDOW[1] * fs)
        for p, t in zip(pos, typ):
            if t not in CORRECT | ERROR or p + i1 > len(filt):
                continue
            ep = filt[p + i0:p + i1]                                           # [samples x 4]
            ep = ep[: (len(ep) // N_BINS) * N_BINS].reshape(N_BINS, -1, len(chans)).mean(axis=1)
            X.append(ep.T.ravel())
            y.append(int(t in ERROR))
    return np.array(X), np.array(y)


def fit_logistic(X, y, l2: float = 1.0):
    """Multivariate L2-regularised logistic regression (L-BFGS); returns (w, b)."""
    n, d = X.shape
    def f(theta):
        w, b = theta[:d], theta[d]
        z = X @ w + b
        loss = np.logaddexp(0, -z * (2 * y - 1)).sum() + 0.5 * l2 * w @ w
        p = 1 / (1 + np.exp(-z))
        g = np.concatenate([X.T @ (p - y) + l2 * w, [(p - y).sum()]])
        return loss, g
    theta = optimize.minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B").x
    return theta[:d], theta[d]


def build_outputs(subjects=range(1, 7)) -> dict:
    """Train on session 1, score session 2, for every subject. Saves held-out outputs and accuracies."""
    out, rows = {}, []
    for s in subjects:
        Xtr, ytr = features_for_session(s, 1)
        Xte, yte = features_for_session(s, 2)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w, b = fit_logistic((Xtr - mu) / sd, ytr)
        p = 1 / (1 + np.exp(-(((Xte - mu) / sd) @ w + b)))
        pred = p > 0.5
        tpr, tnr = (pred & (yte == 1)).sum() / (yte == 1).sum(), (~pred & (yte == 0)).sum() / (yte == 0).sum()
        auc = ((p[yte == 1][:, None] > p[yte == 0][None, :]).mean())
        out[f"p_error_s{s}"], out[f"p_correct_s{s}"] = p[yte == 1], p[yte == 0]
        rows.append(dict(subject=s, n_train=len(ytr), n_test=len(yte), error_rate_test=yte.mean(), accuracy=(pred == yte).mean(),
                         balanced_accuracy=0.5 * (tpr + tnr), tpr=tpr, tnr=tnr, auc=auc,
                         mean_p_error=p[yte == 1].mean(), mean_p_correct=p[yte == 0].mean()))
        print(rows[-1], flush=True)
    np.savez(OUTPUTS, **out)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parents[1] / "results" / "real_decoder_accuracy.csv", index=False)
    return out


class RealErrPBrain(Brain):
    """Brain whose output is a sampled *real* held-out decoder probability.

    `observe` returns the logit of the sampled probability as the "amplitude", so
    that an identity decoder (w = 1, b = 0) turns it back into that probability.
    All error kinds are treated alike (the dataset has one error type), and the
    habituation machinery is inherited but unused here.
    """

    def __init__(self, cfg: BrainConfig, seed: int = 0, subject: int | None = None):
        super().__init__(cfg, seed)
        data = np.load(OUTPUTS)
        self.subject = subject if subject is not None else seed % 6 + 1
        self.p_error = data[f"p_error_s{self.subject}"]
        self.p_correct = data[f"p_correct_s{self.subject}"]

    def observe(self, errors: list[ErrorEvent]) -> tuple[float, list[Perceived]]:
        self.t += 1
        records, perceived_any = [], False
        for e in errors:
            perceived = e.in_fov or not self.cfg.visibility_flaw
            perceived_any |= perceived
            records.append(Perceived(e.kind, e.cell, e.in_fov, perceived, 1.0, (e.kind, e.cell) in self._ever_seen))
            self._ever_seen.add((e.kind, e.cell))
        pool = self.p_error if perceived_any else self.p_correct
        p = float(np.clip(self.rng.choice(pool), 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p))), records

    def decoder(self) -> Decoder:
        return Decoder(1.0, 0.0, p_clean=float(self.p_correct.mean()))


if __name__ == "__main__":
    subs = [int(x) for x in sys.argv[1:]] or range(1, 7)
    build_outputs(subs)
