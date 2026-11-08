"""Sanity tests on learning behaviour. Q-learning runs in ~1 s each; PPO is marked slow."""
import numpy as np
import pytest

from rlnf.config import TrainConfig
from rlnf.train import run


def eval_stats(df):
    ev = df[df.phase == "eval"]
    return ev.true_return.mean(), ev.reached_goal.mean()


def test_alpha_zero_is_unsolved_by_design_but_solvable():
    """v2 premise: with neutral Q-init the sparse task is not learned at alpha=0 ...
    ... but with optimistic init (step cost as exploration bonus) it is, so the maze itself is fine."""
    _, goal_neutral = eval_stats(run(0.0, "honest", 0, "q", TrainConfig(total_steps=100_000)))
    _, goal_optimistic = eval_stats(run(0.0, "honest", 0, "q", TrainConfig(total_steps=100_000, q_init=0.0)))
    assert goal_neutral < 0.5
    assert goal_optimistic >= 0.95


def test_honest_feedback_makes_the_task_learnable():
    """Moderate honest neural feedback is enough to find the safe route."""
    for seed in range(3):
        true_ret, goal = eval_stats(run(0.5, "honest", seed, "q", TrainConfig(total_steps=50_000)))
        assert goal >= 0.95
        assert true_ret == pytest.approx(0.72, abs=0.05)


def test_honest_brain_high_alpha_does_not_degrade():
    moderate = np.mean([eval_stats(run(0.5, "honest", s, "q", TrainConfig(total_steps=50_000)))[0] for s in range(3)])
    high = np.mean([eval_stats(run(4.0, "honest", s, "q", TrainConfig(total_steps=50_000)))[0] for s in range(3)])
    assert moderate > 0.65
    assert high >= moderate - 0.1


@pytest.mark.slow
def test_ppo_honest_reaches_goal():
    _, goal = eval_stats(run(1.0, "honest", 0, "ppo", TrainConfig(total_steps=60_000)))
    assert goal >= 0.95


def test_centered_reward_makes_clean_steps_free():
    """Centered: neural return of a clean safe-route episode is ~0; uncentered: ~ -28 * p_clean."""
    c = run(1.0, "honest", 0, "q", TrainConfig(total_steps=50_000, centered=True))
    u = run(1.0, "honest", 0, "q", TrainConfig(total_steps=50_000, centered=False))
    ce, ue = c[c.phase == "eval"], u[u.phase == "eval"]
    assert ce.n_errors.mean() == 0 and ue.n_errors.mean() == 0
    assert abs(ce.neural_return.mean()) < 0.5
    assert ue.neural_return.mean() < -1.5


def test_episode_log_columns_and_bookkeeping():
    df = run(1.0, "both", 0, "q", TrainConfig(total_steps=10_000, eval_episodes=3, checkpoint_every=5_000))
    needed = {"neural_return", "true_return", "n_errors", "errors_in_fov", "errors_out_fov", "repeated_errors",
              "first_time_errors", "reached_goal", "steps", "phase", "step", "stuckness", "kl", "centered"}
    assert needed <= set(df.columns)
    tr = df[df.phase == "train"]
    assert (tr.errors_in_fov + tr.errors_out_fov == tr.n_errors).all()
    assert (tr.repeated_errors + tr.first_time_errors == tr.n_errors).all()
    assert (tr.n_hazard + tr.n_wall + tr.n_away == tr.n_errors).all()
    assert tr.stuckness.between(0, 1).all() and (tr.repeat_frac >= tr.stuckness).all()
    assert (df.phase == "eval").sum() == 3
    ck = df[df.phase == "checkpoint"]
    assert list(ck.step) == [5_000, 10_000]
    assert ck.kl.iloc[0] == 0.0 and ck.kl.iloc[1] >= 0   # reference policy is taken at the first checkpoint past 10%


def test_stuckness_detects_a_frozen_agent():
    """A policy that bumps the same wall forever has stuckness ~1; the safe route has 0."""
    from rlnf.config import BrainConfig, EnvConfig
    from rlnf.train import build_env, rollout
    env = build_env(EnvConfig(), BrainConfig(), TrainConfig(alpha=1.0), seed=0)
    frozen = rollout(env, lambda obs: 3, 1)[0]            # LEFT into the border from the start, forever
    assert frozen["stuckness"] == pytest.approx(0.99) and frozen["repeat_frac"] == pytest.approx(0.99)
    assert frozen["n_errors"] == 100 and not frozen["reached_goal"]
    # A two-cell oscillation that errs on every other step: one key repeated ~50 times.
    flip = {"n": 0}
    def oscillate(obs):
        flip["n"] += 1
        return 0 if flip["n"] % 2 else 2        # UP (away error, hidden) then DOWN (no error)
    osc = rollout(env, oscillate, 1)[0]
    assert osc["stuckness"] == pytest.approx(0.49) and osc["n_errors"] == 50
    g = env.unwrapped
    from rlnf.env import ACTIONS
    def optimal(obs):
        r, c = g.pos
        return min(ACTIONS, key=lambda a: g.cost_to_go[r + ACTIONS[a][0], c + ACTIONS[a][1]]
                   if g.in_bounds((r + ACTIONS[a][0], c + ACTIONS[a][1])) else np.inf)
    good = rollout(env, optimal, 1)[0]
    assert good["stuckness"] == 0.0 and good["reached_goal"]


@pytest.mark.parametrize("variant,n", [("double", 1), ("maxmin", 4), ("asrn", 1)])
def test_q_variants_learn_the_safe_route_with_honest_feedback(variant, n):
    df = run(1.0, "honest", 0, "q", TrainConfig(total_steps=50_000, q_variant=variant, q_maxmin_n=n))
    true_ret, goal = eval_stats(df)
    assert goal >= 0.95 and true_ret == pytest.approx(0.72, abs=0.05)


def test_asrn_noise_equalises_bin_variance():
    from rlnf.train import ASRN
    cfg = TrainConfig(asrn_warmup=2_000, asrn_bins=4, asrn_min_samples=10)
    asrn = ASRN(cfg, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    # Warm-up: small updates come with quiet rewards, large updates with noisy rewards.
    for _ in range(2_000):
        quiet = rng.random() < 0.5
        asrn.noise(rng.normal(0, 0.01 if quiet else 1.0), 0.01 if quiet else 1.0)
    assert asrn.noise_std is not None
    assert asrn.noise_std[0] == pytest.approx(1.0, abs=0.1)     # quiet bin gets ~unit noise
    assert asrn.noise_std[-1] == pytest.approx(0.0, abs=1e-9)   # noisiest bin gets none
    out = np.array([asrn.noise(0.0, 0.01) for _ in range(5_000)])
    assert abs(out.mean()) < 0.05 and out.std() == pytest.approx(1.0, abs=0.1)   # symmetric


def test_region_values_and_hidden_share_are_logged():
    df = run(1.0, "visibility", 0, "q", TrainConfig(total_steps=5_000, eval_episodes=2))
    tr = df[df.phase == "train"]
    assert {"q_visible", "q_hidden", "hidden_share", "n_visible", "n_hidden"} <= set(tr.columns)
    assert (tr.n_visible + tr.n_hidden == tr.steps).all()
    assert tr.hidden_share.between(0, 1).all()


@pytest.mark.parametrize("mode", ["masked", "imputed", "pessimistic"])
def test_observability_modes_run_and_log(mode):
    df = run(1.0, "visibility", 0, "q", TrainConfig(total_steps=5_000, eval_episodes=2, observability=mode))
    tr = df[df.phase == "train"]
    assert tr.observed_frac.between(0, 1).all()


def test_observability_masked_gives_no_reward_for_unobserved_steps():
    from rlnf.config import BrainConfig, EnvConfig
    from rlnf.train import build_env
    env = build_env(EnvConfig(), BrainConfig(visibility_flaw=True, noise_sigma=0.0), TrainConfig(alpha=1.0, observability="masked"), seed=0)
    env.reset()
    _, r, _, _, _ = env.step(0)                       # UP into the hidden half: unobserved, an 'away' error
    assert r == pytest.approx(-0.01)                  # only the step cost
    env.reset()
    _, r_obs, _, _, _ = env.step(3)                   # LEFT: visible wall bump at the start -> penalised and rescaled
    assert r_obs < -0.01


def test_observability_imputed_uses_observed_estimates():
    from rlnf.config import BrainConfig, EnvConfig
    from rlnf.train import build_env
    env = build_env(EnvConfig(), BrainConfig(visibility_flaw=True, noise_sigma=0.0), TrainConfig(alpha=1.0, observability="imputed"), seed=0)
    env.reset()
    for _ in range(5):
        env.step(3)                                   # observed LEFT bumps at the start: P(error) ~ 1 learned for LEFT
    env.step(0); env.step(0)                          # move into the hidden half
    _, r_hidden_left, _, _, _ = env.step(3)           # unobserved LEFT bump: imputed from the per-action mean
    assert r_hidden_left < -0.5
