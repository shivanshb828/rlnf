"""Sanity tests on learning behaviour. Q-learning runs in ~1 s; PPO is marked slow."""
import numpy as np
import pytest

from rlnf.config import TrainConfig
from rlnf.train import run


def eval_stats(df):
    ev = df[df.phase == "eval"]
    return ev.true_return.mean(), ev.reached_goal.mean()


def test_alpha_zero_q_learning_reaches_goal():
    df = run(alpha=0.0, condition="honest", seed=0, learner="q", train_cfg=TrainConfig(total_steps=30_000))
    true_ret, goal_rate = eval_stats(df)
    assert goal_rate >= 0.95
    # With no neural signal and gamma < 1 the agent should take the short hazard route.
    assert true_ret == pytest.approx(0.40, abs=0.05)


@pytest.mark.slow
def test_alpha_zero_ppo_reaches_goal():
    df = run(alpha=0.0, condition="honest", seed=0, learner="ppo", train_cfg=TrainConfig(total_steps=40_000))
    _, goal_rate = eval_stats(df)
    assert goal_rate >= 0.95


def test_honest_brain_high_alpha_does_not_degrade():
    """Honest neural feedback at alpha=4 should be about as good as at alpha=0.5."""
    moderate = np.mean([eval_stats(run(0.5, "honest", s, "q", TrainConfig(total_steps=50_000)))[0] for s in range(3)])
    high = np.mean([eval_stats(run(4.0, "honest", s, "q", TrainConfig(total_steps=50_000)))[0] for s in range(3)])
    assert moderate > 0.7                     # honest feedback steers to the safe route
    assert high >= moderate - 0.1


def test_episode_log_columns_and_bookkeeping():
    df = run(alpha=1.0, condition="both", seed=0, learner="q", train_cfg=TrainConfig(total_steps=5_000, eval_episodes=3))
    needed = {"neural_return", "true_return", "n_errors", "errors_in_fov", "errors_out_fov",
              "repeated_errors", "first_time_errors", "reached_goal", "steps", "phase", "step"}
    assert needed <= set(df.columns)
    assert (df.errors_in_fov + df.errors_out_fov == df.n_errors).all()
    assert (df.repeated_errors + df.first_time_errors == df.n_errors).all()
    assert (df.n_hazard + df.n_wall + df.n_away == df.n_errors).all()
    assert (df[df.phase == "eval"].shape[0]) == 3
    assert df.step.is_monotonic_increasing
