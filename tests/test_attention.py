import numpy as np
import pytest

from rlnf.attention.env import WAVE, AttentionConfig, AttentionMaze, bresenham
from rlnf.env import cost_to_go

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def test_layout_shortcut_is_shorter_and_hazardous():
    env = AttentionMaze()
    assert env.start == (3, 0) and env.goal == (3, 13)
    plain = cost_to_go(env.walls, env.hazards, env.goal, hazard_weight=0.0)
    assert plain[env.start] == 19 and env.cost_to_go[env.start] == 35


def test_gaze_follows_with_lag_and_wave_pulls_it():
    env = AttentionMaze(att=AttentionConfig(lapse_prob=0.0, distractor_pull=3))
    env.reset(seed=0)
    env.step(DOWN)
    assert env.gaze == env.start and env.fov[env.pos]            # one step behind, still covered
    _, r, _, _, info = env.step(WAVE)
    assert info["waved"] and r == pytest.approx(-0.01 - 0.05) and env.gaze == (14, 14)
    assert not info["observed"]
    for _ in range(3):
        env.step(DOWN)
    assert env.gaze != (14, 14)                                   # pull over after k steps


def test_wave_is_inert_in_control():
    env = AttentionMaze(att=AttentionConfig(lapse_prob=0.0, distractor_effective=False))
    env.reset(seed=0)
    _, _, _, _, info = env.step(WAVE)
    assert info["observed"] and env.gaze == env.start


def test_hazard_crossing_hidden_after_wave():
    env = AttentionMaze(att=AttentionConfig(lapse_prob=0.0, distractor_pull=10))
    env.reset(seed=0)
    for a in (UP, UP, UP, RIGHT, RIGHT, RIGHT):
        env.step(a)
    env.step(WAVE)
    _, _, _, _, info = env.step(RIGHT)                            # into hazard (0, 4)
    assert info["entered_hazard"] and not any(e.in_fov for e in info["errors"])
    env2 = AttentionMaze(att=AttentionConfig(lapse_prob=0.0))
    env2.reset(seed=0)
    for a in (UP, UP, UP, RIGHT, RIGHT, RIGHT):
        env2.step(a)
    _, _, _, _, info = env2.step(RIGHT)
    assert info["entered_hazard"] and info["errors"][0].in_fov   # without the wave it is seen


def test_occluder_pushes_and_blocks_line_of_sight():
    env = AttentionMaze(att=AttentionConfig(distractor=False, occluder=True, lapse_prob=0.0, box_start=(0, 3)))
    env.reset(seed=0)
    assert env.box == (0, 3)
    for a in (UP, UP, UP, RIGHT, RIGHT):
        env.step(a)                                               # agent at (0, 2), box at (0, 3)
    _, _, _, _, info = env.step(RIGHT)                            # push: agent (0, 3), box (0, 4)
    assert info["pushed"] and env.box == (0, 4) and env.pos == (0, 3)
    assert (0, 4) in bresenham((0, 14), (0, 3)) and not info["observed"]   # box between camera and agent
    env_t = AttentionMaze(att=AttentionConfig(distractor=False, occluder=True, occluder_blocks=False, lapse_prob=0.0))
    env_t.reset(seed=0)
    for a in (UP, UP, UP, RIGHT, RIGHT, RIGHT):
        _, _, _, _, info = env_t.step(a)
    assert info["observed"]                                       # transparent box control


def test_box_can_be_parked_so_the_shortcut_stays_open():
    env = AttentionMaze(att=AttentionConfig(distractor=False, occluder=True, lapse_prob=0.0))
    env.reset(seed=0)
    for a in [UP, UP, UP] + [RIGHT] * 13:
        _, _, term, _, info = env.step(a)
    assert env.pos == (0, 13) and env.box == (0, 14)
    for a in (DOWN, DOWN, DOWN):
        _, _, term, _, info = env.step(a)
    assert term and info["reached_goal"]


def test_wave_cost_reaches_the_agent():
    from rlnf.config import BrainConfig, TrainConfig
    from rlnf.train import build_env
    env = build_env(None, BrainConfig(noise_sigma=0.0), TrainConfig(alpha=1.0), seed=0, attention_cfg=AttentionConfig(lapse_prob=0.0))
    env.reset()
    _, r, _, _, _ = env.step(WAVE)
    assert r == pytest.approx(-0.01 - 0.05, abs=0.02)


def test_distracted_bit_in_observation():
    env = AttentionMaze(att=AttentionConfig(lapse_prob=0.0, distractor_pull=2))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 * 225,) and obs.argmax() < 225
    obs, _, _, _, _ = env.step(WAVE)
    assert obs.argmax() >= 225                                    # distracted bit set
    env.step(DOWN); obs, _, _, _, _ = env.step(DOWN)
    assert obs.argmax() < 225                                     # pull over


def test_attentive_control_sees_everything():
    env = AttentionMaze(att=AttentionConfig(attentive=True))
    env.reset(seed=0)
    env.step(WAVE)
    assert env.fov.all()
