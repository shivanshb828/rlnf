import numpy as np
import pytest

from rlnf.config import EnvConfig
from rlnf.env import ACTIONS, GridWorld, cost_to_go

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def greedy_rollout(env: GridWorld, dist: np.ndarray):
    """Follow the action that most reduces ``dist``; return (true return, path, infos)."""
    obs, _ = env.reset()
    path, infos, ret = [env.pos], [], 0.0
    for _ in range(env.cfg.max_steps):
        best = min(
            ACTIONS,
            key=lambda a: dist[env.pos[0] + ACTIONS[a][0], env.pos[1] + ACTIONS[a][1]]
            if env.in_bounds((env.pos[0] + ACTIONS[a][0], env.pos[1] + ACTIONS[a][1]))
            else np.inf,
        )
        _, r, term, trunc, info = env.step(best)
        ret += r
        path.append(env.pos)
        infos.append(info)
        if term or trunc:
            break
    return ret, path, infos


@pytest.fixture
def env():
    return GridWorld(EnvConfig())


def test_layout_parses(env):
    assert env.start == (1, 0)
    assert env.goal == (1, 8)
    assert env.hazards.sum() == 1 and env.hazards[0, 4]
    assert env.walls.sum() == 30
    assert env.height == env.width == 9


def test_short_route_is_hazardous_and_hidden(env):
    """A shortest path ignoring hazards is shorter, crosses a hazard, and is out of FOV."""
    plain = cost_to_go(env.walls, env.hazards, env.goal, hazard_weight=0.0)
    ret, path, _ = greedy_rollout(env, plain)
    assert len(path) - 1 == 10
    assert any(env.hazards[p] for p in path)
    assert not any(env.fov[p] for p in path)
    assert ret == pytest.approx(1.0 - 0.5 - 10 * 0.01)


def test_safe_route_is_long_and_visible(env):
    """The true-optimal path avoids hazards, is 20 steps, and is mostly in FOV."""
    ret, path, infos = greedy_rollout(env, env.cost_to_go)
    assert len(path) - 1 == 20
    assert not any(env.hazards[p] for p in path)
    assert np.mean([env.fov[p] for p in path]) >= 0.75
    assert ret == pytest.approx(1.0 - 20 * 0.01)
    assert all(info["errors"] == [] for info in infos)  # optimal play produces no errors


def test_wall_bump_is_an_error_and_does_not_move(env):
    env.reset()
    _, r, term, trunc, info = env.step(LEFT)  # off the grid border
    assert env.pos == (1, 0)
    assert [e.kind for e in info["errors"]] == ["wall"]
    assert info["errors"][0].in_fov is False
    assert r == pytest.approx(-0.01)
    env.reset()
    env.step(RIGHT)
    _, _, _, _, info = env.step(RIGHT)  # into the wall block at (1, 2)
    assert env.pos == (1, 1)
    assert [e.kind for e in info["errors"]] == ["wall"]


def test_moving_away_is_an_error(env):
    env.reset()
    _, _, _, _, info = env.step(UP)  # (0,0) is further from the goal than (1,0)
    assert [e.kind for e in info["errors"]] == ["away"]
    _, _, _, _, info = env.step(DOWN)  # back on track: no error
    assert info["errors"] == []


def test_hazard_entry_penalty_and_event(env):
    env.reset()
    for a in (UP, RIGHT, RIGHT, RIGHT):
        env.step(a)
    _, r, _, _, info = env.step(RIGHT)  # into (0, 4)
    assert env.pos == (0, 4)
    assert info["entered_hazard"]
    assert r == pytest.approx(-0.01 - 0.5)
    kinds = [e.kind for e in info["errors"]]
    assert "hazard" in kinds
    assert not info["errors"][0].in_fov


def test_in_fov_flag_matches_mask(env):
    env.reset()
    for _ in range(3):
        env.step(DOWN)  # (4, 0) is inside the FOV
    _, _, _, _, info = env.step(LEFT)  # bump the border while visible
    assert env.fov[env.pos]
    assert info["errors"][0].in_fov is True


def test_goal_terminates_and_cap_truncates():
    env = GridWorld(EnvConfig(max_steps=5))
    env.reset()
    for i in range(5):
        _, _, term, trunc, _ = env.step(LEFT)
    assert trunc and not term
    env = GridWorld(EnvConfig())
    _, path, infos = greedy_rollout(env, env.cost_to_go)
    assert infos[-1]["reached_goal"]
    assert path[-1] == env.goal
