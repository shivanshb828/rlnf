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
    assert env.start == (7, 0)
    assert env.goal == (7, 14)
    assert env.hazards.sum() == 2 and env.hazards[0, 4] and env.hazards[0, 10]
    assert env.height == env.width == 15
    # Bottom half mirrors the top half (hazards aside).
    assert np.array_equal(env.walls[:7], env.walls[8:][::-1])


def test_two_routes_of_equal_length(env):
    """Shortest path ignoring hazards (top) and avoiding them (bottom) are both 28 steps."""
    plain = cost_to_go(env.walls, env.hazards, env.goal, hazard_weight=0.0)
    assert plain[env.start] == 28
    assert env.cost_to_go[env.start] == 28
    # Top route: forced through the hazard columns, entirely hidden.
    top = cost_to_go(env.walls | (np.arange(15)[:, None] > 7), env.hazards, env.goal, hazard_weight=0.0)
    assert top[env.start] == 28
    ret, path, _ = greedy_rollout(env, top)
    assert sum(env.hazards[p] for p in path) == 2
    assert np.mean([env.fov[p] for p in path]) < 0.1
    assert ret == pytest.approx(1.0 - 2 * 0.5 - 28 * 0.01)


def test_safe_route_is_visible_and_error_free(env):
    ret, path, infos = greedy_rollout(env, env.cost_to_go)
    assert len(path) - 1 == 28
    assert not any(env.hazards[p] for p in path)
    assert all(env.fov[p] for p in path)
    assert ret == pytest.approx(1.0 - 28 * 0.01)
    assert all(info["errors"] == [] for info in infos)


def test_wall_bump_is_an_error_and_does_not_move(env):
    env.reset()
    _, r, term, trunc, info = env.step(LEFT)  # off the grid border
    assert env.pos == (7, 0)
    assert [e.kind for e in info["errors"]] == ["wall"]
    assert info["errors"][0].in_fov is True
    assert r == pytest.approx(-0.01)
    _, _, _, _, info = env.step(RIGHT)  # into the wall at (7, 1)
    assert env.pos == (7, 0)
    assert [e.kind for e in info["errors"]] == ["wall"]


def test_moving_away_is_an_error(env):
    env.reset()
    _, _, _, _, info = env.step(UP)  # toward the hazardous top: cost-to-go rises
    assert [e.kind for e in info["errors"]] == ["away"]
    assert info["errors"][0].in_fov is False
    _, _, _, _, info = env.step(DOWN)  # back on track: no error
    assert info["errors"] == []


def test_hazard_entry_penalty_and_event(env):
    env.reset()
    for _ in range(7):
        env.step(UP)
    for _ in range(3):
        env.step(RIGHT)
    _, r, _, _, info = env.step(RIGHT)  # into (0, 4)
    assert env.pos == (0, 4)
    assert info["entered_hazard"]
    assert r == pytest.approx(-0.01 - 0.5)
    assert "hazard" in [e.kind for e in info["errors"]]
    assert not info["errors"][0].in_fov


def test_goal_terminates_and_cap_truncates():
    env = GridWorld(EnvConfig(max_steps=5))
    env.reset()
    for _ in range(5):
        _, _, term, trunc, _ = env.step(LEFT)
    assert trunc and not term
    env = GridWorld(EnvConfig())
    _, path, infos = greedy_rollout(env, env.cost_to_go)
    assert infos[-1]["reached_goal"]
    assert path[-1] == env.goal
