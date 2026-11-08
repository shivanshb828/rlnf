"""Gridworld with walls, hazards, and an observer field of view (FOV).

The env returns the *true* task reward. It also reports, in ``info``, every
"error event" the agent produced this step and whether that event happened
inside the observer's FOV. The simulated brain (rlnf/brain.py) consumes those
events; the agent never sees the true reward during training.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlnf.config import EnvConfig

# Action index -> (d_row, d_col): up, right, down, left.
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# The three kinds of observable mistake.
#   hazard: the agent entered a hazard cell
#   wall:   the agent bumped into a wall or the grid border (and did not move)
#   away:   the step increased the agent's true cost-to-go to the goal
ERROR_TYPES = ("hazard", "wall", "away")


@dataclass(frozen=True)
class ErrorEvent:
    kind: str                  # one of ERROR_TYPES
    cell: tuple[int, int]      # agent position after the step
    in_fov: bool               # whether that cell is inside the observer's FOV


def parse_grid(text: str) -> np.ndarray:
    """Turn an ASCII block into a 2-D array of single characters."""
    rows = [list(line) for line in text.strip("\n").splitlines()]
    assert len({len(r) for r in rows}) == 1, "ragged layout"
    return np.array(rows)


def cost_to_go(walls: np.ndarray, hazards: np.ndarray, goal: tuple[int, int],
               hazard_weight: float) -> np.ndarray:
    """Cheapest cost from every cell to the goal (Dijkstra, run backwards).

    Each move costs 1; entering a hazard costs an extra ``hazard_weight``.
    With ``hazard_weight = hazard_penalty / step_penalty`` this is exactly the
    true-reward-optimal cost-to-go measured in steps. Unreachable cells and
    walls get +inf.
    """
    h, w = walls.shape
    dist = np.full((h, w), np.inf)
    dist[goal] = 0.0
    frontier = [(0.0, goal)]
    while frontier:
        d, (r, c) = heapq.heappop(frontier)
        if d > dist[r, c]:
            continue
        # A step *into* (r, c) from a neighbour costs 1 + weight if (r, c) is a hazard.
        step_cost = 1.0 + hazard_weight * hazards[r, c]
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc]:
                if d + step_cost < dist[nr, nc]:
                    dist[nr, nc] = d + step_cost
                    heapq.heappush(frontier, (dist[nr, nc], (nr, nc)))
    return dist


class GridWorld(gym.Env):
    """9x9 gridworld. Observation: one-hot agent position. Actions: 4 moves."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        grid = parse_grid(self.cfg.layout)
        self.height, self.width = grid.shape
        self.walls = grid == "#"
        self.hazards = grid == "H"
        self.fov = parse_grid(self.cfg.fov) == "1"
        assert self.fov.shape == grid.shape, "FOV mask must match layout"
        self.start = tuple(int(x) for x in np.argwhere(grid == "S")[0])
        self.goal = tuple(int(x) for x in np.argwhere(grid == "G")[0])
        # How the observer judges "moved away from the goal": against the
        # true-reward-optimal route, where a hazard costs as much as
        # hazard_penalty / step_penalty extra steps.
        self.hazard_weight = self.cfg.hazard_penalty / self.cfg.step_penalty
        self.cost_to_go = cost_to_go(self.walls, self.hazards, self.goal, self.hazard_weight)

        self.observation_space = spaces.Box(0.0, 1.0, (self.height * self.width,), np.float32)
        self.action_space = spaces.Discrete(4)
        self.pos = self.start
        self.t = 0

    # -- helpers -----------------------------------------------------------
    def state_index(self, pos: tuple[int, int] | None = None) -> int:
        r, c = self.pos if pos is None else pos
        return r * self.width + c

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.height * self.width, dtype=np.float32)
        obs[self.state_index()] = 1.0
        return obs

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.t = 0
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        dr, dc = ACTIONS[int(action)]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        errors: list[ErrorEvent] = []
        entered_hazard = False
        progress = False

        if not self.in_bounds(nxt) or self.walls[nxt]:
            nxt = self.pos  # bump: stay put
            errors.append(ErrorEvent("wall", nxt, bool(self.fov[nxt])))
        else:
            if self.hazards[nxt]:
                entered_hazard = True
                errors.append(ErrorEvent("hazard", nxt, bool(self.fov[nxt])))
            if self.cost_to_go[nxt] > self.cost_to_go[self.pos]:
                errors.append(ErrorEvent("away", nxt, bool(self.fov[nxt])))
            progress = bool(self.cost_to_go[nxt] < self.cost_to_go[self.pos])
        self.pos = nxt

        reached_goal = self.pos == self.goal
        true_reward = (
            -self.cfg.step_penalty
            - self.cfg.hazard_penalty * entered_hazard
            + self.cfg.goal_reward * reached_goal
        )
        terminated = reached_goal
        truncated = self.t >= self.cfg.max_steps and not terminated
        info = {
            "true_reward": true_reward,
            "reached_goal": reached_goal,
            "entered_hazard": entered_hazard,
            "errors": errors,
            "progress": progress,          # a "good event": the step reduced the true cost-to-go
            "in_fov": bool(self.fov[self.pos]),
            "pos": self.pos,
        }
        return self._obs(), true_reward, terminated, truncated, info

    def region_masks(self) -> dict[str, np.ndarray]:
        """Boolean masks over state indices: open cells inside / outside the observer's FOV."""
        open_cells = ~self.walls.flatten()
        return {"visible": open_cells & self.fov.flatten(), "hidden": open_cells & ~self.fov.flatten()}

    def render_ascii(self) -> str:
        grid = parse_grid(self.cfg.layout)
        grid[self.pos] = "A"
        return "\n".join("".join(row) for row in grid)
