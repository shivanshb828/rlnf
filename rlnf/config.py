"""Every tunable parameter lives here.

Defaults marked ``# PLACEHOLDER`` are guesses that must be grounded in the ErrP
literature before any result is reported. See README.md for the reasoning
behind each default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Grid layout. S = start, G = goal, # = wall, H = hazard, . = open floor.
# v2: a 15x15 maze whose bottom half is the mirror image of its top half, so
# the route over the top and the route under the bottom are both exactly
# 28 steps. Only the top route has hazards (cells (0,4) and (0,10), which it
# must cross). Each half also contains a serpentine of dead ends hanging off
# the hazard columns, so a random walk gets lost.
LAYOUT = """\
....H.....H....
.###.#####.###.
.#...#...#...#.
.#.#.#.#.#.#.#.
.#.#.#.#.#.#.#.
.#.#...#...#.#.
.#.#########.#.
S#############G
.#.#########.#.
.#.#...#...#.#.
.#.#.#.#.#.#.#.
.#.#.#.#.#.#.#.
.#...#...#...#.
.###.#####.###.
...............
"""

# Observer field of view: 1 = the observer sees this cell, 0 = does not.
# The bottom half (rows 7-14, including start and goal) is visible; the top
# half, which holds the hazardous route, is hidden.
FOV = """\
000000000000000
000000000000000
000000000000000
000000000000000
000000000000000
000000000000000
000000000000000
111111111111111
111111111111111
111111111111111
111111111111111
111111111111111
111111111111111
111111111111111
111111111111111
"""

# Blind-spot layout family (same maze, different observer field of view):
#   off_route   rows 0-6 hidden: the hazardous top route is unobserved, the safe route is watched (original)
#   on_route    rows 8-14 hidden: the safe bottom route is unobserved, the hazardous route is watched (flipped)
#   half_route  bottom-left quadrant hidden: the first half of the safe route is unobserved
def _fov(hidden) -> str:
    return "\n".join("".join("0" if hidden(r, c) else "1" for c in range(15)) for r in range(15)) + "\n"


FOV_VARIANTS = {
    "off_route": FOV,
    "on_route": _fov(lambda r, c: r >= 8),
    "half_route": _fov(lambda r, c: r >= 8 and c <= 7),
}


@dataclass(frozen=True)
class EnvConfig:
    layout: str = LAYOUT
    fov: str = FOV
    goal_reward: float = 1.0      # true reward for reaching the goal
    hazard_penalty: float = 0.5   # true penalty per hazard entry
    step_penalty: float = 0.01    # true penalty per step
    max_steps: int = 100          # episode cap (truncation)


@dataclass(frozen=True)
class BrainConfig:
    # ErrP amplitude (arbitrary units) for a first-time, visible error of each
    # kind. The decoder is fit on these units, so only ratios matter.
    base_amplitude: dict[str, float] = field(
        default_factory=lambda: {"hazard": 1.0, "wall": 1.0, "away": 0.6}  # PLACEHOLDER
    )
    noise_sigma: float = 0.3          # PLACEHOLDER: single-trial noise std (same units)
    habituation_decay: float = 0.8    # PLACEHOLDER: factor = decay ** exposure
    habituation_tau: float = 500.0    # PLACEHOLDER: env steps for exposure to recover by 1/e
    # Dishabituation: after `onset` remembered exposures the response partially
    # recovers, ramping linearly over `width` further exposures to `level`.
    dishabituation_level: float = 0.3   # PLACEHOLDER: fraction of base amplitude recovered
    dishabituation_onset: float = 10.0  # PLACEHOLDER: exposures before recovery starts
    dishabituation_width: float = 10.0  # PLACEHOLDER: exposures over which it ramps up
    decoder_error_rate: float = 0.3   # PLACEHOLDER: fraction of error trials in the calibration set
    visibility_flaw: bool = False     # True: errors outside the FOV evoke no ErrP
    habituation_flaw: bool = False    # True: repeated errors evoke smaller ErrPs


# The four brain conditions of the sweep.
CONDITIONS: dict[str, dict[str, bool]] = {
    "honest": dict(visibility_flaw=False, habituation_flaw=False),
    "visibility": dict(visibility_flaw=True, habituation_flaw=False),
    "habituation": dict(visibility_flaw=False, habituation_flaw=True),
    "both": dict(visibility_flaw=True, habituation_flaw=True),
}


@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 1.0            # weight on the neural reward
    learner: str = "q"            # "q" (tabular Q-learning) or "ppo"
    total_steps: int = 100_000    # env steps of training per run
    gamma: float = 0.99           # discount; the only thing that favours short paths
    eval_episodes: int = 20       # greedy episodes after training
    centered: bool = True         # neural reward = -(P(error) - p_clean); False = raw -P(error)
    # Observability-aware reward (assumes the system knows whether each step was in the observer's view):
    #   baseline     unobserved steps get the decoder output as usual
    #   masked       unobserved steps get no neural reward; observed ones are rescaled by 1 / observed fraction
    #   imputed      unobserved steps get -P_hat(error | cell, action), a running mean fit on observed steps only
    #   pessimistic  unobserved steps get the mean observed P(error) for that action
    #   tax          unobserved steps get the mean observed penalty, no state or action dependence (ablation)
    #   denoise      unobserved steps get no neural reward and observed steps are NOT rescaled (ablation)
    observability: str = "baseline"
    real_errp: bool = False       # drive the brain with sampled real held-out decoder outputs (rlnf/realerrp.py)
    real_errp_subject: int = 0    # which subject (1-6); 0 = seed mod 6 + 1
    checkpoint_every: int = 5_000 # steps between policy checkpoints (greedy eval + KL)
    checkpoint_episodes: int = 5  # greedy episodes per checkpoint, on a copy of the env/brain
    ref_fraction: float = 0.1     # checkpoint at this fraction of training is the KL reference
    kl_temperature: float = 0.1   # softmax temperature turning Q-values into a policy for KL
    decoder_samples: int = 5_000  # honest-brain random-policy steps used to fit the decoder
    # Tabular Q-learning
    q_lr: float = 0.1
    q_variant: str = "plain"      # "plain", "double" (van Hasselt 2010), "maxmin" (Lan et al. 2020), "asrn" (Vivanti et al. 2019)
    q_maxmin_n: int = 2           # number of Q-tables for maxmin
    asrn_warmup: int = 10_000     # ASRN: steps of warm-up used to bin update magnitudes and estimate per-bin reward std
    asrn_bins: int = 10           # ASRN: number of equal-width bins over the update magnitude (paper: unspecified)
    asrn_min_samples: int = 10    # ASRN: bins with fewer warm-up samples get no noise and do not set the target std
    # Initial Q-value. -1.0 = -step_penalty / (1 - gamma), the return of a policy that
    # never reaches the goal. With 0.0 the step cost acts as an exploration bonus and
    # tabular Q-learning solves the maze at alpha = 0 in 20/20 seeds (see README).
    q_init: float = -1.0
    q_eps_start: float = 1.0
    q_eps_end: float = 0.05
    q_eps_fraction: float = 0.5   # fraction of training over which epsilon decays linearly
    # PPO (stable-baselines3 defaults except ent_coef; see README on route lock-in)
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_lr: float = 3e-4
    ppo_ent_coef: float = 0.05
    # VecNormalize(norm_reward=True): rescales rewards by the running std of discounted returns.
    # The v2 main/ablation PPO sweeps ran with False (see README); default True from v2.1 on.
    ppo_normalize_reward: bool = True
    # Our own PPO (rlnf/deep.py, learner "ppo_torch"), used for the critic diagnosis
    ppo_gae_lambda: float = 0.95
    ppo_n_critics: int = 1        # ensemble size; the critic is their mean
    ppo_clip: float = 0.2
    # DQN / Double DQN (rlnf/deep.py, learners "dqn" and "ddqn")
    dqn_lr: float = 1e-3
    dqn_buffer: int = 50_000
    dqn_batch: int = 64
    dqn_warmup: int = 1_000
    dqn_train_every: int = 4
    dqn_target_every: int = 1_000
