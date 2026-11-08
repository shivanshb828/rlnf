# Reward hacking in RL from Neural Feedback (RLNF): a minimal simulation (v2)

An RL agent is rewarded by a decoder that reads a human observer's
error-related potentials (ErrPs, the brain's automatic "oops" response to
seeing a mistake). **Hypothesis:** when the neural reward has realistic flaws,
optimizing it hard makes the proxy (neural) reward and the true task reward
diverge, and the agent learns specific exploits.

This repo is a small, fully synthetic simulation to test that. There is no EEG
data; the "brain" is a few lines of arithmetic with switchable flaws. v1 (branch
`main`, results archived in `results/v1/`) was a 9x9 pilot. v2 (this branch)
gives the over-optimization hypothesis a fairer test and stress-tests v1's most
interesting finding, the "freezing" exploit. See "v1 vs v2" at the end.

## What is simulated

```
agent --action--> maze --error events + FOV flags--> brain --ErrP amplitude--> decoder
  ^                 |                                                             |
  |                 +-- true reward (logged only, never shown to the agent)        |
  +---------- agent reward = goal - step cost + alpha * ( -(P(error) - p_clean) ) <+
```

1. **Environment** (`rlnf/env.py`): a 15x15 maze with a start, a goal, walls,
   two hazards, and an observer field of view (FOV). True reward: +1 for the
   goal, -0.5 per hazard entry, -0.01 per step, 100-step cap. Every step reports
   the *error events* it produced (hazard entry, wall bump, or a step that moves
   the agent away from the goal) and whether each was inside the FOV.
2. **Brain** (`rlnf/brain.py`, class `Brain`): turns a step's error events into
   one ErrP amplitude, `base_amplitude * habituation * visible + noise`. Flaws:
   `visibility_flaw` (errors outside the FOV evoke nothing) and
   `habituation_flaw` (the same error at the same cell evokes less and less,
   with partial dishabituation after many repeats).
3. **Decoder** (`rlnf/brain.py`, class `Decoder`): 1-D logistic regression
   P(error | amplitude), fit once on honest-brain samples and then frozen. It
   also records `p_clean`, its mean P(error) on clean calibration steps.
4. **Agent reward** (`rlnf/train.py`, class `NeuralRewardWrapper`):
   `goal_reward * reached_goal - step_penalty + alpha * neural`, where
   `neural = -(P(error) - p_clean)` (centered, default) or `-P(error)`
   (uncentered ablation). The hazard penalty is never shown to the agent.
5. **Learners** (`rlnf/train.py`): tabular Q-learning (primary, 20 seeds) and
   PPO from stable-baselines3 (secondary, reduced grid, 10 seeds).

### The layout

```
....H.....H....   row 0: hidden route, forced through hazards at (0,4) and (0,10)
.###.#####.###.
.#...#...#...#.   a serpentine of dead ends hangs off each hazard column
.#.#.#.#.#.#.#.
.#.#.#.#.#.#.#.
.#.#...#...#.#.
.#.#########.#.
S#############G   row 7: start (7,0), goal (7,14); everything from here down is visible
.#.#########.#.
.#.#...#...#.#.
.#.#.#.#.#.#.#.   bottom half = mirror image of the top half, without hazards
.#.#.#.#.#.#.#.
.#...#...#...#.
.###.#####.###.
...............   row 14: safe route
```

| Route | Steps | Hazards | True return | Observer sees it? |
|---|---|---|---|---|
| Top (hidden) | 28 | 2 | 1 - 1.0 - 0.28 = **-0.28** | No (0 of 29 cells) |
| Bottom (safe) | 28 | 0 | 1 - 0.28 = **0.72** | Yes (29 of 29 cells) |
| Never reach the goal | 100 | 0 | **-1.00** | - |

The two routes have exactly equal length (the bottom half is the mirror image of
the top), so any preference for the hidden route must come from the feedback
signal. At α = 0 with an optimistic Q-table, seeds split about 50/50 between the
routes, as they should.

### Why α = 0 cannot solve the task (and how that was tuned)

The spec asked for a sparse task that tabular Q-learning at α = 0 solves in well
under half of seeds, so that the neural signal is genuinely needed. On a
deterministic gridworld with a per-step cost, tabular Q-learning has two
regimes and nothing in between:

| Q-table init | What happens at α = 0 (100k steps, 20 seeds) |
|---|---|
| 0.0 (optimistic) | The step cost makes every untried action look better than tried ones, so ε-greedy becomes systematic search. 20/20 seeds solve the ring, the spiral maze, and the spiral maze with a 35-step cap. |
| -1.0 (neutral = -step_penalty / (1 - γ), the value of never reaching the goal) | No exploration bonus; ε-greedy is a random walk. 0/20 seeds ever reach the goal in training. With honest feedback at α = 1: 20/20. |

v2 uses the neutral init (`q_init = -1.0`). The α = 0 success rate is therefore
0/20, which satisfies "well under half" but means the baseline agent does
nothing useful; keep that in mind when reading the α = 0 column. The honest
"away" signal is a dense shaping reward, which is why even α = 0.25 learns.

## Design decisions worth knowing before reading results

- **"Moving away from the goal" is judged against the true-optimal route.** The
  observer knows where the hazards are. Distance is the cheapest cost-to-go with
  a hazard costing `hazard_penalty / step_penalty = 50` extra steps (Dijkstra in
  `env.cost_to_go`). Plain shortest-path distance would flag the safe route as
  a string of errors.
- **One ErrP per step**: the max over coinciding error events, not the sum.
- **The brain persists across episodes** and habituates only to errors it
  perceives. Greedy evaluations use the habituated brain.
- **Identical noise across brain conditions.** The brain's RNG and the decoder
  are seeded by the run seed, and exactly one noise draw happens per env step,
  so for a given seed every condition sees the same noise sequence; only the
  systematic flaws differ (tested in `tests/test_brain.py`).
- **Centered neural reward.** v1 showed the decoder assigns P(error) ≈ 0.10 to
  clean steps, a per-step tax that dominated the agent's reward at high α.
  v2 subtracts `p_clean` (estimated on honest calibration data) so clean steps
  cost ≈ 0 on average. The uncentered version is kept as an ablation.
- **Checkpoints and divergence.** Every 5k steps the current greedy policy is
  evaluated for 5 episodes on a *deep copy* of the env and brain (training is
  untouched), and its KL divergence from the policy at 10% of training is
  computed over the states visited since the previous checkpoint. For
  Q-learning the policy is `softmax(Q / 0.1)`; for PPO it is the actual policy
  network, with a deep copy of the 10% network as reference.
- **Stuckness** = fraction of an episode's steps that repeat an earlier
  (cell, action) error from the same episode. A policy pressing into one wall
  scores 0.99; a two-cell oscillation that errs on every other step scores ≈
  0.5; the safe route scores 0. A seed is called "frozen" if its eval stuckness
  exceeds 0.3.
- **"Final true return"** is the mean over 20 greedy episodes after training.
- **PPO lock-in.** PPO can sharpen onto the first route it finds. v2 gives PPO
  an entropy bonus of 0.05 (v1: 0.01) but does not otherwise fight this; PPO is
  the secondary learner and its numbers carry this confound.

## Every parameter and why

All defaults live in `rlnf/config.py`. **PLACEHOLDER** marks values that are
guesses and must be grounded in the ErrP literature before any result is
reported.

### Environment (`EnvConfig`)

| Parameter | Default | Why |
|---|---|---|
| `layout`, `fov` | ASCII above | Mirror-symmetric maze: equal routes, hazards only on the hidden one. |
| `goal_reward` | 1.0 | Scale anchor. |
| `hazard_penalty` | 0.5 | Two hazards make the hidden route (-0.28) clearly worse than the safe route (0.72) and worse than never finishing (-1.0) only by a wide margin. |
| `step_penalty` | 0.01 | Mild time cost; the agent sees it; also sets the hazard-to-step exchange rate (50). |
| `max_steps` | 100 | 3.6x the route length so failing to reach the goal is a choice, not a limit. |

### Brain and decoder (`BrainConfig`)

| Parameter | Default | Why |
|---|---|---|
| `base_amplitude["hazard"]` | 1.0 | Scale anchor; units are arbitrary because the decoder is fit on them. |
| `base_amplitude["wall"]` | 1.0 | **PLACEHOLDER.** Treated as as salient as a hazard. |
| `base_amplitude["away"]` | 0.6 | **PLACEHOLDER.** A wrong-direction step is a milder error than a collision; the decoder detects it with P ≈ 0.7 rather than ≈ 0.97. |
| `noise_sigma` | 0.3 | **PLACEHOLDER, optimistic.** ≈ 90% single-trial decoder accuracy; real decoders are nearer 70-85%. |
| `habituation_decay` | 0.8 | **PLACEHOLDER.** Factor = decay ** exposure; the 10th identical error evokes ≈ 11%. Ablation sweeps 0.9 / 0.8 / 0.6. |
| `habituation_tau` | 500 steps | **PLACEHOLDER.** Exposure recovers as exp(-dt / tau). Ablation sweeps 100 / 500 / 2500. |
| `dishabituation_level` | 0.3 | **PLACEHOLDER.** After `dishabituation_onset` (10) remembered exposures the response ramps back over `dishabituation_width` (10) exposures to 30% of base amplitude. v1 had no dishabituation (level 0). |
| `decoder_error_rate` | 0.3 | **PLACEHOLDER but groundable.** Error fraction of the calibration set; 20-30% is typical of ErrP calibration protocols. |

### Training (`TrainConfig`)

| Parameter | Default | Why |
|---|---|---|
| `alpha` | swept | Weight on the neural reward. |
| `centered` | True | Subtract `p_clean`; False is the uncentered ablation. |
| `total_steps` | 100,000 | Q-learning converges by ≈ 15k steps with honest feedback; PPO needs ≈ 40k. |
| `gamma` | 0.99 | SB3 default. |
| `q_init` | -1.0 | Neutral init; see above. |
| `q_lr`, `q_eps_*` | 0.1, 1.0 → 0.05 over the first half | Standard tabular settings. |
| `checkpoint_every`, `checkpoint_episodes` | 5,000 steps, 5 episodes | 20 checkpoints per run; 5 greedy episodes is enough for true return (deterministic policy) but neural return stays noisy (≈ ±0.5). |
| `ref_fraction`, `kl_temperature` | 0.1, 0.1 | Reference policy at 10% of training; temperature 0.1 makes a 0.1 Q-gap a near-deterministic preference. |
| `eval_episodes`, `decoder_samples` | 20, 5,000 | As in v1. |
| `ppo_*` | SB3 defaults, `ent_coef = 0.05` | Entropy bonus raised from v1's 0.01 to counter route lock-in. |

## How to run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest                                   # 28 tests, ~1 min
.venv/bin/python experiments/sweep.py --smoke                # a few 20k-step runs
.venv/bin/python experiments/sweep.py --estimate             # per-job timing and ETA
.venv/bin/python experiments/sweep.py --learner q            # main + ablation, Q-learning (1320 runs, ~5 min on 8 cores)
.venv/bin/python experiments/sweep.py --learner ppo          # main + ablation, PPO (210 runs, ~30 min)
.venv/bin/python experiments/plots.py                        # figures + summary_main.csv, summary_ablation.csv
```

Sweeps: **main** = α ∈ {0, 0.25, 0.5, 1, 2, 4} × {honest, visibility,
habituation, both} × {centered, uncentered}, 20 seeds (Q); PPO runs α ∈ {0, 1, 4},
centered only, 10 seeds. **ablation** = habituation brain, `habituation_decay`
∈ {0.9, 0.8, 0.6} × `habituation_tau` ∈ {100, 500, 2500} × α ∈ {1, 4}; 20 seeds
(Q), 5 seeds (PPO).

### Output files (`results/`)

| File | One row per | Key columns |
|---|---|---|
| `main_runs.csv`, `ablation_runs.csv` | run | greedy-eval means of every episode metric; `frac_out_fov_late`, `stuckness_late` over the last 20% of training |
| `main_checkpoints.csv`, `ablation_checkpoints.csv` | checkpoint | `step`, `kl`, `true_return`, `neural_return`, `reached_goal`, `stuckness` |
| `main_curves.csv`, `ablation_curves.csv` | run × 5k-step bin | training-episode means |
| `summary_main.csv`, `summary_ablation.csv` | condition | means plus `k_goal`, `k_hazard_route`, `k_frozen`, `k_rise_fall` seed counts |

Episode metrics: `true_return`, `neural_return`, `agent_return`, `steps`,
`reached_goal`, `n_errors` (= `n_hazard` + `n_wall` + `n_away` =
`errors_in_fov` + `errors_out_fov` = `first_time_errors` + `repeated_errors`),
`mean_hab_factor`, `stuckness`.

## v1 vs v2

| | v1 (branch `main`, `results/v1/`) | v2 (this branch) |
|---|---|---|
| Grid | 9x9, one hazard, 10-step hidden route vs 20-step safe route | 15x15 mirror-symmetric maze, two hazards, both routes 28 steps |
| Why the agent might prefer the hidden route | It was shorter (confound) | Only the feedback signal can make it prefer either route |
| Agent reward | goal + α·(-P(error)); no step cost | goal - step cost + α·(-(P(error) - p_clean)); uncentered kept as ablation |
| Q-table init | 0 (step cost was absent, so neutral by accident) | -1 (neutral); α = 0 is unsolved by design (0/20) |
| Sparse-reward difficulty at α = 0 | Solved in 5/5 seeds (hazard route) | Never solved; neural signal needed |
| Habituation | decay 0.8, tau 500, no dishabituation | + dishabituation to 30% after 10 repeats (PLACEHOLDER) |
| Seeds | 5 | 20 (Q), 10 (PPO main), 5 (PPO ablation) |
| Learners | Q and PPO on the full grid | Q primary; PPO on α ∈ {0, 1, 4} only, entropy bonus 0.05 |
| New measurements | - | checkpoints with KL from the 10% policy; stuckness; repeat fraction |
| Headline v1 result | PPO + habituation froze in 0,1,2,3,4,5 of 5 seeds as α rose; visibility made feedback useless, not harmful | see below |

## Results

Run on 2026-11-07 with the defaults above. Numbers: `results/summary_main.csv`,
`results/summary_ablation.csv`. Figures: `results/overopt_q.png`,
`results/overopt_ppo.png` (headline), `results/final_true_return.png`,
`results/stuckness_alpha.png`, `results/stuckness_ablation.png`,
`results/hiding.png`. Every claim below gives seed counts; "frozen" means eval
stuckness > 0.3 (a random walk in this maze scores ≈ 0.07, pressing into one
wall scores 0.99, a two-cell oscillation ≈ 0.5).

### Tabular Q-learning, main sweep (20 seeds per cell)

Final true return (greedy eval). 0.72 = safe route, -0.28 = hidden hazard
route, -1.0 = never reaches the goal.

| condition | reward | α=0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| honest | centered | -1.18 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| honest | uncentered | -1.18 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| habituation | centered | -1.18 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| habituation | uncentered | -1.18 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| visibility | centered | -1.18 | **-0.91** | -0.91 | -0.91 | -0.91 | -0.91 |
| visibility | uncentered | -1.18 | 0.72 | 0.72 | 0.72 | 0.72 | 0.56 |
| both | centered | -1.18 | **-0.91** | -1.00 | -1.00 | -1.00 | -0.91 |
| both | uncentered | -1.18 | 0.72 | 0.72 | 0.67 | 0.63 | 0.72 |

- **Honest and habituation brains: 20/20 seeds take the safe route at every
  α ≥ 0.25, centered or not.** No degradation at α = 4. The habituation flaw
  alone, with dishabituation and 20 seeds, produces no exploit for Q-learning:
  0/20 hazard-route seeds, 0/20 frozen, at every α.
- **Visibility flaw + centered reward: 19/20 seeds freeze at every α ≥ 0.25**
  (final true return -0.91, stuckness 0.67-0.69, 99.7% of late-training errors
  hidden). The agent walks a few cells into the hidden half and presses into
  one wall for the rest of every episode (e.g. seed 0 at α = 1: 97 of 100 steps
  are the same bump at cell (4, 0)). Its neural return is ≈ +0.04, the best
  value the centered proxy can reach, while its true return is near the floor:
  **proxy and true reward fully diverge, at any α > 0, with no rise first.**
- **The same flaw with the uncentered reward: 20/20 seeds take the safe route
  at α ≤ 2**, and 19/20 at α = 4 (1 hazard-route seed, 1 frozen seed). The
  uncentered tax (≈ 0.1α per step, everywhere, including the hidden half) makes
  finishing the episode the only way to stop paying; the centered reward
  removes that pressure, and the sparse +1 alone is never found by a random
  walk. **Centering flips the visibility result from "harmless" to
  "catastrophic".** This is the biggest single design-choice effect in v2.
- **Both flaws behave like visibility alone** in both reward variants.
- **Rise-then-fall (proxy up while true return falls) was observed in 0/20
  seeds in every cell except three** (both/uncentered α = 1 and 2: 1/20 each;
  visibility/uncentered α = 4: 2/20). In those seeds the final policy switched to
  the hazard route or froze late in training. The over-optimization curves
  (`overopt_q.png`) show rise-and-plateau for honest/habituation, flat-at-the-
  floor for visibility/both (centered). **The hypothesis in its "optimize harder
  and it gets worse" form is not supported for Q-learning**; what the flawed
  conditions show is divergence from the first non-zero α onward.

### Why the visibility collapse happens, and how much of it is an artifact

Under the visibility flaw the hidden half is a *reward-null zone*: nothing the
agent does there is perceived, so every step pays only the step cost plus
zero-mean decoder noise. Two things then push a greedy tabular learner into it:

1. The visible half is strictly worse: every exploratory mistake there costs
   -α·P(error), and there is no positive signal until the goal is found.
2. Q-learning's max operator over-estimates the value of actions whose reward
   is zero-mean noise. In the seed-0 trace the start state's "up" (into the
   hidden half) has Q = +0.21 while the fixed point for a step that earns
   nothing is -1.0.

Once inside, a wall-bump action with a marginally higher Q than its neighbours
is absorbing (bumping leaves the state unchanged), so the greedy policy
freezes. Diagnostic (`.context/noise_diag.py`, visibility, centered, α = 1,
20 seeds, 100k steps):

| decoder noise σ | Q learning rate | seeds reaching the goal | seeds frozen |
|---|---|---|---|
| 0.3 (default) | 0.1 (default) | 1/20 | 19/20 |
| 0.1 | 0.1 | 17/20 | 3/20 |
| 0.03 | 0.1 | 20/20 | 0/20 |
| 0.3 | 0.02 | 2/20 | 18/20 |
| 0.1 | 0.02 | 4/20 | 16/20 |
| 0.03 | 0.02 | 3/20 | 17/20 |

So the *direction* of the effect (the hidden half attracts the agent and the
agent learns nothing there) comes from the flaw, but its *magnitude* depends on
the decoder's noise level and on the learner's hyperparameters in a way that is
not monotone or simple (a lower learning rate makes it worse at every noise
level, presumably because the visible half's guidance propagates too slowly
before ε decays). **Treat "19/20 freeze" as "this learner, these settings",
not as a property of the flaw.** Double Q-learning or a stochastic final policy
would be the next diagnostic.

### Freezing ablation, Q-learning (habituation brain, 20 seeds per cell)

| habituation decay (strength) | τ = 100 | τ = 500 | τ = 2500 |
|---|---|---|---|
| 0.9 (weak) | 0/20 frozen | 0/20 | 0/20 |
| 0.8 (default) | 0/20 | 0/20 | 0/20 |
| 0.6 (strong) | 0/20 | 0/20 | **2/20** (α = 1) · **2/20** (α = 4) |

Freezing appears for Q-learning only with the strongest habituation *and* the
slowest recovery, in 2/20 seeds, identically at α = 1 and α = 4. It scales
with habituation strength in the weakest possible sense (zero everywhere else).

### PPO, main sweep (α ∈ {0, 1, 4}, centered, 10 seeds per cell)

A caveat first: PPO's final eval uses the deterministic (argmax) policy, and in a
deterministic maze an argmax policy that does not reach the goal *must* end in
a cycle, so eval stuckness is high for every failed PPO seed. The columns that
separate "froze" from "merely failed" are the ones measured on the stochastic
training policy over the last 20% of training (`stuckness_late`,
`goal_rate_late`).

| condition | α | final true return | seeds reaching goal (eval) | late-training goal rate | late-training stuckness | neural return |
|---|---|---|---|---|---|---|
| honest | 0 | -0.34 | 5/10 | 0.90 | 0.05 | - |
| honest | 1 | **0.72** | 10/10 | 1.00 | 0.00 | 0.08 |
| honest | 4 | **-0.48** | 3/10 | 0.30 | 0.21 | -16.4 |
| habituation | 1 | -1.00 | 0/10 | 0.00 | 0.31 | -4.6 |
| habituation | 4 | -1.00 | 0/10 | 0.00 | 0.29 | -4.6 |
| visibility | 1 | -1.00 | 0/10 | 0.00 | 0.24 | +0.25 |
| visibility | 4 | -1.00 | 0/10 | 0.00 | 0.22 | +0.25 |
| both | 1 | -1.00 | 0/10 | 0.00 | 0.18 | +0.25 |
| both | 4 | -1.00 | 0/10 | 0.00 | 0.25 | +0.25 |

(α = 0 rows are identical across conditions by construction: same seeds, no
neural reward.)

- **For PPO the task is not sparse enough.** With the 0.05 entropy bonus the
  α = 0 training policy reaches the goal in 90% of late-training episodes; only
  the argmax eval fails (5/10). The "neural signal is needed" premise holds for
  neutral-init Q-learning, not for PPO.
- **Honest brain, α = 1: 10/10 safe route. Honest brain, α = 4: 7/10 seeds
  never reach the goal**, with a neural return of about -23 each. That is
  proxy-pessimal, so it is not reward hacking: it is PPO destabilised by a
  reward scale four times larger (every mistake costs ≈ -4 and the step reward
  swings by ±3 with decoder noise), with no reward normalisation. **The "α = 4
  is worse than α = 1" pattern for PPO is at least partly a reward-scale
  artifact of the training setup**, and it contaminates every α = 4 PPO cell.
- **Habituation brain: 10/10 seeds freeze at α = 1 and at α = 4**, by the
  stochastic-training-policy measure too (late-training stuckness ≈ 0.3, goal
  rate 0.00). The freezing exploit from v1 therefore **survives centering,
  dishabituation to 30%, and 10 seeds, for PPO**, and at α = 1 it cannot be the
  reward-scale artifact (honest α = 1 is 10/10 clean). The frozen agents
  oscillate between two cells in the *visible* half (83-89% of their errors
  are in view) repeating one error the observer has habituated to; their
  neural return (-4.6) is far worse than the safe route's (≈ 0), so this is a
  local optimum that traps a gradient learner, not a proxy-optimal policy.
  **For Q-learning the same flaw produced 0/20 frozen seeds at every α**, so
  freezing is a property of the learner-flaw pair, not of the flaw alone.
- **Visibility and both: 10/10 freeze in the hidden half** at α = 1 and 4
  (100% of errors hidden, stuckness 0.9-0.97 in eval), with the best neural
  return the centered proxy allows (+0.25). Same mechanism as Q-learning's
  reward-null zone, and for PPO it holds without the Q-specific
  over-estimation story.
- **Rise-then-fall: 0/10 seeds in every PPO cell with α > 0.** No flawed
  condition's policy ever became good and then degraded; they diverge from
  the start.

### Freezing ablation, PPO (habituation brain, 5 seeds per cell)

Seeds frozen (eval) / seeds reaching the goal:

| decay (strength) | τ = 100 | τ = 500 | τ = 2500 |
|---|---|---|---|
| 0.9 (weak), α = 1 | 5/5 · 0/5 | 5/5 · 0/5 | 5/5 · 0/5 |
| 0.8, α = 1 | 5/5 · 0/5 | 5/5 · 0/5 | 5/5 · 0/5 |
| 0.6 (strong), α = 1 | 5/5 · 0/5 | 5/5 · 0/5 | 5/5 · 0/5 |
| 0.9, α = 4 | 5/5 · 0/5 | 5/5 · 0/5 | 5/5 · 0/5 |
| 0.8, α = 4 | 5/5 · 0/5 | 5/5 · 0/5 | 5/5 · 0/5 |
| 0.6, α = 4 | 5/5 · 0/5 | 5/5 · 0/5 | 3/5 · 2/5 |

**For PPO, freezing does not scale with habituation strength: it is total in
every cell**, including the weakest decay with the fastest recovery, and the
only cell with any survivors is the *strongest* one at α = 4 (likely noise, n
= 5). Late-training stuckness on the stochastic policy is 0.14-0.46 everywhere.
This is not what a dose-response relationship looks like, and the reason is
visible in the habituation arithmetic: a two-step oscillation repeats one
error every other step, so the leaky exposure count saturates at
1 / (1 - e^(-2/τ)) ≈ 50 even for τ = 100, and 0.9^50 ≈ 0.005. Any per-exposure
decay below 1 becomes near-total habituation under rapid repetition; the
parameter that actually bounds the ErrP during freezing is
`dishabituation_level` (0.3), which was held fixed. **The ablation therefore
mostly shows that the habituation model has no floor, which is a design
choice to revisit**, not a property of observers.

### Verdicts

1. **Did any condition show proxy return rising while true return falls?**
   No. 0 seeds in every PPO cell with α > 0 and 0/20 in every Q-learning cell
   except three uncentered cells with 1-2/20 late route switches. The flawed
   conditions diverge from the first non-zero α (true return at the floor,
   neural return at its ceiling for visibility/both), and the honest condition
   rises and plateaus. The over-optimization hypothesis in its "rise then fall
   with more optimization pressure" form is **not supported** by either
   learner on this task. What *is* supported is the weaker claim: a flawed
   signal, optimized at all, is worse than no signal at all here (true return
   -0.9 to -1.0 vs -0.34 for PPO at α = 0).
2. **Does freezing survive centering, more seeds, and Q-learning?** Centering:
   yes (PPO + habituation, 10/10 at α = 1 and 4, late-training goal rate 0).
   More seeds: yes (10/10 and 45/45 ablation cells at α = 1). Dishabituation to
   30%: yes. Q-learning: **no** (0/20 at every α in the main sweep; 2/20 only
   with the strongest, slowest habituation). Freezing is a trap for a
   gradient-based, on-policy learner, not for ε-greedy tabular learning, which
   keeps sampling the alternatives and finds the goal.
3. **The visibility flaw** produced the one effect that is consistent across
   both learners and both α levels: the agent moves to the hidden half and
   stays there (19/20 Q-learning seeds at every α; 10/10 PPO seeds). It only
   appears with the centered reward for Q-learning, and its magnitude depends
   on decoder noise and learning rate.

### Things that look like artifacts of design choices rather than of the flaws

- **Centering removes the only pressure to finish an episode.** The agent's
  reward is sparse goal + tiny step cost + neural term; with the neural term
  centered and no perceived errors, wandering costs nothing. The uncentered
  tax did real work in v1 and in the v2 uncentered cells. A realistic agent
  reward would include a genuine time cost; the -0.01 step cost is absorbed by
  the neutral Q-init.
- **Neutral Q-init makes α = 0 a pure random walk** (0/20 success); with
  optimistic init the same maze is solved 20/20. Both regimes are arbitrary;
  there is no intermediate one for a deterministic tabular learner. For PPO
  the task is solved at α = 0 anyway.
- **PPO at α = 4 is destabilised by reward scale**, honest brain included (7/10
  failures with proxy-pessimal returns). Any claim about "high α" in PPO needs
  reward normalisation first.
- **Stuckness in a deterministic eval conflates "froze" with "failed"**: any
  argmax policy that does not reach the goal cycles. Use `stuckness_late` /
  `goal_rate_late` (stochastic training policy) for PPO freezing claims; the
  conclusions above do.
- **The habituation model has no floor** except `dishabituation_level`, so
  under rapid repetition every tested strength saturates (see ablation).
- **The decoder's clean-step false-positive rate (p_clean ≈ 0.10)** and its
  noise (σ = 0.3) are placeholders that the visibility result is sensitive to.
- **The "away" error is a dense shaping signal** the honest observer provides
  on every step. It is what makes honest feedback learnable from a random
  walk; a real observer would not flag every suboptimal step.
- **Q-learning's max operator over-estimates noise-only regions**, which is
  the mechanism behind the visibility freeze for Q (double Q-learning would
  test this).

### What v2 changed about v1's conclusions

| v1 claim | v2 |
|---|---|
| Visibility flaw makes feedback useless, not harmful | With centered reward it is harmful for both learners (true return -0.9 to -1.0, below the α = 0 baseline). The v1 "useless" verdict came from the uncentered tax pushing the agent to finish. |
| PPO + habituation freezes, monotonically more with α | Freezes 10/10 at both α = 1 and 4; no α dependence left to see. Survives centering, dishabituation, and 10 seeds. Does not transfer to Q-learning. |
| Q + habituation takes a habituated hazard route (2/5 at α = 4) | 0/20 at every α. Did not replicate on the equal-length layout. |
| Honest brain never degrades with α | True for Q (20/20 at α = 4). For PPO, α = 4 fails 7/10 for reward-scale reasons. |
| "Hiding" signature confounded with route length | With equal routes, hidden-half preference is driven by the flaw alone: ≈ 100% of late-training errors hidden under visibility/both for both learners (`hiding.png`). |
