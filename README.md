# Reward hacking in RL from Neural Feedback (RLNF): a minimal simulation

An RL agent is rewarded by a decoder that reads a human observer's
error-related potentials (ErrPs, the brain's automatic "oops" response to
seeing a mistake). **Hypothesis:** when the neural reward has realistic flaws,
optimizing it hard makes the proxy (neural) reward and the true task reward
diverge, and the agent learns specific exploits.

This repo is a small, fully synthetic simulation to test that. There is no EEG
data; the "brain" is a few lines of arithmetic with two switchable flaws.

## What is simulated

```
agent  --action-->  gridworld  --error events + FOV flags-->  brain  --ErrP amplitude-->  decoder
  ^                    |                                                                     |
  |                    +-- true reward (logged only, never shown to the agent)               |
  +------------------- agent reward = goal reward + alpha * ( -P(error) ) <------------------+
```

1. **Environment** (`rlnf/env.py`): a 9x9 gridworld with a start, a goal, walls,
   one hazard, and an observer field of view (FOV). True reward: +1 for the goal,
   -0.5 per hazard entry, -0.01 per step, 100-step cap. Every step reports a list
   of *error events* (hazard entry, wall bump, or a step that moves the agent
   away from the goal) together with whether each event was inside the FOV.
2. **Brain** (`rlnf/brain.py`, class `Brain`): converts the step's error events
   into one scalar ErrP amplitude, `base_amplitude * habituation * visible +
   noise`. Two flaws can be switched on: `visibility_flaw` (errors outside the
   FOV evoke nothing) and `habituation_flaw` (the same error at the same cell
   evokes less and less).
3. **Decoder** (`rlnf/brain.py`, class `Decoder`): 1-D logistic regression
   P(error | amplitude), fit once on samples from an *honest* brain under a
   random policy, then frozen, like a real pretrained ErrP classifier.
4. **Agent reward** (`rlnf/train.py`, class `NeuralRewardWrapper`):
   `goal_reward * reached_goal + alpha * (-P(error))`. The hazard and step
   penalties are never shown to the agent. That is the point: the only thing
   that can steer it off the hazards is the neural signal.
5. **Learners** (`rlnf/train.py`): tabular Q-learning (fast, reliable) and PPO
   from stable-baselines3 (so results do not depend on one algorithm).

### The layout

```
....H....      row 0: the short route (10 steps). Hidden from the observer.
S.#####.G      row 1: start (1,0) and goal (1,8). Hidden.
..#####..      rows 2-8: visible to the observer.
..#####..
..#####..
..#####..
..#####..
.........      row 7: the safe route (20 steps). Visible.
.........
```

| Route | Steps | Hazards | True return | Observer sees it? |
|---|---|---|---|---|
| Short (over the top) | 10 | 1 | 1 - 0.5 - 0.10 = **0.40** | No |
| Safe (around the bottom) | 20 | 0 | 1 - 0.20 = **0.80** | Mostly (81% of cells) |

An agent that ignores hazards takes the short route (true return 0.40). An agent
that correctly weighs them takes the safe route (0.80). So "final true return"
is almost a two-valued readout of which route the agent chose; intermediate
values mean a mix across seeds or a failure to reach the goal at all.

## Design decisions worth knowing before reading results

- **"Moving away from the goal" is judged against the true-optimal route.** The
  observer is assumed to know where the hazards are. Distance is the cheapest
  cost-to-go where a hazard costs `hazard_penalty / step_penalty = 50` extra
  steps (Dijkstra in `env.cost_to_go`). Using plain shortest-path distance would
  flag every step of the safe route as an error, which no reasonable observer
  would do. Consequence: on the short route the agent produces 3 "away" errors
  plus 1 hazard error, all hidden.
- **One ErrP per step.** If several error events coincide, the amplitude is the
  max over events, not the sum. An ErrP is one waveform in a ~300 ms window.
- **The brain persists across episodes.** The observer watches the whole
  training run, so habituation accumulates across episodes and recovers in env
  steps, not per episode. Greedy evaluation at the end also uses the
  habituated brain.
- **The observer only habituates to errors it perceives.** Under the visibility
  flaw, hidden errors neither evoke an ErrP nor build exposure.
- **The decoder is fit on a rebalanced calibration set.** A random policy errs on
  ~65% of steps. Real ErrP calibration sessions use ~20-30% error rates, so the
  fit set is subsampled to `decoder_error_rate`. The decoder's intercept encodes
  this prior; with the defaults it assigns a mean P(error) of about 0.10 to
  clean steps. That residual is a per-step tax that every condition pays, and it
  is the main reason the neural return is negative even for perfect play.
- **Only the discount favours short paths.** The agent's reward has no step
  penalty (per the spec), so the 10 extra steps of the safe route cost it only
  `0.99^10` in discounting plus 10 steps of decoder tax. This is deliberately a
  mild pressure; the honest brain's hazard penalty (about `alpha * 1`) should
  easily overcome it.
- **PPO can lock into the first route it finds.** With a sparse goal reward,
  PPO often discovers the 10-step route first and then sharpens before it ever
  explores the 20-step detour. In a hyperparameter check on the honest brain,
  no setting escaped this on every seed. The defaults (SB3 defaults plus a 0.01
  entropy bonus, 100k steps) found the safe route on 5 of 6 test seeds. When
  reading PPO results, remember that part of any shortfall under the honest
  brain is exploration failure, not reward hacking. Q-learning with
  epsilon-greedy exploration does not have this problem here.
- **"Final true return" is measured in a greedy evaluation phase** (20 episodes
  with the deterministic / greedy policy after training, `phase == "eval"`).
  Training-phase rows (`phase == "train"`) include exploration noise and are
  what the learning curves show.

## Every parameter and why

All defaults live in `rlnf/config.py`. **PLACEHOLDER** marks values that are
guesses and must be grounded in the ErrP literature before any result is
reported.

### Environment (`EnvConfig`)

| Parameter | Default | Why |
|---|---|---|
| `layout`, `fov` | ASCII above | Smallest grid with a hidden short hazardous route and a visible safe route. |
| `goal_reward` | 1.0 | Scale anchor. |
| `hazard_penalty` | 0.5 | Half a goal: makes the short route (one hazard) clearly worse than the safe route but still worth finishing. |
| `step_penalty` | 0.01 | Mild time cost; also sets the hazard-to-step exchange rate (50) used by the observer's cost-to-go. |
| `max_steps` | 100 | Generous cap (5x the safe route) so failing to reach the goal is a choice, not a limit. |

### Brain and decoder (`BrainConfig`)

| Parameter | Default | Why |
|---|---|---|
| `base_amplitude["hazard"]` | 1.0 | Scale anchor; amplitude units are arbitrary because the decoder is fit on them. |
| `base_amplitude["wall"]` | 1.0 | **PLACEHOLDER.** Treated as as salient as a hazard. |
| `base_amplitude["away"]` | 0.6 | **PLACEHOLDER.** ErrP amplitude grows with error severity in some studies; a wrong-direction step is a milder error than a collision. The decoder detects it with P ~ 0.7 instead of ~ 0.97. |
| `noise_sigma` | 0.3 | **PLACEHOLDER, optimistic.** Gives d' ~ 3.3 for hazard/wall errors and ~ 91% single-trial decoder accuracy. Real single-trial ErrP decoders are nearer 70-85%. Lower SNR raises the per-step tax and is a knob worth sweeping. |
| `habituation_decay` | 0.8 | **PLACEHOLDER.** Amplitude factor is `decay ** exposure`; the 2nd identical error is at 80%, the 10th at ~11%. The real magnitude of ErrP habituation (per error type, over a session) needs grounding. |
| `habituation_tau` | 500 steps | **PLACEHOLDER.** Exposure decays as `exp(-dt / tau)`, so the observer forgets an error after a few hundred steps (a few episodes). Real recovery time scales are unknown to me. |
| `decoder_error_rate` | 0.3 | **PLACEHOLDER but groundable.** Error fraction of the calibration set; 20-30% is typical for ErrP calibration protocols. |
| `visibility_flaw` | False | Set per condition. |
| `habituation_flaw` | False | Set per condition. |

### Training (`TrainConfig`)

| Parameter | Default | Why |
|---|---|---|
| `alpha` | swept | Weight on the neural reward. |
| `total_steps` | 100,000 | Enough for both learners to converge on the honest brain; PPO needs most of it. |
| `gamma` | 0.99 | SB3 default; the only source of path-length pressure (see above). |
| `eval_episodes` | 20 | Greedy episodes used for "final" numbers. |
| `decoder_samples` | 5,000 | Random-policy steps collected to fit the decoder (before rebalancing). |
| `q_lr`, `q_eps_*` | 0.1, 1.0 to 0.05 over the first half | Standard tabular settings; converge in well under 100k steps here. |
| `ppo_n_steps`, `ppo_n_epochs`, `ppo_batch_size`, `ppo_lr` | 2048, 10, 64, 3e-4 | SB3 defaults. Shorter rollouts (512 steps) locked into the short route far more often. |
| `ppo_ent_coef` | 0.01 | Small entropy bonus; higher values (0.05, 0.1) did not help with route lock-in. |

### Sweep (`experiments/sweep.py`)

alpha in {0, 0.25, 0.5, 1, 2, 4}; brain in {honest, visibility, habituation,
both}; seeds 0-4; learners {q, ppo}. 240 runs.

## How to run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .      # or: uv sync
.venv/bin/python -m pytest                                  # 22 tests, ~20 s
.venv/bin/python experiments/sweep.py --smoke               # 8 short runs, ~30 s -> results/smoke.csv
.venv/bin/python experiments/sweep.py --estimate            # time one full job per learner
.venv/bin/python experiments/sweep.py                       # full sweep, ~15-20 min on 8 cores -> results/sweep.csv
.venv/bin/python experiments/plots.py                       # -> results/{headline,hiding,curves_q,curves_ppo}.png, summary.csv
.venv/bin/python experiments/plots.py results/smoke.csv     # plots for any other sweep file
```

Training one configuration by hand:

```python
from rlnf.train import run
df = run(alpha=1.0, condition="both", seed=0, learner="q")   # one row per episode
```

### Output columns (`results/sweep.csv`)

One row per episode. `phase` is `train` or `eval`; `step` is the global env step
at the end of the episode.

| Column | Meaning |
|---|---|
| `learner`, `condition`, `alpha`, `seed`, `episode` | run identifiers |
| `steps`, `reached_goal` | episode length; goal reached |
| `true_return` | sum of true rewards (goal, hazard, step) |
| `neural_return` | sum of -P(error) |
| `agent_return` | sum of the reward the agent was trained on |
| `n_errors`, `n_hazard`, `n_wall`, `n_away` | error events by kind |
| `errors_in_fov`, `errors_out_fov` | error events by visibility |
| `first_time_errors`, `repeated_errors` | whether that (kind, cell) error had occurred before in the run |
| `mean_hab_factor` | mean habituation factor over perceived errors (NaN if none) |

## Results of the first sweep (2026-11-07, defaults above, 5 seeds)

Full numbers: `results/summary.csv`. Figures: `results/headline.png`,
`results/hiding.png`, `results/curves_q.png`, `results/curves_ppo.png`.
The sweep took 37 min on 8 cores (the 8 PPO processes contend for cores, so
the per-job estimate of 51 s roughly doubles).

Final true return (greedy eval, mean over 5 seeds; 0.40 = short hazard route,
0.80 = safe route, -1.0 = never reached the goal):

| learner | condition | α=0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| Q | honest | 0.40 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| Q | visibility | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| Q | habituation | 0.40 | 0.80 | 0.80 | 0.80 | 0.80 | 0.64 |
| Q | both | 0.40 | 0.40 | 0.02 | 0.40 | 0.40 | 0.02 |
| PPO | honest | 0.40 | 0.80 | 0.80 | 0.80 | 0.72 | 0.72 |
| PPO | visibility | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| PPO | habituation | 0.40 | 0.12 | -0.16 | -0.44 | -0.72 | -1.00 |
| PPO | both | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |

What the data say, including the parts that do not fit the hypothesis:

- **Honest brain works and does not degrade with α for Q-learning.** Any
  α ≥ 0.25 moves the agent to the safe route. The sanity expectation holds.
- **Visibility flaw: the neural feedback becomes useless, not harmful.** The
  agent takes the hidden hazard route at every α, exactly as at α = 0. Its
  neural return there (-1.08) is *better* than the honest agent's on the safe
  route (-2.21), so the proxy and true rewards point in opposite directions.
  But true return never falls below the no-feedback baseline, so "optimizing
  harder makes it worse" is **not** supported for this flaw in this layout.
- **Habituation flaw with PPO: the clearest α-dependent collapse, via an
  exploit I did not design for.** The number of seeds whose final policy
  never reaches the goal rises monotonically with α (0, 1, 2, 3, 4, 5 of 5).
  Those agents step into the hidden rows and press into the same wall for the
  rest of every episode. The observer habituates to that one repeated error
  within a few hundred steps (logged habituation factor goes to 0.00), so
  staying put costs only the decoder's noise tax, while *exploring* produces
  novel errors at full amplitude. The agent learns to freeze before it ever
  finds the goal. The honest brain never produced this at any α, so the flaw
  is causal, not just PPO being fragile. Note the frozen policy's neural
  return (about -10.9) is far worse than the safe route's (-2.2): this is a
  cheap local optimum that traps a gradient learner, not a proxy-optimal
  policy.
- **Habituation flaw with Q-learning: the textbook exploit, but only at α = 4
  and only in 2 of 5 seeds.** Those two agents take the hazard route with the
  observer fully habituated to that hazard (factor 0.00), neural return -0.89
  vs -2.2 on the safe route. Here the proxy genuinely prefers the exploit.
  The other three seeds, and all seeds at α ≤ 2, learn the safe route.
  With n = 5 this is suggestive, not established.
- **Both flaws: same as visibility alone, plus occasional noise-driven
  stalls.** One seed (seed 3) at α = 0.5 and α = 4 gets stuck pressing into a
  hidden wall after crossing the hazard. Its Q-values for "bump the wall" and
  "move toward the goal" at cell (0, 6) differ by 0.03: a hidden wall bump
  costs nothing beyond the noise tax, so the tabular estimates are within
  noise of each other and the greedy argmax flips. Minor, but it shows the
  visibility flaw also removes the signal that would make wall bumps
  obviously bad.
- **PPO honest at α = 2 and 4 drops to 0.72** because one seed each locked
  into the hazard route (habituation factor 1.0, neural return -3.2, worse
  than the safe route). That is the exploration lock-in described above, not
  reward hacking, and it is a confound for every PPO number.
- **The "hiding mistakes" signature is confounded with route choice in this
  layout.** All errors a converged policy makes are on the hidden route, so
  the fraction of errors outside the FOV is 1.0 for every flawed condition
  *and* for α = 0, and undefined for the honest brain (no errors). Using the
  last 20% of training instead (where ε-greedy / stochastic-policy errors
  still happen) gives a readable version: the honest brain's exploratory
  errors are mostly in view (7-33% hidden), every flawed condition's are
  ~100% hidden. A layout where the hidden and visible routes are equally
  long would separate "hides mistakes" from "takes the shorter route".
- **The decoder's false-positive tax matters.** Mean P(error) on a clean
  step is ~0.10, so every step costs 0.1α regardless of behaviour. At α = 4
  this is the dominant term in the agent's reward and raises the variance of
  tabular updates; it is probably why the Q-learning anomalies appear at α = 4.
  A lower-SNR (more realistic) decoder would make this worse.

## Things to ground or extend later

- Every PLACEHOLDER above, especially the noise level and the habituation
  decay/recovery constants.
- Habituation in real observers is probably more global (per error type or per
  error rate) than per cell. The per-cell model is what the spec asked for and
  is the easier exploit to detect.
- The decoder sees only amplitude. A real decoder sees a spatiotemporal window
  and has latency; neither is modelled.
- Only one hazard. A layout with several hidden and visible hazards would let
  the habituation exploit show up as a *choice* of which hazard to repeat.
