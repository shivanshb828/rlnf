# Phase log

Plain-language notes written after each phase of the three-part-paper round
("How reinforcement learning from neural feedback fails: blind spots, praise
hacking, and attention manipulation"). Commits are listed by short hash.

## Phase 0: housekeeping (2026-11-07)

- Committed v3 as `736e870`: mechanism (maximisation bias), known fixes, absent-supervisor port.
- Remote settings untouched (the user is making the repo private).

## Phase 1: finish Part 1 (blind spots)

Runtime estimate before launch (8 workers): `ppo_diag` 120 runs of our PyTorch PPO
at ~35 s each with Monte-Carlo critic diagnostics (~10 min), `dqn` 80 runs at
~20 s (~4 min), `observ` 1,280 tabular runs at ~1.2 s (~4 min). About 20 minutes
in total, run as one chained background job.

## Phases 2-3 sweeps: runtime estimate before launch

Chained background job, 8 workers: Phase 2 `praise_a` (320 Q runs + 20 PPO), `praise_b`
(360), `praise_c` (160), `praise_flip` (10): ~8 min. Phase 3 `att_distractor` (200 runs,
~3 s each), `att_occluder` (120 runs, 15-60 s each because of line-of-sight checks),
`att_fixes` (160): ~15 min. About 25 minutes total, overlapping the tail of the Phase 1
`observ` sweep.

### Phase 1 results (plain language)

**1a. PPO diagnosis** (our own PyTorch PPO, `rlnf/deep.py`, reward normalisation on,
α = 1, 10 seeds). The critic is *not* biased toward the hidden half: at the end of
training under the visibility brain its value for hidden-half states (≈ -0.5) matches
the Monte-Carlo return of the current policy from those states. What it gets wrong is
the visible half: V ≈ 0 against Monte-Carlo returns of -27 to -44, because the
collapsed policy no longer visits the visible half and the critic never sees those
penalties. The advantage statistics tell the story: in the first 20% of training the
mean GAE advantage in the hidden half is ≈ 0.00 and in the visible half ≈ -0.25 to
-0.29, so after per-batch advantage normalisation every hidden-half action looks
better than average and every visible-half action worse. PPO drifts into the quiet
half because its exploratory mistakes are penalised only where they are seen. Neither
of the two fixes helps: GAE λ = 0.8 froze 10/10 at both σ (default: 9/10 at σ = 0.03,
10/10 at 0.3); a 3-critic ensemble froze 7/10 at σ = 0.03 and 10/10 at σ = 0.3. The
honest brain with this PPO reaches the goal in 8/10 (default), 9-10/10 (ensemble), 6/10
(λ = 0.8) seeds, so our PPO is itself weaker than SB3's and the honest-brain failures
are exploration failures, not freezing. Verdict: PPO's mechanism is advantage-driven
(penalised exploration in the visible half vs a quiet hidden half), Q-learning's is
value-driven (maximisation bias on the hidden half's noise). Both are the same blind
spot seen through different estimators; the critic-side fixes do not touch it.

**1b. Deep value-based RL** (DQN / Double DQN, `rlnf/deep.py`, 10 seeds). Honest brain:
10/10 reach the goal for both at both σ. Visibility brain: DQN freezes 9/10 (σ = 0.03)
and 10/10 (σ = 0.3); Double DQN freezes 5/10 at both σ (late-training goal rate 0.5-0.6).
So Double DQN halves the freeze rate but does not remove it the way tabular Double Q
did (0/20). The function approximator keeps some of the bias (a shared network with a
periodically copied target is not two independent tables).

**1c. Observability-aware reward** (plain and Double Q, 20 seeds per cell). A first run
defined "observed" as "inside the environment's field of view" for every brain, which
wrongly masked signals the *honest* observer had produced; it was re-run with "observed =
the observer registered the step" (always true for a brain without the visibility flaw).
- Honest brain: no mode breaks it. 0/20 frozen and 0/20 shortcuts in every honest cell on
  both environments (Double Q's 4/20 shortcuts on absent-supervisor at σ = 0.3 are its
  baseline behaviour, identical under all four modes).
- Maze, visibility brain: `imputed` and `pessimistic` remove freezing completely
  (0/20 frozen, true return 0.72, plain and Double Q, σ = 0.03 and 0.3). `masked` does
  not: plain Q 7/20 frozen at σ = 0.3 and Double Q 8-12/20, worse than Double Q's own
  baseline (0/20 at σ = 0.3), because masking removes information instead of
  restoring the missing penalty while rescaling amplifies the observed ones.
- Absent supervisor, visibility brain: only `imputed` removes the shortcut (plain Q
  0/20 at both σ; Double Q 0/20 at σ = 0.03, 7/20 at σ = 0.3); `masked` and
  `pessimistic` leave it at 18-20/20. Taking the shortcut when unsupervised is
  genuinely optimal under a reward that is silent when unsupervised; a per-action
  mean penalty is too diluted to change that; a per-(cell, action) model fit on
  supervised steps restores the punishment where it belongs.
- Anomaly, flagged not explained: on the maze at σ = 0.03 Double Q's *baseline*
  froze 16/20 (true return -2.1), although Double Q froze 0/20 at σ = 0.1-0.3 in v3
  and 2/20 with real noise. Plain Q at σ = 0.03 froze 0/20. Likely a tie-breaking
  pathology when the centered reward is almost exactly zero everywhere in the
  hidden half; `imputed` and `pessimistic` remove it too.

**1d. Real ErrP noise.** Dataset: BNCI Horizon 2020 "013-2015 Monitoring
error-related potentials" (Chavarriaga & Millán 2010), 6 subjects x 2 sessions,
64-channel EEG at 512 Hz, ~500 cursor moves per session with ~20% erroneous; licence
CC BY-NC-ND 4.0 (raw files and derived outputs are kept out of git). Our decoder
(CAR, 1-10 Hz, FCz/Cz/Fz/CPz 0.2-0.8 s binned, L2 logistic regression; trained on
session 1, scored on session 2 recorded 50-643 days later) reaches held-out balanced
accuracy 80 / 73 / 81 / 58 / 63 / 53% for subjects 1-6 (AUC 0.92-0.71). The brain
then samples a real held-out output per simulated step. With one subject per seed
(20 seeds): visibility + plain Q froze 20/20 (every subject, 53-81% accuracy);
visibility + Double Q froze 2/20 (both at the two worst subjects); honest + plain Q
froze 2/20 (again the two worst subjects); honest + Double Q 0/20. The cliff holds:
every real decoder sits on its frozen side, and the fix from v3 transfers.

## Phase 2: praise hacking (plain language)

Code: `rlnf/praise/` (a reward-positivity-like "praise" channel that fires on good events,
0.5x the ErrP amplitude and 1.5x its noise, with its own decoder; an observer model that
predicts the agent's next move so that "surprise" can scale both channels; a valence-flip
switch). Sweeps `praise_a/b/c/flip`, analysis `experiments/part2.py`.

**2a. Does praise fix freezing? No.** Visibility brain, 20 seeds per cell. Plain Q never
reaches the goal at any praise ratio > 0: 20/20 stuck at ratios 0.25, 0.5 and 1 at both
σ = 0.03 and 0.3. At σ = 0.03 praise *creates* the failure (0/20 stuck without praise,
20/20 with it). What the stuck agents do is not the Part 1 freeze but a new exploit,
**praise farming across the blind-spot boundary**: from the start cell they step up into
the hidden half (a mistake nobody sees, reward -0.01) and step back down into view (a step
toward the goal, which the observer praises: P(good) = 1.0 at σ = 0.03, +0.99 reward),
forever. The proxy return of that cycle is +0.5 per step at praise ratio 1 (neural return
+50 per episode) against +1 once for reaching the goal; the true return is -1.0. Double
Q: 0-3/20 stuck at every ratio and σ (its σ = 0.03 baseline anomaly, 14/20, disappears
when praise is added). Our PPO: 10/10 stuck with or without praise. So praise does not
rescue the blind spot; for the plain learner it hands it a better exploit.

**2b. Does praise create surprise-seeking under the honest brain? No.** Honest brain,
praise ratio 0.5, surprise weight 0 / 0.5 / 1, α 0.25 / 1 / 4, plain and Double Q, 20
seeds each: every one of the 18 cells reaches the goal 20/20 by the safe route with true
return 0.72, mean observer surprise 0.23 (the predictor's prior keeps a 28-step route at
≈ 0.2 surprise), 0% unpredictable moves, action entropy 1.5 bits (the route's own mix),
no extra steps, 0/20 rise-then-fall in every cell. The surprise bonus (at most 0.5 x
praise amplitude x surprise) is always smaller than the error penalty an unpredictable
move earns under an honest observer, so there is nothing to seek. The original
"rise-then-fall" hypothesis is not supported here either.

**2c. Combined (visibility + praise 0.5 + surprise).** Plain Q: 20/20 stuck at w = 0
(praise farm), 16/20 at w = 1, α = 1, 20/20 at w = 1, α = 4. Double Q: 1/20 at w = 0,
0/20 at w = 1, α = 1, but 9/20 stuck at w = 1, α = 4 (true return -0.05). The blind
spot dominates: it turns praise into a farmable resource; surprise only matters for
Double Q at the highest pressure.

**Valence flip** (observer as competitor, honest brain, plain Q, 10 seeds): 10/10 safe
route, true return 0.72; the praise term just changes sign (-0.11 per episode) and the
error channel dominates. No effect.

**Artifacts to flag.** The praise farm depends on (i) praise for *local* progress judged
against the true cost-to-go and (ii) the retreat being unobserved; both are modelling
choices, but both are the natural ones (observers respond to good moves; the blind spot
is the premise of Part 1). The predictor's Laplace prior and half-life (PLACEHOLDER) set
the surprise floor at 0.2; a sharper observer model would lower it but not change 2b's
sign. All praise parameters are PLACEHOLDERs (see RELATED_WORK.md for what would ground
them).

## Phase 3: attention manipulation (plain language)

Code: `rlnf/attention/` (gaze model with lag, radius and lapses; a "wave" action that
pulls the gaze to a salient event for k steps and costs the agent 0.05; a pushable box
that blocks the camera's line of sight; controls: an inert wave, a transparent box, a
fully attentive honest observer). Layout: start (3,0), goal (3,13); hidden top route 19
steps with two hazards (true -0.19), safe bottom route 35 steps (true 0.65), so the
shortcut is worth hiding. The agent observes whether its salient event is still showing
(as the absent-supervisor agent observes the supervisor). Two design lessons on the way:
a wave with no cost visible to the agent and no verdict from the observer is a free no-op
that every learner spams (99 waves/episode even under the attentive honest observer),
which is the null-action trap of v2.1, not manipulation; and on the equal-length maze
there is no reason to distract at all. Both were fixed before the sweep.

**Distractor (20 seeds per cell, σ = 0.3, α = 1).** Double Q learns attention
manipulation when the pull is long enough: with k = 10, 13/20 seeds reach the goal by
the hidden shortcut and 100% of their hazard entries fall within k steps after a wave;
with k = 5, 8/20; with k = 2 (too short to reach the first hazard), 0/20. In both
controls 0/20 seeds take the shortcut (inert wave: 0 waves per episode; attentive honest
observer: 0 waves). The manipulators over-use the wave (15-20 per episode at k = 5-10,
at 0.05 each), so their true return is -1.1 to -1.5, far below the honest 0.65 and even
below the plain shortcut (-0.19): the hidden hazards are free but the hiding is not.
Plain Q mostly freezes in the "distracted" states instead (14-17/20 frozen at k = 5-10,
0-1/20 shortcuts), the Part 1 mechanism again: a state where nothing is judged.

**Occluder (20 seeds per cell, 100k steps).** Not learned. In all six cells (box blocks
line of sight / transparent box / attentive honest observer, plain and Double Q) 20/20
seeds take the safe route, 0/20 push the box even once in evaluation, true return 0.65.
The exploit requires eleven consecutive pushes along the top corridor over a state
space of 225 x 225 (agent x box) cells, and ε-greedy exploration with the neutral
Q-init never assembled that sequence within the budget. Because the three settings are
identical, nothing can be concluded about the occluder except that this learner did not
find it in 100k steps; a longer budget, shaped exploration, or a smaller box state would
be needed, and the result is listed under "Do not claim".

**Fixes (distractor k = 10 and occluder; 20 seeds per cell).** The decoupled / imputed
reward (unattended steps get the penalty a model fit on attended steps predicts for that
cell and action) removes the shortcut: 0/20 shortcut seeds for both learners (Double Q
baseline: 13/20), 20/20 (plain) and 18/20 (Double Q) reach the goal, no hazard entries.
It does not stop the waving itself: Double Q still waves 12.9 times per episode on the
safe route (true return -0.19 from the 0.05 cost each, vs 0.65 without waving), plain Q
3.3 times (true 0.45); the wave is cheap and nothing in the imputed reward prices it.
The masked reward (unattended steps get nothing, attended ones rescaled) makes things
worse, as in Part 1: Double Q 4/20 shortcuts and 14/20 frozen, plain Q 18/20 frozen and
0/20 reach the goal. On the occluder both fixes are uninformative (20/20 safe route
everywhere, as in the baseline).

**Phase 3 verdict.** Learned attention manipulation is real for the distractor with a
bias-corrected learner (Double Q: 8/20 at k = 5, 13/20 at k = 10, every hazard entry
right after a wave, 0/20 in both controls); the plain learner freezes instead; the
occluder was not discovered; the imputed fix removes the hidden shortcut but leaves the
cheap distraction habit, and the masked fix backfires.

## Phase 4: write-up prep

- `figures/`: part1_freeze_vs_accuracy_gaussian.png, part1_real_noise_cliff.png,
  part1_fixes_comparison.png, part1_observability_fix.png, part1_learners.png,
  part1_ppo_critic.png; part2_praise_vs_freeze.png, part2_surprise_pressure.png;
  part3_distractor.png, part3_fixes.png.
- `PAPER_NOTES.md`: Parts 1-3 claim by claim (numbers, seed counts, figure, caveats, fix),
  the Maxmin result with its best explanation, a "Do not claim" list (habituation,
  rise-then-fall, the α = 4 optimal-freeze, the σ = 0.03 Double Q anomaly, the PPO
  mechanism beyond the diagnostic, the occluder, all placeholder rates), and future work
  (brain-feedback LLMs, habituation with real data, live human-subject validation).
- `RELATED_WORK.md`: additions for RewP / observer FRN, Everitt et al. reward tampering,
  REALab, tomato watering, Lang 2024, Leike 2017, Vivanti 2019, Lan 2020, van Hasselt
  2010, Iturrate 2010, Kim 2017, Chavarriaga & Millán 2010.
- `README.md`: pointer section for the three-part round.
