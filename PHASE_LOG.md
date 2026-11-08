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
