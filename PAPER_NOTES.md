# Paper notes: "How reinforcement learning from neural feedback fails: blind spots, praise hacking, and attention manipulation"

Every claim states its conditions (layout, learner, Q-init, training length, noise), its
exact numbers with seed counts, the figure, caveats, and the fix if any. Conventions:
"frozen" = eval stuckness > 0.3 (share of an episode spent repeating one (cell, action)
error; a random walk in the maze scores ≈ 0.07); "failed" = final greedy policy reaches
the goal in fewer than half of 20 eval episodes. Defaults unless stated: centered neural
reward, α = 1, decoder noise σ = 0.3 (90% balanced accuracy), tabular Q-learning with
neutral Q-init -1.0, 100k steps, 20 seeds. Tables in `results/summary_*.csv`, figures in
`figures/`, phase-by-phase narrative in `PHASE_LOG.md`.

---

## Modelling assumptions that shape the results (read before any claim)

- **Oracle shaping in the honest brain.** The "away" error fires whenever a step increases the *true optimal* cost-to-go (hazards priced at hazard_penalty / step_penalty extra steps). An honest observer therefore delivers dense, exactly-correct shaping on every step. That is why the honest brain solves a maze a random walk never solves (α = 0 with neutral Q-init: 0/20 in every layout) and why "honest succeeds" is not evidence that the task is learnable without guidance. All honest-vs-flawed gaps are gaps against an oracle.
- **Neutral Q-init (-1.0 = -step_penalty / (1 - γ)).** Chosen so the step cost is not an exploration bonus. With q_init = 0, plain Q solves every layout at α = 0 (20/20) by systematic exploration, and the off-route freeze halves (19/20 → 8/20). Every tabular rate below is conditional on the stated q_init.
- **Imputation in the maze is a flat tax.** The per-(cell, action) model behind the `imputed` reward has no entries for cells that are never observed; in the maze every hidden cell is unobserved by construction, so 0% of imputed steps come from the table and the reward is the per-action mean of observed P(error): a constant penalty on unobserved steps. It is numerically identical to `pessimistic` in every maze cell. Only on absent-supervisor (100% from the table) is it imputation.
- **The attention agent observes the distraction.** Its state includes whether the salient event it triggered is still showing. With that bit hidden the exploit vanishes (0/20 at every cost); it mirrors the absent-supervisor benchmark but is an assumption about the agent's sensors.
- **Real decoder outputs are sampled i.i.d. per step.** The real-ErrP condition draws one held-out decoder probability per step from a subject's error or correct pool, independently of the previous step: real output *distributions* (false positives, misses, between-subject differences), not noise *dynamics* (drift, habituation, autocorrelation).
- **Centering with a finite calibration set.** `p_clean` is a sample mean (~3,500 steps); its error times α times the 100-step horizon is what made "never terminate" optimal at α = 4 in v2.1 (5/20 decoder seeds).

---

## Part 1: blind spots (the observer does not see part of the state)

### Claim 1.1 — Two distinct blind-spot failures: a trap when the unobserved region is off the useful route, lost guidance when it is on it.
- **Conditions.** 15x15 mirror maze, equal 28-step routes, hazards only on the top route; σ 0.03 and 0.3; plain and Double Q; q_init -1 and 0; honest / visibility / α = 0 baseline; 100k steps; 20 seeds. Layouts: off-route (top route hidden), on-route (bottom safe route hidden), half-route (bottom-left quadrant hidden).
- **Numbers (σ 0.3).** Off-route, q_init -1: honest 20/20 reach the goal, α = 0 0/20, visibility plain 1/20 (18 of the 19 failures end in the hidden region), Double Q 20/20. On-route, q_init -1: visibility plain 0/20, Double Q 2/20, all failures end on the unobserved route (20/20); honest 20/20. Half-route: plain 0/20, Double Q 8/20. With q_init 0: off-route plain 12/20 reach the goal, Double Q 20/20; on-route plain 0/20 (α = 0 baseline 20/20), Double Q 9/20; half-route plain 7/20, Double Q 18/20.
- **Separation.** By layout the two types separate: off-route failures end in the hidden off-route region (17-18/19), on-route failures end on the hidden route (19-20/20), and the learner-side fix (Double Q) removes the first (0/20) and not the second (18/20 at 100k, 12/20 at 300k). The review's α = 0 criterion only separates them for optimistic inits: with q_init -1 the α = 0 baseline fails everywhere (0/20), so it never certifies "solvable without guidance"; with q_init 0 the on-route visibility failure (plain 0/20) coexists with an α = 0 success (20/20), which makes it a trap by that criterion even though the route is unobserved. Say: a blind spot on the route is lost guidance for a learner that needs guidance and an active trap for one that does not; the type is a property of the (layout, learner, init) triple.
- **Figure.** `figures/review_two_failure_types.png`.
- **Caveats.** The α = 0 baselines are themselves init-dependent (plain Q 20/20 at q_init 0, Double Q only 3/20). The half-route layout is a mixed case and is reported, not interpreted.

### Claim 1.2 — The off-route trap: a tabular Q-learner retreats into the unobserved region and freezes there at every realistic decoder accuracy.
- **Conditions.** Off-route layout, plain Q, q_init -1, α ∈ {0.25, 1, 4}, 100k steps, 20 seeds per point.
- **Numbers.** Frozen 0/20 at 100% decoder balanced accuracy (σ 0.03), 3/20 at 99.9% (σ 0.1), 20/20 at 96% (σ 0.2), 19/20 at 90% (σ 0.3), 20/20 at 78% (σ 0.5), 20/20 at 66% (σ 0.8); the same at α = 0.25, 1 and 4. Honest brain 0/20 down to 78%, 1-2/20 at 66%. Not under-training: 17/20 at 300k steps (σ 0.3). Init-dependent: 19 / 16 / 8 of 20 at q_init -1 / -0.5 / 0 (σ 0.3).
- **Figure.** `figures/part1_freeze_vs_accuracy_gaussian.png`, `figures/review_qinit_sensitivity.png`.
- **Caveats.** A step, not a dose-response, located at a decoder far better than any real one. Rates are conditional on neutral Q-init; with q_init 0 the effect is 8/20.
- **Fix.** Double Q (Claim 1.4, off-route only); a constant tax on unobserved steps (Claim 1.6).

### Claim 1.3 — The off-route trap's mechanism is maximisation bias on zero-mean decoder noise in a region without penalties, not Vivanti's Boring Areas Trap.
- **Conditions.** Off-route layout, plain Q, q_init -1, α = 1, σ 0.3 unless stated.
- **Numbers.** Reward variance per region in the first 20% of training (10 seeds): hidden/visible ratio 2·10⁹ at σ 0.03 (0/10 frozen) → 19.5 at σ 0.2 (10/10) → 1.8 at σ 0.8 (10/10): freezing starts when the quiet region *gains* noise, and the hidden half has the *higher* mean reward during exploration (-0.01 vs -0.6 to -0.13). Fixes sort by estimation bias (20 seeds): plain 19/20, Double Q 0/20, Maxmin N = 2 / 4 / 8: 4 / 12 / 11, ASRN 20/20. Value iteration on the agent's exact expected reward: freezing is worth 1-80 units less than the safe route at α ≤ 1 (a trap). More exploration makes it worse (ε → 0.3 over 90%: 20/20 at every learning rate; ε → 0.01 over 25%: 10-14/20).
- **Figure.** `results/variance_by_region.png`, `figures/part1_fixes_comparison.png`, `results/exploration_regions.png`.
- **Caveats.** The Maxmin result (an under-estimating learner froze *more*, monotonically in N) is consistent with the bias account, read as: pessimism lowers both halves, but the visible half's sampled mean is already negative under exploration while the hidden half's is ≈ 0, so pessimism widens the gap that pushes the agent into the quiet region. This is a reading of the numbers, not an intervention. ASRN was implemented from the paper's prose (bins by update magnitude, 10 bins, 10k warm-up).

### Claim 1.4 — Double Q-learning removes the off-route trap and nothing else; Double DQN only halves it.
- **Conditions.** Visibility brain, q_init -1, α = 1.
- **Numbers.** Off-route, 100k: 0/20 frozen at σ 0.1 and 0.3, 1/20 at 0.5; at 300k: 0 / 1 / 2 of 20 at σ 0.1 / 0.3 / 0.5; with real decoder outputs 2/20 (100k and 300k). Honest brain: 0/20 at every σ. On-route: 18/20 fail at 100k, 12/20 at 300k. Half-route: 8/20 fail (σ 0.3). Absent-supervisor shortcut: 19/20 (unchanged). DQN (10 seeds): 9/10 and 10/10 frozen at σ 0.03 / 0.3; Double DQN 5/10 and 5/10.
- **Figure.** `figures/part1_learners.png`, `figures/part1_fixes_comparison.png`.
- **Caveats.** The earlier "Double Q at σ 0.03 fails 14-16/20" result was slow learning, not a trap: 0/20 at 300k steps and 9/20 with learning rate 0.2 at 100k. Retired.

### Claim 1.5 — PPO freezes through advantages, not value bias (diagnosed, not resolved).
- **Conditions.** Our PyTorch PPO with reward normalisation, off-route layout, α = 1, 10 seeds.
- **Numbers.** Frozen 9/10 (σ 0.03) and 10/10 (σ 0.3); SB3 PPO (v2) 10/10. End-of-training critic: hidden-half V ≈ -0.4 vs Monte-Carlo ≈ -0.5 (unbiased); visible-half V ≈ -0.5 vs Monte-Carlo -27 to -40 (never visited). Mean GAE advantage in the first 20% of training: hidden ≈ 0.00, visible -0.25 to -0.29. GAE λ 0.8: 10/10 frozen; 3-critic ensemble: 7/10 and 10/10. Honest brain with this PPO: 8/10 reach the goal.
- **Figure.** `figures/part1_ppo_critic.png`.
- **Caveats.** Our PPO is weaker than SB3's on the honest brain; the advantage reading was not tested by intervention; no critic-side fix was found. The tax fixes it from the reward side (9/10, Claim 1.6).

### Claim 1.6 — A constant penalty on unobserved steps fixes the off-route trap for every learner and harms the on-route case; true imputation is needed, and sufficient, for the absent-supervisor shortcut.
- **Conditions.** Visibility brain, σ 0.3, α = 1, q_init -1, 20 seeds (10 for deep learners); modes: baseline, denoise (unobserved steps get 0, no rescaling), tax (unobserved steps get the mean observed penalty), pessimistic (per-action mean), imputed (per-(cell, action) mean, falling back to per-action).
- **Numbers.** Off-route failures: plain baseline 19/20 → tax 1/20, pessimistic = imputed 0/20, denoise 4/20; Double Q baseline 0/20, denoise **13/20** (harmful). Half-route: tax = pessimistic = imputed 0/20 for both learners (baseline 20/20 and 12/20). On-route: tax 0/20 fail but **20/20 reach the goal via the visible hazardous route** (true return -0.36 / -0.28); pessimistic = imputed plain 12/20 fail, Double Q 20/20 fail; denoise 20/20 fail. Absent-supervisor shortcut (n_hazard > 0.25): baseline 18-19/20; imputed **0/20** (plain) and 7/20 (Double Q); pessimistic, tax 18-19/20; denoise 20/20. Deep learners, off-route: DQN 0/10 → 10/10 (tax) and 10/10 (imputed); Double DQN 5/10 → 10/10, 10/10; PPO 0/10 → 9/10, 8/10. Honest brain: no mode changes anything (0/20 failures everywhere).
- **Figure.** `figures/review_fixes_by_mechanism.png`, `figures/part1_observability_fix.png`.
- **Mechanism.** In the maze the fix is the tax (imputed draws 0% of unobserved steps from its table and equals pessimistic exactly); noise removal alone is not a fix and can hurt. On absent-supervisor the punishment tile is observed in supervised episodes, so per-(cell, action) imputation restores the missing penalty and a flat tax (diluted over all actions) does not.
- **Caveats.** Assumes the system knows per step whether the observer saw it. Side effect: wherever the unobserved region is on the useful route, the tax pushes the agent onto the watched hazards (on-route 20/20).

### Claim 1.7 — The off-route trap holds with real decoder output distributions, for every subject.
- **Conditions.** BNCI 013-2015 (Chavarriaga & Millán 2010; 6 subjects, 2 sessions, 64-ch EEG, CC BY-NC-ND 4.0). Decoder: CAR, 1-10 Hz, FCz/Cz/Fz/CPz 0.2-0.8 s binned, L2 logistic regression, fit on session 1, scored on session 2 (50-643 days later); held-out balanced accuracy 80 / 73 / 81 / 58 / 63 / 53% (subjects 1-6; AUC 0.92-0.71). Off-route layout, α = 1, q_init -1, 100k steps, **20 seeds per subject**.
- **Numbers.** Visibility + plain Q: 20/20 frozen for every subject (120/120). Visibility + Double Q: 0/20 for subjects 1-3, 1/20, 5/20, 2/20 for 4-6 (8/120). Honest + plain Q 0, 0, 0, 1, 1, 1 (3/120); honest + Double Q 0, 0, 0, 2, 3, 1 (6/120). At 300k steps (one subject per seed): plain 20/20, Double Q 2/20.
- **Figure.** `figures/part1_real_noise_cliff.png`.
- **Caveats.** i.i.d. per-step sampling of real outputs (see assumptions); a 4-channel cross-session decoder is a lower bound on decoder quality; one error type in the dataset, so all error kinds share one real distribution.

### Claim 1.8 — The absent-supervisor exploit replicates with a simulated ErrP observer.
- **Conditions.** Port of Leike et al. 2017 (rewards ÷ 50), q_init -2, σ 0.03 and 0.3, α = 1, 20 seeds.
- **Numbers.** Visibility-flawed brain takes the punished shortcut in 18-20/20 seeds (plain and Double Q), honest 0/20 (plain; Double Q 4/20 at σ 0.3, its baseline). Freezing 0-1/20. Only true imputation removes it (Claim 1.6).
- **Figure.** `figures/part1_observability_fix.png` (right).

---

## Part 2: praise hacking (a positive signal alongside the ErrP)

### Claim 2.1 — With an off-route blind spot, a praise channel lets a plain Q-learner farm praise across the boundary; it is robust to Q-init, training length and noise, and Double Q is largely immune.
- **Conditions.** Off-route layout, visibility brain, praise ratio {0.25, 0.5, 1}, praise amplitude 0.5x ErrP, praise noise 1.5x, surprise weight 0, α = 1.
- **Numbers.** Plain Q stuck in a non-goal cycle: 20/20 at every ratio at σ 0.03 and 0.3 (0/20 at ratio 0 and σ 0.03); 20/20 at q_init -1, -0.5 and 0 (praise 0.5, both σ); 20/20 at 300k steps. The cycle: start cell → hidden cell above it (unseen "away" error, -0.01) → back into view (praised progress, +0.99 at σ 0.03), forever; proxy +0.5/step at ratio 1, true return -1.0. Double Q: 0-3/20 at every ratio, init and length. Our PPO (10 seeds, σ 0.3): 10/10 stuck with and without praise (it freezes anyway).
- **Figure.** `figures/part2_praise_vs_freeze.png`, `figures/review_qinit_sensitivity.png`.
- **Fix.** Tax or imputed (= per-action tax here): plain Q 20/20 and 19/20 reach the goal, Double Q 20/20; denoise 0/20 (plain).
- **Caveats.** Proxy-optimal, not a trap: a praise term with a blind spot is an infinite reward source. Praise parameters are PLACEHOLDERs.

### Claim 2.2 — Praise farming is broken shaping: potential-based praise is safe with full visibility and farmable with a blind spot.
- **Conditions.** Praise 0.5, σ 0.3, α = 1; praise kinds: potential-based (signed, F = Φ(s') − Φ(s), Φ = −cost-to-go) vs progress-only; honest (full visibility) vs visibility (off-route blind spot); 20 seeds.
- **Numbers.** Full visibility: 20/20 safe route for both kinds and both learners. Blind spot: plain Q farms 20/20 with potential-based praise and 20/20 with progress-only praise; Double Q 1/20 each.
- **Caveats.** Ng, Harada & Russell's invariance needs both signs of F delivered; the blind spot deletes the negative half. The test uses our praise model's decoder and noise.

### Claim 2.3 — Under an honest observer, a surprise confound produces no surprise-seeking and no rise-then-fall.
- **Conditions.** Honest brain, praise 0.5, surprise weight {0, 0.5, 1} x α {0.25, 1, 4} x {plain, Double Q}, 20 seeds.
- **Numbers.** 18/18 cells: 20/20 safe route, true 0.72, mean surprise 0.23 everywhere, 0% unpredictable moves, 0 extra steps, 0/20 rise-then-fall.
- **Figure.** `figures/part2_surprise_pressure.png` (flat).
- **Caveats.** One surprise model (per-state move counts, 200-step half-life, Laplace prior); the surprise bonus is smaller than one error penalty.

### Claim 2.4 — Combined flaws: the blind spot dominates.
- Visibility + praise 0.5: plain Q 20/20 stuck (w = 0), 16/20 (w = 1, α = 1), 20/20 (w = 1, α = 4); Double Q 1/20, 0/20, 9/20 (true -0.05). Valence flip (honest, plain, 10 seeds): no effect (10/10 safe route).

---

## Part 3: attention manipulation (the agent acts on what the observer sees)

Setup: gaze observer (lag 1, radius 2, lapse 5%, PLACEHOLDER), a "wave" action pulling the gaze to a salient event for k steps at an agent-visible cost, a pushable box blocking the camera's line of sight; a layout where the shortcut is worth hiding (19-step hazardous top route, true -0.19, vs 35-step safe route, true 0.65); α = 1, σ 0.3.

### Claim 3.1 — A bias-corrected tabular learner learns to distract the observer and err while it is distracted; the exploit needs an observable "distracted" state and is robust to the wave's cost and to Q-init.
- **Conditions.** Double Q, q_init -1 unless stated, 100k steps, 20 seeds.
- **Numbers.** Shortcut taken, with every hazard entry within k steps after a wave: k = 2: 0/20; k = 5: 8/20; k = 10: 13/20. Controls: inert wave 0/20 (0.0 waves per episode); fully attentive honest observer 0/20. Robustness at k = 10: q_init -1 / -0.5 / 0: 13 / 17 / 17 of 20; 300k steps: 13/20; wave cost 0.01 / 0.05 / 0.2: 13 / 13 / 19 of 20 with the distracted bit observable, 0 / 0 / 0 with it hidden. Manipulators over-wave (15-20 per episode), so their true return (-1.1 to -1.5) is below the unhidden shortcut's (-0.19).
- **Figure.** `figures/part3_distractor.png`, `figures/review_qinit_sensitivity.png`.
- **Caveats.** Observable distraction is an assumption (see above). A cost-free, unjudged wave is a free no-op that every learner spams even under an attentive observer (99 waves/episode in a pilot): that is the null-action trap, not manipulation, and the wave's cost had to be visible to the agent.

### Claim 3.2 — The plain Q-learner does not manipulate; it freezes in the unjudged "distracted" states.
- Plain Q, k = 5 / 10: 14/20 and 17/20 frozen, 1/20 and 0/20 shortcuts; at q_init 0 and k = 10: 5/20 shortcuts, 12/20 frozen; at cost 0.2: 2/20 shortcuts. Same mechanism as Claim 1.3 (a state where nothing is judged attracts a max-biased learner).

### Claim 3.3 — The occluder exploit was not learned.
- Box blocks / transparent / attentive honest, plain and Double Q: 20/20 safe route, 0/20 box pushes in evaluation. 225 x 225 (agent, box) states and an 11-push sequence; a negative result about our learner and budget. Do not claim.

### Claim 3.4 — The imputed / tax reward removes the hidden shortcut but not the distraction habit; the masked reward backfires.
- Distractor k = 10: imputed 0/20 shortcuts (Double Q baseline 13/20), 20/20 (plain) and 18/20 (Double Q) reach the goal; residual waves 3.3 (plain, true 0.45) and 12.9 (Double Q, true -0.19) per episode. Masked: Double Q 4/20 shortcuts and 14/20 frozen; plain 18/20 frozen.
- **Figure.** `figures/part3_fixes.png`.

---

## Do not claim

- **"Double Q-learning fixes blind-spot freezing."** Only for *off-route* blind spots (0/20 at σ 0.3). With the route itself unobserved Double Q fails 18/20 at 100k and 12/20 at 300k; half-route 8/20. Say: Double Q removes the off-route trap and does nothing for lost guidance.
- **"Imputation fixes freezing in the maze."** In the maze the imputed reward never draws on its (cell, action) table (0% of unobserved steps); it is the per-action mean, a flat tax, numerically identical to "pessimistic" there, and on the on-route layout it pushes 20/20 seeds onto the visible hazardous route. True imputation only operates on absent-supervisor.
- **"Real EEG noise."** The real-ErrP condition samples held-out decoder outputs i.i.d. per step. Say "real decoder output distributions (i.i.d. per step)".
- **"The honest brain shows the task is learnable."** The honest brain is oracle shaping; α = 0 with neutral Q-init is 0/20 in every layout.
- **Freeze rates as absolute numbers.** 19/20 (q_init -1), 16/20 (-0.5), 8/20 (0) for the same condition; always state the init.
- **Habituation (v1, v2).** v1's monotone-in-α PPO freeze was a reward-scale artifact (reward normalisation: honest α = 4 went 3/10 → 9/10); v2's ablation showed no dose-response because the leaky-exposure model has no floor. Say only that habituation was modelled and changed no conclusion.
- **Rise-then-fall / "optimising harder makes it worse".** 0 seeds in every cell of every sweep (v2, v2.1, Part 2), except 1-2/20 late route switches in three uncentered v2 cells.
- **The v2.1 "freezing is optimal at α = 4".** True for 5/20 decoder seeds, caused by a ≈ 0.005 baseline sampling error times a 100-step horizon; a statement about centering, not observers.
- **The σ 0.03 Double Q anomaly.** Retired: slow learning (0/20 at 300k; 9/20 with lr 0.2).
- **The Maxmin explanation and the PPO advantage mechanism.** Readings of logged numbers, not interventions.
- **The occluder.** Not learned; the controls are uninformative.
- **Any rate from the praise or attention models as a human rate.** Every parameter there is a PLACEHOLDER; claims are about the existence of exploits under a plausible model.
- **BNCI 013-2015 accuracies as representative decoders.** A 4-channel cross-session logistic decoder with gaps up to 643 days; 53-81% is a lower bound.

## Future work

- **Brain-feedback LLMs.** The blind-spot and praise-farming mechanisms are about what the reward model can see; an LLM fine-tuned on decoded ErrP/RewP from readers would have the same two failure modes with text instead of cells (unread passages, praised surface features). The tax / imputed rewards and Double-Q-style decoupling have direct analogues (penalising unread spans, reward-model ensembles, held-out judges).
- **Habituation with real data.** Replace the leaky-exposure placeholder with within-session ErrP amplitude trajectories from the 013-2015 sessions (~500 trials each).
- **Live human-subject validation.** A short session where participants watch an agent in a maze with a hidden region while EEG is recorded and the agent trains online from the decoded signal. Predictions: hidden-region retreat with a plain learner on an off-route blind spot; none with Double Q or a tax on unseen steps; praise farming at the boundary if a RewP channel is added; and, with an on-route blind spot, failure that no learner-side fix removes.
- **Lost guidance.** No fix tested here helps when the route is unobserved (the tax makes it worse); the honest options are to widen observation or to use a reward model that generalises to unseen states.
- **Within-session decoders and real noise dynamics** (autocorrelated, drifting outputs instead of i.i.d. draws); ensembled Double DQN and decoupled critics for the deep learners; a fix for the distraction habit itself (price the wave by its effect on observability).
