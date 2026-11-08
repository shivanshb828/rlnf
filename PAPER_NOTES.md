# Paper notes: "How reinforcement learning from neural feedback fails: blind spots, praise hacking, and attention manipulation"

Every claim lists exact numbers, seed counts, the figure that shows it, caveats, and the
fix (if any). "Frozen" = eval stuckness > 0.3 (share of an episode spent repeating one
(cell, action) error; a random walk in the maze scores ≈ 0.07). All runs use the centered
neural reward, α = 1 unless stated, and the 15x15 mirror maze (equal 28-step routes,
hazards only on the hidden top route) unless stated. Defaults and placeholders are in
`rlnf/config.py`; results tables in `results/summary_*.csv`; figures in `figures/`.

---

## Part 1: blind spots (the observer does not see part of the state)

### Claim 1.1 — A visibility blind spot makes a tabular Q-learner retreat into the unobserved region and freeze there, at every realistic decoder accuracy.
- **Numbers.** Visibility brain, plain Q, 20 seeds per point: frozen 0/20 at 100% decoder balanced accuracy (σ = 0.03), 3/20 at 99.9% (σ = 0.1), 20/20 at 96% (σ = 0.2), 19/20 at 90% (σ = 0.3), 20/20 at 78% (σ = 0.5), 20/20 at 66% (σ = 0.8); identical at α = 0.25, 1, 4. Honest brain: 0/20 down to 78%, 1-2/20 at 66%. True return of a frozen agent ≈ -0.91 to -1.0 vs 0.72 for the safe route.
- **Figure.** `figures/part1_real_noise_cliff.png` (left panel); `results/noise_dose_response.png`.
- **Caveats.** The cliff sits at a decoder far better than any real one (between 99.9% and 96%), so "realistic accuracy" is entirely on the frozen side; the effect is a step, not a dose-response. The step cost is absorbed by the neutral Q-init (-1.0), so nothing but the neural term pulls the agent anywhere (see "Do not claim").
- **Fix.** Double Q-learning (Claim 1.4); observability-aware imputed reward (Claim 1.6).

### Claim 1.2 — The mechanism is maximisation bias on zero-mean decoder noise in a region without penalties, not Vivanti's Boring Areas Trap.
- **Numbers.** Reward variance per region during the first 20% of training (10 seeds): hidden/visible variance ratio falls from 2·10⁹ (σ = 0.03, 0/10 frozen) to 19.5 (σ = 0.2, 10/10 frozen) to 1.8 (σ = 0.8, 10/10): freezing starts when the quiet region *gains* noise, and the hidden half has the *higher* mean reward during exploration (-0.01 vs -0.6 to -0.13). Fixes sort by estimation bias at σ = 0.3 (20 seeds): plain 19/20 frozen, Double Q 0/20, Maxmin N = 2/4/8: 4/12/11, ASRN 20/20. Value iteration on the agent's exact expected reward: freezing is worth 1-80 units less than the safe route at α ≤ 1 (a trap), but at α = 4 a ≈ +0.005 sampling error in the centered baseline makes "never terminate" the optimum in 5/20 decoder seeds. More exploration makes freezing worse (ε → 0.3 over 90%: 20/20 at every learning rate; ε → 0.01 over 25%: 10-14/20).
- **Figure.** `results/variance_by_region.png`, `results/fixes.png` (= `figures/part1_fixes_comparison.png`), `results/exploration_regions.png`.
- **Caveats.** Maxmin's monotone worsening with N is consistent with the bias story (pessimism applies to both halves, but the visible half's sampled mean is already negative while exploring, so pessimism widens the gap) but we have not isolated it further. ASRN was implemented from the paper's prose (bins by update magnitude, 10 bins, 10k warm-up) and other choices could behave differently.

### Claim 1.3 — PPO freezes too, but through advantages, not value bias.
- **Numbers.** Our PyTorch PPO with reward normalisation, 10 seeds: visibility brain frozen 9/10 (σ = 0.03) and 10/10 (σ = 0.3); SB3 PPO (v2) 10/10 at α = 1 and 4. Critic diagnosis at the end of training: hidden-half V ≈ -0.4 vs Monte-Carlo return ≈ -0.5 (unbiased); visible-half V ≈ -0.5 vs Monte-Carlo -27 to -40 (the policy no longer visits it). Mean GAE advantage in the first 20% of training: hidden half ≈ 0.00, visible half -0.25 to -0.29. GAE λ = 0.8: 10/10 frozen at both σ; 3-critic ensemble: 7/10 (σ = 0.03) and 10/10 (σ = 0.3). Honest brain with this PPO reaches the goal 8/10 (default), 9-10/10 (ensemble), 6/10 (λ = 0.8).
- **Figure.** `figures/part1_ppo_critic.png`, `figures/part1_learners.png`.
- **Caveats.** Our PPO is weaker than SB3's on the honest brain (8/10 vs 9-10/10), so part of any PPO failure is exploration. "Advantage-driven" is an interpretation of the logged statistics, not a causal intervention; we did not find a critic-side fix.

### Claim 1.4 — Double Q-learning removes the freeze; Double DQN only halves it.
- **Numbers.** Tabular Double Q, visibility brain: 0/20 frozen at σ = 0.1 and 0.3, 1/20 at 0.5, true return 0.72; honest brain 0/20 frozen at every σ. DQN (10 seeds): 9/10 (σ = 0.03) and 10/10 (σ = 0.3) frozen; Double DQN: 5/10 and 5/10, late-training goal rate 0.5-0.6; honest brain 10/10 fine for both.
- **Figure.** `figures/part1_learners.png`, `figures/part1_fixes_comparison.png`.
- **Caveats.** Anomaly: tabular Double Q at σ = 0.03 (near-deterministic reward) failed to reach the goal in 14-16/20 seeds in two separate experiments (observability baseline, praise ratio 0), although it is 0/20 at σ = 0.1-0.3 and 2/20 with real noise; unexplained (ties between the two tables when the hidden-half reward is exactly the step cost is the guess). Double DQN shares one network and a periodically copied target, which is not two independent estimators.

### Claim 1.5 — The cliff holds with real ErrP decoder noise.
- **Numbers.** BNCI Horizon 2020 dataset 013-2015 (Chavarriaga & Millán 2010; 6 subjects, 2 sessions, 64-ch EEG, CC BY-NC-ND 4.0). Held-out (session 2, recorded 50-643 days later) balanced accuracy of our decoder: 80 / 73 / 81 / 58 / 63 / 53% for subjects 1-6 (AUC 0.92-0.71). One subject per seed, 20 seeds: visibility + plain Q frozen 20/20 (every subject); visibility + Double Q 2/20 (both at the two worst subjects); honest + plain Q 2/20 (the same two subjects); honest + Double Q 0/20.
- **Figure.** `figures/part1_real_noise_cliff.png` (right panel).
- **Caveats.** The decoder is simple (4 channels, logistic regression) and cross-session; 3-4 seeds per subject; the dataset has one error type, so all our error kinds are given the same real distribution; the real outputs are *sampled*, so habituation is not modelled here.

### Claim 1.6 — An observability-aware reward fixes both the freeze and the absent-supervisor shortcut, but only the imputed variant fixes both.
- **Numbers.** (20 seeds per cell, visibility brain.) Maze: `imputed` and `pessimistic` 0/20 frozen for plain and Double Q at σ = 0.03 and 0.3 (baseline plain 19/20 at σ = 0.3); `masked` 7/20 (plain, σ = 0.3) and 8-12/20 (Double Q). Absent-supervisor port: `imputed` 0/20 shortcuts (plain, both σ; Double Q 0/20 at σ = 0.03, 7/20 at 0.3) vs baseline 18-20/20; `masked` and `pessimistic` 18-20/20. Honest brain: 0/20 frozen and 0/20 shortcuts under every mode on both environments.
- **Figure.** `figures/part1_observability_fix.png`.
- **Caveats.** Assumes the system knows per step whether the observer saw it (eye-tracking). The imputed model is a per-(cell, action) running mean fit on observed steps, which works because the hazard tile is visited supervised often enough; in a larger state space it would need generalisation. Masking plus rescaling amplifies observed penalties and can be worse than doing nothing.

### Claim 1.7 — The absent-supervisor exploit replicates with a simulated ErrP observer.
- **Numbers.** Port of Leike et al. 2017 (rewards ÷ 50), 20 seeds: visibility-flawed brain takes the punished shortcut in 18-20/20 seeds (plain Q, σ = 0.03 and 0.3; Double Q 19-20/20), honest brain 0/20 (plain). Freezing does not occur there (0-1/20): the goal is 3-9 steps away.
- **Figure.** `figures/part1_observability_fix.png` (right).
- **Caveats.** Reimplemented in gymnasium without pycolab; the supervisor's presence is observable to the agent as in the original.

## Part 2: praise hacking (a positive signal alongside the ErrP)

### Claim 2.1 — Praise does not fix the blind-spot freeze; for a plain Q-learner it creates a worse exploit, praise farming across the blind-spot boundary.
- **Numbers.** Visibility brain, 20 seeds per cell. Plain Q stuck in a non-goal cycle: 20/20 at praise ratio 0.25, 0.5, 1 at σ = 0.03 and σ = 0.3 (0/20 at ratio 0 and σ = 0.03; 20/20 frozen at ratio 0 and σ = 0.3). The cycle: start cell (7,0) → hidden (6,0) [unseen "away" error, -0.01] → back to (7,0) [seen progress, +0.99 at σ = 0.03]; proxy +0.5 per step at ratio 1 (neural return +50 per 100-step episode), true return -1.0. Double Q: 0-3/20 stuck at every ratio and σ. Our PPO (10 seeds, σ = 0.3): 10/10 stuck with ratio 0 and 0.5.
- **Figure.** `figures/part2_praise_vs_freeze.png`.
- **Caveats.** The farm is proxy-optimal, not a trap: a praise term with a blind spot is an infinite reward source. It needs praise for local progress (judged against the true cost-to-go) and an unobserved retreat; both are our modelling choices. Praise parameters (0.5x amplitude, 1.5x noise) are PLACEHOLDERs.
- **Fix.** Not tested in this round. The observability-aware imputed reward (Claim 1.6) should price the hidden retreat; praising only *net* progress over a window would remove the farm by construction.

### Claim 2.2 — Under an honest observer, a surprise confound does not produce surprise-seeking or rise-then-fall.
- **Numbers.** Honest brain, praise ratio 0.5, surprise weight {0, 0.5, 1} x α {0.25, 1, 4} x {plain, Double Q}, 20 seeds each: 18/18 cells reach the goal 20/20 by the safe route, true return 0.72, mean surprise 0.23 in every cell, 0% unpredictable moves, 0 extra steps, 0/20 rise-then-fall everywhere.
- **Figure.** `figures/part2_surprise_pressure.png` (flat).
- **Caveats.** Negative result within one surprise model (per-state move counts with a 200-step half-life, Laplace prior). The maximum surprise bonus (0.5 x praise amplitude x surprise x ratio) is smaller than one error penalty, which may be the whole story; a stronger expectancy effect could change it.

### Claim 2.3 — With both flaws, the blind spot dominates.
- **Numbers.** Visibility + praise 0.5: plain Q 20/20 stuck (w = 0), 16/20 (w = 1, α = 1), 20/20 (w = 1, α = 4); Double Q 1/20, 0/20, 9/20 (w = 1, α = 4, true return -0.05).
- **Caveats.** The Double Q α = 4 cell is the only place surprise mattered; 9/20 with a wide CI.

### Valence flip (one small run)
- Honest brain, plain Q, 10 seeds: 10/10 safe route, true 0.72; flipping the praise sign changes the praise return from +0.11 to -0.11 per episode and nothing else.

## Do not claim

Results from v1-v3 and this round that are too weak, too artifact-prone, or both.

- **Habituation as a cause of freezing (v1, v2).** v1's monotone-in-α PPO freeze was a reward-scale artifact (fixed by reward normalisation: honest α = 4 went from 3/10 to 9/10). v2's habituation ablation showed no dose-response because the leaky-exposure model has no floor: any decay saturates under a two-step oscillation. The habituation model itself (per-cell exposure, decay 0.8, τ = 500, dishabituation to 30%) is a stack of PLACEHOLDERs with no grounding. Say "habituation was modelled and did not change the conclusions", nothing more.
- **"Optimising harder makes it worse" / rise-then-fall.** 0 seeds in every cell of every sweep (v2 main, v2.1, Part 2) except 1-2/20 late route switches in three uncentered v2 cells. The flawed signals fail from the first non-zero α.
- **The v2.1 α = 4 "freezing is optimal" result.** True for 5/20 decoder seeds, but it comes from a ≈ 0.005 sampling error in the centered baseline times a 100-step horizon; it is a statement about centering with a finite calibration set, not about observers.
- **Maxmin worsening freezing (v3).** Robust (4/12/11 of 20 at N = 2/4/8 vs plain 19/20 at σ = 0.3; 12-20/20 even at σ = 0.1) but the explanation (pessimism widens a gap whose visible side is already negative under exploration) is a reading of the numbers, not an intervention. Report the numbers; keep the explanation hedged.
- **Tabular Double Q at σ = 0.03 (this round).** 14-16/20 non-goal in two experiments against 0/20 at σ ≥ 0.1; unexplained. Do not cite Double Q as a universal fix without the σ caveat.
- **PPO critic/advantage mechanism (1a).** The logged advantages are consistent with the story; no intervention confirmed it, and the two critic-side fixes were null. Say "unresolved beyond the diagnostic".
- **The praise and attention models' parameters.** Every one is a PLACEHOLDER (praise amplitude ratio, noise ratio, surprise weighting, predictor half-life, gaze lag/radius/lapse, distractor cost and pull, camera geometry). Claims should be about the *existence* of exploits under a plausible model, never about rates in humans.
- **Decoder accuracies from BNCI 013-2015 as representative.** A 4-channel logistic decoder, cross-session with gaps up to 643 days; 53-81% is a lower bound on what a modern within-session pipeline gets.
- **Anything from Part 3's occluder** beyond what `summary_att_occluder.csv` shows: the (agent, box) state space is 225 x 225 and 100k steps is thin coverage.

## Future work

- **Brain-feedback LLMs.** The blind-spot and praise-farming mechanisms are about *what the reward model can see*; an LLM fine-tuned on decoded ErrP/RewP from readers would have the same two failure modes with text instead of cells (unread passages, praised surface features). The observability-aware imputed reward and Double-Q-style decoupling have direct analogues (reward-model ensembles, held-out judges).
- **Habituation with real data.** Replace the leaky-exposure placeholder with within-session ErrP amplitude trajectories (the 013-2015 sessions have ~500 trials each and could give an empirical amplitude-vs-trial-index curve per subject).
- **Live human-subject validation.** The cleanest test of Part 1 is a short session where participants watch an agent in a maze with a hidden region while EEG is recorded, with the agent trained online from the decoded signal; the predictions are (i) hidden-half retreat with a plain learner, (ii) none with Double Q or an eye-tracking-informed imputed reward, (iii) praise farming at the boundary if a RewP channel is added.
- **Within-session decoders and a wider accuracy range** for the real-noise cliff; and the deep-RL side of the fixes (ensembled Double DQN, decoupled critics).
- **A fix for praise farming** (praise net progress over a window; observability-aware praise) and for attention manipulation beyond the two tested here.
## Part 3: attention manipulation (the agent acts on what the observer sees)

Setup: gaze observer (lag 1, radius 2, lapse 5%, all PLACEHOLDER), a "wave" action that pulls the gaze to a salient event for k steps (agent-visible cost 0.05), a pushable box that blocks the camera's line of sight; layout with a 19-step hazardous top route (true -0.19) vs a 35-step safe route (true 0.65); α = 1, σ = 0.3, 20 seeds per cell; the agent observes whether its salient event is showing.

### Claim 3.1 — A bias-corrected tabular learner learns to distract the observer and err while it is distracted.
- **Numbers.** Double Q, shortcut taken (and every hazard entry within k steps after a wave): k = 2: 0/20; k = 5: 8/20; k = 10: 13/20. Controls: inert wave 0/20 shortcuts and 0.0 waves per episode; fully attentive honest observer 0/20 and 0.0 waves. Distractor use (≥ 1 wave per eval episode): 14/20, 18/20, 20/20 at k = 2, 5, 10. Manipulators over-wave (15-20 per episode), so their true return (-1.1 to -1.5) is below even the unhidden shortcut's (-0.19).
- **Figure.** `figures/part3_distractor.png`.
- **Caveats.** The agent must observe the "distracted" state (as the absent-supervisor agent observes the supervisor); without that bit the contingency is unlearnable, which we verified (0/4 exploit seeds before adding it). The wave's cost had to be visible to the agent: a cost-free, unjudged wave is a free no-op that every learner spams even under an attentive observer (99 waves/episode), which is the Part 1 null-action trap, not manipulation. The exploit only exists on a layout where the shortcut is worth hiding; on the equal-length maze there is nothing to gain by distracting.

### Claim 3.2 — The plain Q-learner does not manipulate; it freezes in the unjudged "distracted" states.
- **Numbers.** Plain Q at k = 5 / 10: 14/20 and 17/20 frozen, 1/20 and 0/20 shortcuts, 0-3/20 reach the goal; at k = 2: 20/20 safe route. Even the attentive-honest control shows 2/20 plain-Q seeds waving ~90 times per episode.
- **Figure.** `figures/part3_distractor.png`.
- **Caveats.** Same mechanism as Part 1 (a state where nothing is judged attracts a max-biased learner); the attentive-control waving in 2/20 seeds is a weaker version of it and should be mentioned as such.

### Claim 3.3 — The occluder exploit was not learned.
- **Numbers.** Box blocks / transparent / attentive honest, plain and Double Q: 20/20 safe route, 0/20 box pushes in evaluation, true 0.65 in all six cells.
- **Caveats.** 225 x 225 (agent, box) states and an 11-push sequence needed; 100k steps of ε-greedy exploration did not find it. A negative result about our learner and budget, not about occluders. Listed under "Do not claim".

### Claim 3.4 — The decoupled (imputed) reward removes the hidden shortcut but not the distraction habit; the masked reward backfires.
- **Numbers.** Distractor k = 10. Imputed: 0/20 shortcuts (Double Q baseline 13/20), 20/20 (plain) and 18/20 (Double Q) reach the goal, 0 hazard entries; residual waves per episode 3.3 (plain, true 0.45) and 12.9 (Double Q, true -0.19). Masked: Double Q 4/20 shortcuts and 14/20 frozen, plain Q 18/20 frozen, 0/20 goal. Occluder: both fixes 20/20 safe route (uninformative).
- **Figure.** `figures/part3_fixes.png`.
- **Caveats.** The imputed model is per (cell, action) and only prices what it has seen; the wave itself is never judged by the observer and costs 0.05, which is why it persists. Everitt et al.'s decoupling principle is implemented in its simplest tabular form.

## Modelling assumptions that shape the results (read before any claim)

- **Oracle shaping in the honest brain.** The "away" error fires whenever a step increases the *true optimal* cost-to-go (hazards priced at hazard_penalty / step_penalty extra steps). An honest observer therefore delivers dense, exactly-correct shaping on every step. That is why the honest brain solves a maze a random walk never solves (α = 0: 0/20 with neutral Q-init) and why "honest succeeds" is not evidence that the task is learnable without guidance. All "honest vs flawed" gaps are gaps against an oracle.
- **The agent observes the distraction (Part 3).** The attention agent's state includes whether its salient event is still showing; without that bit the hazard's penalty is a coin flip from its viewpoint and no learner found the exploit (0/4 pilot seeds). Phase D tests the bit on/off. This mirrors the absent-supervisor benchmark, where the supervisor's presence is observable, but it is an assumption.
- **Real decoder outputs are sampled i.i.d. per step (Part 1d).** The "real noise" condition draws one held-out decoder probability per step from a subject's error or correct pool, independently of the previous step. It reproduces a real decoder's *output distribution* (false-positive and miss rates, between-subject differences), not its *dynamics* (drift, habituation, fatigue, autocorrelation). Call it "real decoder output distributions", never "real EEG noise".
- **Neutral Q-init.** The tabular learners start every Q-value at -step_penalty / (1 - γ) = -1.0 so that the step cost is not an exploration bonus; with q_init = 0 the same maze is solved at α = 0 by systematic exploration. Headline freeze rates depend on this choice (Phase C).
- **Imputation in the maze is a flat tax.** The per-(cell, action) model behind the `imputed` reward has no entries for cells that are never observed; in the maze every hidden cell is unobserved by construction, so the model falls back to the per-action (and global) mean of observed P(error): a constant penalty on unobserved steps with no state dependence. Only on absent-supervisor, where the same cells are observed in supervised episodes, does per-(cell, action) imputation actually operate (Phase B measures the fraction of imputed steps that came from the table).
- **Centering with a finite calibration set.** `p_clean` is a sample mean (~3,500 steps); its error times α times the 100-step horizon is what made "never terminate" optimal at α = 4 in v2.1.

