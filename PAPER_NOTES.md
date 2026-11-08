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

