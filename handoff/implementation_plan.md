# ActSafe-CT Implementation Plan (current as of 2026-06-29)

This is the single source of truth for "where we left off." Start here on any new device
or chat. The historical cartpole investigation is preserved verbatim in **Appendix A** at the
bottom — it is superseded by the 2026-06-23 PointGoal pivot but its findings are still cited.

Companion docs in this `handoff/` folder (mirror of the auto-memory): `MEMORY.md` (index),
`project_strategy_2026-06-23.md` (the decisions), `project_paper_direction.md`,
`project_bugs_fixed.md`, `project_ct_architecture.md`, `project_cluster_infra.md`,
`user_profile.md`, `feedback_style.md`.

---

## NOTE FOR SUPERVISOR MEETING — reward-discount confound (deferred 2026-06-30)

**Status: known, understood, NOT fixed. Deliberately deferred so it doesn't block TASE.**

The fixed-AR reward numbers are **not directly comparable across control frequencies**, because
the agent discounts per *decision* (flat `discount=0.99` per agent step), not per unit physical
time. At `action_repeat=k` one decision spans `k` base steps, so:
- the effective physical horizon scales with `k` (AR1 ≈ 100 base steps, AR8 ≈ 800), and
- the discounted reward objective scales ~linearly in `k`.

This is why AR8's reported reward (~20) looked higher than AR1's (~8) — largely a units/horizon
artifact, not a competence gap. **The cost/safety result is unaffected:** realized cost is summed
undiscounted over the same 1000 base steps, and the safety budget is already made frequency-fair
via budget scaling in `make_actor_critic.py`. So the *headline (cost rises as frequency drops,
crosses budget)* stands; only the reward *axis* is confounded.

**The fix (if/when we want it):** hold the physical horizon fixed by anchoring the discount at the
paper's `action_repeat=2`: `gamma_agent = discount ** (action_repeat / 2)`. This keeps a ~200-base-
step horizon at every frequency, reproduces the paper at AR2, and would make reward land at paper
level (~15) flat across the sweep. Implemented and validated on 2026-06-30, then **reverted** to
keep the baseline byte-identical to the reported figure and avoid stacking unverified changes
before the meeting. The continuous-time/TASE path already discounts correctly per base step
(`base_discount ** dt_ratio`), so this only ever mattered for the discrete study.

**Talking point for the meeting:** "the reward axis across fixed frequencies is confounded by
per-decision discounting; I have the frequency-fair correction ready (anchored at the paper's AR),
but the safety result — which is the contribution — doesn't depend on it."

---

## WHERE WE ARE (2026-06-24)

**Strategy (decided 2026-06-23, see `project_strategy_2026-06-23.md`):** the whole story
moves OFF dm_control cartpole ONTO Safety-Gym PointGoal. Cartpole could not stress the safety
constraint (cost stayed under budget at every action_repeat → the safety-frequency U-shape
physically cannot appear there). PointGoal is ActSafe's home task and its hazards+momentum
should make a long open-loop hold actually cause violations. **All-in on ICLR 2027** (~Sept
2026, ~10 weeks) with a kill gate at ~week 4: if the AR violation U-shape doesn't show on
PointGoal, rescope to a workshop.

**First PointGoal run (AR=1, seed=0, project `actsafe-ct-pointgoal`):** safety mostly (not
always) satisfied, but **reward ≈ 0** — a do-nothing-ish policy. Diagnosed and fixed this
session (below). A cluster crash also appeared; diagnosed as a wandb/NFS shutdown fragility,
NOT a training bug (below).

### RUN ANALYSIS 2026-06-25 (AR=2 safety_gym runs, after the config fix) — BEHAVING LIKE THE PAPER
Pulled the latest `actsafe-ct-pointgoal` runs via wandb API (entity arnavsukhija-eth-zurich).
- **`n15wz73m`** (full 5M, AR=2): exploitation `train/objective` mean 12.3 / max 15.9 (✅ paper-level),
  `train/cost_return` mean 28.2 / **median 27.7** / max 47.4.
- **`o2kcht21`** (2.45M, AR=2): obj mean 6.3, cost mean 20.1 / median 19.8.
- **`8qazikmi` / `aog8hvqd`** (AR=1, older frankenstein config so obj≈0): cost mean 27.2 / 23.7,
  **median 25.5 / 22.0**, max 49 / 55.
- **Interpretation: this is NORMAL ActSafe soft-constraint behavior, matching the paper — NOT a bug.**
  Medians straddle the budget (25); LBSGD drives cost TOWARD the budget and it oscillates around it
  with tail spikes from episode variance. `lbsgd/safe≈0.92`-vs-occasional-overshoot is the nature of a
  SOFT constraint (feasibility in expectation; realized episodic cost is noisy). The paper's Fig-4 cost
  curves hover on the budget line the same way. The user's original "safety mostly met" read was right.

### BUDGET-SCALING CONCERN — RESOLVED & EXONERATED (the user asked to check this first, 2026-06-25)
User hypothesis: the action_repeat budget scaling (5.0 at AR=2 vs paper's 2.5) is "too loose" → critic
thinks safe → violations. VERDICT: not supported, by two decisive checks.
- THEORY: 5.0@AR2 ⟺ same physical 25-cost episode budget as 2.5@AR1 (critic discounts per AGENT step,
  ActionRepeat SUMS R raw costs/agent step, episode = time_limit/R agent steps; the 2× in threshold
  cancels the 2× per-step cost).
- EMPIRICAL #1: if 5.0 were 2× too loose, AR=2 cost would be ~2× AR=1 cost. It is NOT (both ~20–28
  mean) → scaling holds physical cost invariant, as designed.
- EMPIRICAL #2: the same overshoot appears at AR=1, where our formula is BYTE-IDENTICAL to the paper's
  2.5 (`8qazikmi`: 53% over budget). A formula cannot cause a discrepancy already present where the
  formula is unchanged.
- `constraint_pessimism=0.001` is the PAPER's own value (set by Yarden As, original ActSafe author,
  2024-10-04 commit 3d51a45; fork never touched it). So it is NOT a regression and "raise pessimism to
  fix a safety bug" was an over-diagnosis — WITHDRAWN. Raising pessimism is an OPTIONAL knob if we want
  a cleaner sub-budget margin for the AR sweep, not a bug fix.
- NET: nothing to fix on budget or pessimism; both paper-faithful. Keep the scaling. `ba1b659m`
  (latest) crashed with ZERO history — a real early crash, watch for recurrence.

### FINAL SWEEP DESIGN (decided 2026-06-25, after full back-and-forth with the user)
Goal stated by the user: "same budget to compare all agents -> under the same safety budget, do
violations vary with action_repeat?" The fair invariant is the SAME PHYSICAL budget (realized
undiscounted episode cost <= 25) at EVERY action_repeat — which REQUIRES the discounted threshold
to scale by action_repeat (since V_c scales ~linearly with R for fixed physics: J2 = 2*J1).
- DECISION: KEEP the budget scaling (make_actor_critic.py divides by time_limit/action_repeat).
  Briefly reverted to paper formula then RE-APPLIED once the user clarified the fairness goal — the
  scaling IS that goal (one budget d=25, same physical bar at every AR). Paper formula would impose
  a TIGHTER physical budget (25/R) at higher AR -> unfair/confounded.
- PESSIMISM: raised constraint_pessimism 0.001 -> 0.1 in safe_goal_ar_study.yaml, held CONSTANT
  across the sweep. 0.001 (paper) parks the agent AT budget so realized cost overshoots (~28 at AR=2);
  0.1 gives a modest honest margin without the exploration-starvation risk of 0.5 (coupled to OPAX,
  exploration.py:49). Margin is ActSafe's REAL safety mechanism (UCB); 0.5 was judged too risky for an
  unbabysat overnight run. Bump to 0.3-0.5 in round 2 if cost still >=25 but reward survives.
- WINDOW-DISCOUNTING REJECTED: discounting ActionRepeat's within-window repetitions does NOT give
  "correct discounting" (would also need agent-step discount = base^R), is a ~0.5% effect at gamma=0.99,
  breaks the clean physical-cost metric, AND reintroduces the load-bearing budget-gaming bug (longer
  hold reports LESS cost -> high AR looks safer -> masks the signal). The undiscounted sum is correct
  for measuring physical episode cost (the fair bar). The PRINCIPLED dual of the budget scaling is
  DISCOUNT scaling (safety_discount = base_safety_discount ** action_repeat, what the CT path does);
  kept in back pocket as a robustness check, not used in round 1.
- LBSGD: NOT reverted. Audited vs paper (dcbe264): fork is already numerically equivalent on the
  discrete path (Adam normalizes the lr/base_lr & backup_lr rescalings away; step_scale is inert at
  1.0). Only real diffs are the NaN guard (keep — protects log(<=0) in the violating regime) and
  fallback-eta decrease (cosmetic). A 6-file revert is pure risk for zero discrete benefit. Left as-is.
- CALIBRATION: skipped for round 1 (trend is the GO/NO-GO; pristine baseline is a round-2 nicety).
- LAUNCH (Euler login node, no render): full 4x3 sweep fired overnight.

### SWEEP RESULTS — κ=0.1, 3 seeds × AR{1,2,4,8} (analyzed 2026-06-27) — HYPOTHESIS SUPPORTED (caveated)
Pulled exploitation-tail cost (mean of last 20 logged episodes) for the 12 COMPLETED runs (all
reached 5M steps). Budget = 25. `train/cost_return` IS the physical undiscounted episode cost
(ActionRepeat sums raw), directly comparable to 25 at every AR. Completed run ids per (AR,seed):
AR1 {0lb6jc94,45ybj63u,bw56f9o4}, AR2 {9pfcszkg,6n07mgav,we4dh5zl}, AR4 {ijj34j25,2ada9iio,l4br31lz},
AR8 {sfcy0uod,fkbf62km,b0xsefef}.

| AR | cost (all 3 seeds) | cost (reward-healthy) | reward | verdict |
|----|----|----|----|----|
| 1  | 19.5 | **15.7** | 7.0  | safe |
| 2  | 17.7 | **17.7** | 10.1 | safe |
| 4  | 19.8 | **19.8** | 19.2 | safe (at margin) |
| 8  | 32.1 | **27.4** | 18.0 | **VIOLATES** (60–70% episodes > 25) |

- **HONEST FRAMING (corrected 2026-06-27 after user pushback):** only AR8 actually VIOLATES
  (27.4 > 25). AR1–4 are UNDER budget = constraint SATISFIED with an eroding margin. Do NOT call
  AR1–4 "violations." The motivation is: a fixed safe-RL agent (same algo, same d=25, same κ)
  SATISFIES the constraint at high control frequency and FAILS it at low frequency; realized cost
  rises monotonically (15.7 → 17.7 → 19.8 → margin gone → 27.4) and the constraint breaks at AR8.
  That answers the kill-gate ("does there exist a frequency where the same safe setup can't hold
  the budget?") → YES at AR8. The under-budget points are the eroding margin that makes AR8 a
  TREND, not a one-off. GO.
- TWO things muddying the raw curve (both the user flagged independently):
  1. **κ=0.1 margin compresses AR1–4.** The UCB margin parks realized cost a fixed ~5–7 under
     budget, so for AR1–4 the blind-window excess < margin → all safe, differences tiny. Only at
     AR8 does the open-loop hold's excess exceed the margin and break through. (Recall: at κ=0.001,
     AR2 already sat at ~28 — pessimism MASKS the low-AR portion of the trend, the user's intuition.)
  2. **2/12 seeds collapsed — DIAGNOSED 2026-06-27, it is ACTOR ENTROPY COLLAPSE, NOT pessimism
     and NOT LBSGD.** Pulled curves for collapsed (0lb6jc94 AR1-s0, fkbf62km AR8-s1) vs healthy
     (45ybj63u, b0xsefef). Evidence: (a) κ identical across all 12, 10/12 healthy → not a
     too-high-κ effect; (b) LBSGD `eta` decays 0.013→0.002 IDENTICALLY in collapsed & healthy, `lhs`
     ~0.01–0.02 both → penalizer is NOT blowing up; (c) collapsed runs pin `agent/actor/entropy` at
     EXACTLY −16.57 (the entropy floor, same value for both the AR1 and AR8 failure — actor
     saturated to bang-bang, tanh pinned at ±1) vs healthy −12 to −15 and moving. Saturated actor
     earns ~0 reward AND flails into hazards → cost SWINGS high (fkbf62km: 18→56→102→28). (d) ROOT
     CAUSE: actor loss `safe_actor_critic.py:268-269` is pure `-objective` with NO entropy
     regularizer (`actor_entropy` is logging-only) → nothing pulls a saturating actor back. Classic
     Dreamer-style imagination-actor collapse, seed-dependent. These are optimization failures, not
     a safety signal; exclude from the safety claim with a footnote.

### ADAPTED STRATEGY (decided 2026-06-27) — sharpen the motivation, then build TASE on it
1. **Pessimism sweep, not a single κ.** Run κ ∈ {0.0/0.001, 0.1} × AR{1,2,4,8} (drop the 3 seeds
   to 2 if compute-bound, but ADD seeds for the collapse-prone cells). Story becomes "the
   frequency→violation effect is robust across the safety margin": at κ≈0 the ramp appears already
   at low AR (all violate, monotonically worse); at κ=0.1 the margin holds low AR and the
   constraint breaks at AR8. The κ-axis turns the confound INTO a result.
2. **Fix the collapsed seeds — it's ACTOR ENTROPY COLLAPSE (diagnosed above), so target THAT.**
   (a) Add a small ENTROPY BONUS to the actor loss (`safe_actor_critic.py:268-269`, currently pure
   `-objective`) — the direct guard against the saturation collapse; highest leverage. (b) Bump
   seeds to ≥5 so a collapse is an outlier not 1-of-3. NOTE: lowering κ is NOT the targeted fix
   (evidence exonerates pessimism); don't reach for it for this. Report tail-mean + frac>budget,
   exclude reward-dead seeds from the safety claim with a footnote.
3. **Extend the ladder to AR=16** to confirm the ramp keeps climbing past the AR8 break — a steeper
   curve is a stronger figure than a single threshold crossing.
4. **Fix the slurm requeue crash-on-launch.** ~50 of the 73 runs are empty `crashed` requeue
   attempts (4h-interval waves, total_steps=-1) — they died on resume before logging. Wasted
   compute and clutters the project. Investigate the pickle-resume/wandb-init path on requeue
   before the next sweep (see project_cluster_infra.md).
5. METRIC HYGIENE: never use the single last-point summary (noisy, and None on crashed runs); use
   exploitation-tail mean over last ~20 episodes + frac>budget, aggregated over seeds.

NEXT ACTION: relaunch as a κ × AR grid (item 1) with ≥5 seeds on the collapse-prone cells, after a
quick look at the requeue-crash path (item 4) so we don't burn another ~50 empty runs.

### DECIDED 2026-06-27 (strategy session) — claim framing, launch, switch-cost deferral
- **CLAIM FRAMING (resolves the user's "is it a violation if we're under 25?"):** lead with the
  CONTINUOUS claim — "lower control frequency → monotonically higher incurred safety cost" (budget-
  independent, undeniable) — and treat VIOLATION as its extreme ("...and crosses d=25 at low
  frequency"). Do NOT call under-budget AR1–4 "violations"; the headline is "a SAFE agent crosses
  from safe to unsafe purely by lowering control frequency." One crossing = existence; AR16 makes
  it a trend; the κ-axis makes it robust.
- **LAUNCH (the κ × AR grid):** `training.action_repeat=1,2,4,8,16 training.seed=0,1,2,3,4
  agent.sentiment.constraint_pessimism=0.001,0.1` → 50 jobs. κ=0.001 row = "violation grows with AR
  from AR=1"; κ=0.1 row = "safe agent breaks at AR=8". Two-panel robustness figure. 5 seeds
  over-provision vs the ~15–20% actor-collapse rate (entropy fix still deferred). AR16 verified
  safe to run (TimeLimit wraps base env, ActionRepeat breaks on done → clean partial last window).
- **SUPERVISOR DELIVERABLE:** `handoff/supervisor_update_2026-06-27.md` written (table + framing +
  caveats + the 3-step next plan). Present current κ=0.1 data as motivation while the grid lands.
- **THE LOAD-BEARING MOTIVATION GAP (supervisor will poke it):** at AR1 the agent is BOTH safe and
  high-reward, so "why not just always AR1?" stands until there is a COST-OF-CONTROL axis (high
  frequency must be expensive: compute/energy/actuation). Without it fixed-AR1 dominates and TASE
  has nothing to beat. This is the real gap, not the violation wording.
- **SWITCH COST DEFERRED TO THE TASE PHASE (user, 2026-06-27):** do NOT add it to this fixed-freq
  sweep. SwitchCostWrapper IS the CT/variable-dt path (episodic_async_env.py:208 skips ActionRepeat
  when present), so the cost-of-control axis and the adaptive method arrive together. When picked
  up, TWO must-checks on the wrapper vs Safety-Gym: (1) it currently DISCOUNTS cost within the hold
  window (wrappers.py:191-192) = the load-bearing gaming bug — must sum RAW cost (like
  ActionRepeat, wrappers.py:27) or the agent hides cost by stretching dt and TASE makes safety
  WORSE; (2) it penalizes REWARD (wrappers.py:202) not the cost channel — decide separate-accounting
  vs fold-into-return before building the safety-performance frontier plot.
- **DOES TASE FIX THE TRADEOFF OUT OF THE BOX? No — it is DESIGNED to, gated on 3 things:**
  (i) gaming-proof cost accounting (the within-window discount bug above), (ii) state-predictable
  danger so the agent picks small dt BEFORE entering a hazard (ActSafe's pessimistic safety critic
  + epistemic uncertainty is the right tool), (iii) a real cost-of-control axis so the tradeoff
  exists at all. Making it work IS the paper; the fixed-frequency sweep is its motivating baseline.

### FULL κ × AR GRID LANDED (analyzed 2026-06-29) — HYPOTHESIS CONFIRMED, ROBUST ACROSS PESSIMISM
The 50-cell grid (κ∈{0.001,0.1} × AR∈{1,2,4,8,16} × 5 seeds) completed; all 50 cells reached 5M
steps (wandb marks them `crashed` because the slurm job is killed at the end, but training is
complete — the 5M-step run per (κ,AR,seed) is the one to read). Metric = exploitation-tail mean of
`train/cost_return` (last 8 logged eval-epochs, stitched across requeue segments), which is the RAW
undiscounted physical episode cost (ActionRepeat sums raw), directly comparable to d=25 at every AR.
Median over 5 seeds (robust to the entropy-collapse outliers); `n>25` = how many of 5 seeds violate.
Numbers below are reproducible via `handoff/figures/control_frequency_safety.py` (pulls all 138 runs via the wandb
API, tail-averages, and renders the figure); regenerate if seeds/segments change.

| κ | AR=1 | AR=2 | AR=4 | AR=8 | AR=16 |
|---|---|---|---|---|---|
| 0.001 (no margin) | 15.9 (1/5) | 26.3 (3/5) | 21.6 (0/5) | **30.3 (4/5)** | 25.4 (3/5) |
| 0.1 (modest margin) | 14.5 (0/5) | 18.4 (0/5) | 21.2 (1/5) | **25.3 (3/5)** | **26.3 (3/5)** |

(cell = median tail cost; parenthetical = seeds violating d=25.)
Figure: `handoff/figures/control_frequency_safety.png` (3 panels: cost-vs-AR with d=25 line +
per-seed scatter; violation fraction; reward U-curve).

- **HYPOTHESIS CONFIRMED (the user's read is right):** median cost rises with control frequency
  dropping, at BOTH pessimism settings, and crosses d=25 between AR=4 and AR=8. **κ=0.1 is the clean,
  strictly-monotonic panel: 14.5 → 18.4 → 21.2 → 25.3 → 26.3, violations 0/5,0/5,1/5,3/5,3/5** —
  a sharp safe→unsafe transition driven by nothing but control frequency. This is the headline panel.
  κ=0.001 confirms the effect survives removing the safety margin but is NOISY/non-monotonic (AR2
  bumps to 26.3, AR16 dips to 25.4) because with no margin the entropy-collapse seeds flail and their
  cost varies wildly — so lead with κ=0.1, present κ=0.001 as the robustness check, not the figure.
- **REWARD is U-SHAPED in AR (a real, separate finding):** median reward ≈ AR1 low (~2–4) → AR4 peak
  (~19–20) → AR16 low (~0–4), at both κ. TWO compounding causes (both verified in the data): (1) the
  actor's `discount=0.99` is per AGENT step, so at high frequency the goal is many agent-steps away
  and heavily discounted (γ-horizon ≈100 agent-steps ≪ 1000-step episode), making credit assignment
  hard; at AR4 the goal sits well inside the discount horizon. (2) Consequently the high-frequency
  cells are far more entropy-collapse-prone — several AR1/AR2 seeds sit at obj≈0–1 (saturated),
  dragging the median down, while all AR4 seeds are healthy ~18–21. AR16 falls again because the
  open-loop hold is too coarse to fine-tune at the goal. CONSEQUENCE FOR THE STORY: AR4 is the
  fixed-frequency "sweet spot" — best reward AND still ~at budget (21.2 ≈ 25). The tension TASE
  exploits is real:
  to get AR4-or-better reward you are pushed toward low frequency, and AR8 already breaks safety. A
  fixed frequency cannot sit at "AR4 reward + AR1 safety margin." (Caveat: the AR1 low reward is
  partly an optimization/horizon artifact, so do NOT lean the motivation on "AR1 is reward-bad" —
  lean on "cost rises monotonically and crosses the budget," which is the clean, robust claim.)
- **Entropy-collapse outliers still present** (the −16.57 saturation failure, diagnosed 2026-06-27):
  visible as the per-cell `obj`≈0/negative seeds (e.g. κ=0.1 AR16 has 3 collapsed seeds → median
  reward −0.1). Using the MEDIAN over 5 seeds already absorbs them; the entropy-bonus fix
  (safe_actor_critic.py:268-269) is still the deferred round-2 robustness item, not blocking.
- **The motivation figure for the supervisor — DONE:** `handoff/figures/control_frequency_safety.png`
  (3 panels): (a) cost-vs-AR for both κ with the d=25 line + per-seed scatter — crossing AR4→AR8 is
  the headline; (b) violation fraction per cell; (c) reward U-curve (sets up "why you'd want low
  frequency at all"). Regenerate via `handoff/figures/control_frequency_safety.py`.

---

## TASE TESTBED ADAPTATION — bringing the variable-dt method to PointGoal (decided 2026-06-29)

GOAL (this week): get the time-adaptive (TASE) agent RUNNING on the PointGoal testbed that the
fixed-AR sweep just validated, so the adaptive method is measured on the same axes (physical cost
vs d=25, reward, and now CONTROL EFFORT). The CT machinery already exists and is tested on
dm_control/cartpole; the work is wiring it to Safety-Gym + fixing the load-bearing accounting. Below
is the component-by-component audit (what exists vs what to change) and the ordered task list.

### What already EXISTS (CT machinery, built+tested on cartpole — reuse as-is)
- **Actor / action space:** `SwitchCostWrapper` augments the env action space with +1 dim
  (`pseudo_time ∈ [-1,1]`, wrappers.py:134-137). The actor's `action_dim` is read from the wrapped
  env, so it picks up the extra dim automatically. `evaluate_actor` (safe_actor_critic.py:226-240)
  extracts `pseudo_time`, maps it through `[t_min,t_max]`, computes `dt_ratio = time/base_dt`, and
  applies the per-step variable discount `γ**dt_ratio` with a straight-through round (STE) and
  `stop_gradient` so the actor can't game the discount by stretching dt. ✓ DONE, carries over.
- **World model / CNN time-channel strip:** with `agent.continuous_time.enabled=true`, the encoder
  drops the LAST obs channel (the `time_to_go` channel SwitchCostWrapper appends) before the CNN —
  `image_channels = image_shape[0] - 1` (world_model.py:139) and the strips at :176, :205, :306,
  :366. For PointGoal image obs (3,64,64) the wrapper makes it (4,64,64) and the model encodes the
  (3,64,64) RGB. ✓ Coded; needs a smoke test on the 4-channel image path (only exercised on
  1-D dm_control obs so far — see task T5).
- **base_dt / t_min / t_max plumbing:** trainer.py:94-106 extracts `base_dt` from the env's `dt`
  attribute at runtime and sets `t_min = min_time_factor·base_dt`, `t_max = max_time_factor·base_dt`.
  ✓ generic, works for any env that exposes `dt`/`control_timestep`.

### What must CHANGE / be ADDED for PointGoal
**T1 — Wire SwitchCostWrapper into the safe_adaptation_gym factory (PRIMARY GAP).**
  SwitchCostWrapper is ONLY constructed in `benchmark_suites/dm_control/__init__.py:298-322`. The
  PointGoal factory (`benchmark_suites/safe_adaptation_gym/__init__.py:make`) never adds it, so
  `continuous_time.enabled=true` on PointGoal today does NOTHING. Mirror the dm_control block: gate
  on `cfg.agent.continuous_time.enabled`, construct `SwitchCostWrapper(env, t_min, t_max,
  switch_cost=ConstantSwitchCost(...), discounting=cfg.agent.discount)`, and place it AFTER
  `ChannelFirst` (so it augments the (3,64,64) image, matching the world-model strip).
  RESOLVED 2026-06-29 (read the source): safe_adaptation_gym does NOT expose `dt`/`control_timestep`
  — one `env.step()` already runs `physics.step(nstep=frequency)` internally
  (safe_adaptation_gym.py:69-72), so one env-step IS the atomic control unit and the trainer's
  `get_attr("dt")` (trainer.py:95) would fail and fall back to 0.01. DON'T chase the mujoco physical
  timestep. Instead set **`base_dt = 1.0` explicitly in the CT config** and treat dt in
  ATOMIC-ENV-STEP units: `t_min = min_time_factor·1.0`, `t_max = max_time_factor·1.0`, so
  `num_repetitions ∈ [min_time_factor, max_time_factor]` is exactly a count of held env-steps —
  the direct continuous analogue of the AR ladder (this also makes dt_ratio = #base-steps and
  N = time_limit/base_dt = 1000, which is precisely what the T3 budget derivation assumes). Either
  pin `agent.continuous_time.base_dt=1.0` in the config (trainer falls back to it cleanly) or have
  the safe_adaptation_gym factory expose a `dt=1.0` attribute so the existing extraction path finds it.

**T2 — FIX the within-window cost discounting (LOAD-BEARING, do this before any TASE number).**
  SwitchCostWrapper currently accumulates `total_cost += discounting**current_step · cost`
  (wrappers.py:192). This (a) makes realized episode cost NOT the raw physical sum, so it is no
  longer comparable to d=25 across different dt choices, and (b) lets the actor HIDE cost by
  stretching dt — later base steps in a long hold are discounted away, so the safety-critic target
  under-counts exactly the blind-window cost we are trying to constrain → TASE would learn to make
  safety LOOK better by acting less often, the opposite of the contribution. Change cost to a RAW
  sum like ActionRepeat (wrappers.py:27): `total_cost += step_info.get('cost', 0.0)`. The REWARD
  within-window discounting (line 191) may stay (it's a modeling choice for the return); only COST
  must be raw. This is item (1) of the two must-checks promised on 2026-06-27.

**T3 — Make the safety-budget formula CT-aware (and understand WHY it gets simpler).**
  In CT mode the discount is per-BASE-step (`γ**dt_ratio`, dt_ratio = base steps in the window), so
  the fair discounted threshold is just the AR=1 / base-step formula and is FREQUENCY-INVARIANT BY
  CONSTRUCTION — no manual action_repeat correction needed (this is the elegant pay-off vs the
  fixed-AR sweep, which needed `÷action_repeat`). Derivation: for a uniform cost rate c per base
  step, V_c ≈ c·Σ_k dt_ratio_k·γ^(cumulative) ≈ c·(1−γ^N)/(−ln γ) with N=time_limit/base_dt; setting
  the physical budget c·N = 25 gives V_c ≈ 2.5 = the AR=1 threshold `25/time_limit/(1−γ)`. So:
  `make_actor_critic.py:34` must use `episode_steps = time_limit / 1` (NOT `/action_repeat`) WHEN
  `continuous_time.enabled` — add that guard. In practice the CT config should also pin
  `training.action_repeat=1` (episodic_async_env.py:208 skips ActionRepeat when SwitchCostWrapper is
  present anyway), so the existing formula already yields the right number; the guard just makes it
  robust to a stray action_repeat override. This is item (2)'s budget half — and it means the
  CONSTRAINT is identical to the fixed-AR1 cell, so TASE is compared on exactly the d=25 bar.

**T4 — Decide the COST-OF-CONTROL accounting (the motivation axis).** Two options:
  (a) keep `switch_cost` as a REWARD penalty (wrappers.py:202) → control effort folded into the
  objective; the actor trades reward vs #decisions, and we REPORT realized control frequency
  (mean dt, decisions/episode) as the efficiency axis. Simplest; recommended for v1.
  (b) make `switch_cost` a separate COST channel that counts toward the safety budget → cleaner
  "two-resource" framing but changes constraint semantics and needs critic re-plumbing.
  DECISION: ship (a) for the first TASE-on-PointGoal runs (no constraint-semantics risk), keep the
  efficiency purely as a reported axis, and revisit (b) only if the frontier plot needs control
  effort to be a hard-constrained resource. This is item (2) of the 2026-06-27 must-checks.

**T5 — New CT experiment config + smoke test.** Add `configs/experiment/safe_goal_tase.yaml`
  (mirror safe_goal_ar_study.yaml but: `agent.continuous_time.enabled=true`,
  `training.action_repeat=1`, `min_time_factor=1`, `max_time_factor=16` to span the SAME dt ladder
  the fixed sweep covered (AR1…AR16), a small `switch_cost`, task=go_to_goal, image obs). Then a
  short local smoke run to confirm: (i) the (4,64,64) image flows through the augmented obs space
  and the world-model strip without shape errors; (ii) `info['dt']` and `agent/.../entropy` log;
  (iii) the dt the actor picks actually varies (not pinned at t_min or t_max). Add a metric for
  per-episode mean dt / decision count and a dt-vs-distance-to-hazard log (the TASE payoff plot).

**T6 — Add an entropy bonus to the actor loss — IMPLEMENTED 2026-06-29 (overnight diagnostic queued).**
  Wired `actor_entropy_coef` end-to-end: `configs/agent/actsafe.yaml` (default **0.0** = byte-identical
  upstream) → `make_actor_critic.py` → `SafeModelBasedActorCritic.__init__/update` →
  `update_safe_actor_critic` → `evaluate_actor`, where `loss = -objective - coef*actor_entropy(actor,
  initial_states)` (guarded by `if coef>0` so the baseline graph is unchanged). It flows into the
  OBJECTIVE gradient via the penalizer (not the safety constraint), and since `actor_entropy` covers
  the full action vector it also regularizes the CT dt head. NOTE: entropy is regularized at
  `initial_states` (imagination start states) — a Dreamer-style proxy; if too weak, extend to all
  imagined trajectory states. Overnight DIAGNOSTIC (bracket the coef, don't bet on one value):
  `actor_entropy_coef=1e-3,1e-2 × action_repeat=1,8 × seed=0,1,2` at κ=0.1 (12 runs) — does the
  collapse heal (AR1 reward recovers toward ~15–20) and does the crossing survive (AR8 still ≈/over
  budget)? Pick the winning coef, then regenerate the clean κ=0.1 curve for the NEXT update.
  Launch (Euler):
  `python train_actsafe.py -m +experiment=safe_goal_ar_study +hardware=4090_rtx hydra/launcher=slurm`
  `+wandb.project=actsafe-ct-pointgoal agent.actor_entropy_coef=1e-3,1e-2 training.action_repeat=1,8`
  `training.seed=0,1,2`. (Original design rationale below.)

  ORIGINAL RATIONALE —
  Decided 2026-06-29 after the κ×AR grid review. The actor loss is pure `-objective` with NO entropy
  term (`safe_actor_critic.py:268-269`); `actor_entropy()` (`actor_critic.py:87`) is computed but
  used only for logging — this has been logging-only since the upstream `dcbe264` commit, so it is
  NOT a fork regression but an inherent ActSafe fragility. On some seeds the policy saturates
  (entropy → the −16.57 floor, reward → 0). In the fixed grid this collapse contaminates the REWARD
  panel (low-AR medians dragged down) and, at AR16, the reward collapses across ALL seeds — so AR16
  is a "both unsafe AND task-failing" point. NOTE (revised 2026-06-29 after user pushback): AR16 is
  NOT dropped — the κ=0.1 COST ladder is monotone through AR16 (…25.3→26.3, still 3/5 violating) and
  is a valid low-frequency-unsafe point; the collapse caveat lives only on the reward panel.
  **Why this is a TASE task, not a baseline re-run:**
  the motivation hypothesis stands WITHOUT it (the AR4→AR8 cost crossing is carried by healthy seeds;
  fixing collapse would, if anything, sharpen it — see the analysis below), so we do NOT re-run the
  baseline. But TASE reuses this same actor (plus the dt head), so collapse would land directly on
  the contribution. Fix = add `+ entropy_coef * actor_entropy(new_actor, states)` to the actor
  objective (small coef ~1e-3–1e-4, config-exposed, default 0 to stay byte-identical until opted in);
  verify on the fixed AR1/AR2 cells that the collapsed seeds recover before trusting TASE seed
  variance. Pairs with OPEN-VERIFICATION (c): the dt head shares `init_stddev=5.0` and may itself
  saturate, so the entropy term should cover the dt dimension too.

### WHY WE DO NOT NEED TO FIX COLLAPSE TO REPORT THE BASELINE (decided 2026-06-29)
The cost-violation hypothesis is INDEPENDENT of the collapse. The AR4→AR8 crossing is carried by
HEALTHY, goal-reaching seeds: at κ=0.1, AR4 = all 5 seeds healthy (obj 18–20), cost ≈ 21, safe;
AR8 = healthy seeds (obj 9–18) at cost 23.8/24.1/25.3/**30.7**. The seed driving the AR8 violation
(`obj=18.2, cost=30.7`) is a fully functional policy that violates because it cannot react during
the long open-loop hold — exactly the claimed mechanism. Fixing collapse would give MORE healthy
seeds, not fewer violations, so it cannot delete the result. Collapse only hurts the REWARD panel.
→ REPORT THE BASELINE AS-IS. Framing (revised after user pushback 2026-06-29): lead with κ=0.1 as
the headline; **AR8 is the cleanest single result — "competent but unsafe"** (healthy seeds that
still violate). **KEEP AR16** — the κ=0.1 cost ladder is monotone the whole way
(14.5→18.4→21.2→25.3→26.3) and AR16 still violates 3/5; its only caveat is that reward has collapsed
there, so it is an "unsafe AND task-failing" point rather than a "competent but unsafe" one. That
caveat is shown on the reward panel, NOT by deleting the cost point. (Detailed 3-panel figure:
`handoff/figures/control_frequency_safety.png` — kept as internal backup.)

### SUPERVISOR UPDATE — sent with the SIMPLE figure, reward not shown (decided 2026-06-29)
Per the user, the 3-panel figure is over-built for a check-in. The supervisor figure is the
SINGLE clean panel `handoff/figures/cost_vs_frequency.png` (κ=0.1 median cost + IQR vs AR, budget
line) — one message: lower frequency → higher cost, crosses the budget. **Reward is deliberately
NOT shown** (it would expose the entropy collapse, which looks bad). Instead the update carries a
one-line competence statement: "at AR4–8 the agent reaches paper-level reward (~18–20), so the cost
comes from a capable policy, not a do-nothing one" — this also pre-empts the "is AR1's low cost just
inaction?" question. Robustness across κ stated verbally, not plotted. Entropy fix (T6) stays
deferred to TASE prep. Update drafted at `handoff/supervisor_update_2026-06-29.md`.

### TASE IMPLEMENTED 2026-06-30 — wiring complete, smoke test pending

T1/T2/T3/T5 are DONE and statically validated (parse + budget math); T6 (entropy) was already
wired. The variable-dt method now builds end-to-end on PointGoal. Changes:
- **T1** — `benchmark_suites/safe_adaptation_gym/__init__.py`: wraps the env in `SwitchCostWrapper`
  (OUTERMOST, after `ChannelFirst`) when `agent.continuous_time.enabled`. The worker
  (`episodic_async_env.py:208`) already skips `ActionRepeat` when it detects `SwitchCostWrapper`.
- **T2** — `rl/wrappers.py`: cost is discounted within the hold by `safety_discount ** sub_step`
  (mirrors reward; revised 2026-06-30 after the user's chunk-invariance argument — a raw sum makes the
  per-step cost depend on how time is chopped into holds once dt is adaptive, breaking the SMDP). The
  critic/world-model see this CHUNK-INVARIANT discounted cost; `info['cost_realized']` carries the raw
  physical sum for the d=25 plot (NOT yet wired into `train/cost_return` — TODO before the publication
  figure; a 6-field Transition change, deferred to avoid a blind ripple. For TASE the gap is small:
  cost is incurred at small dt near hazards where discounted≈realized).
- **T3** — `make_actor_critic.py`: CT path uses the frequency-INDEPENDENT threshold
  `d / (time_limit*(1-safety_discount))` (= 2.5 for d=25), decoupled from action_repeat. Discrete
  AR-study path byte-identical (still 5.0 at AR2).
- **T5** — `configs/experiment/safe_goal_tase.yaml`: dense go_to_goal, image obs, `model.continuous_time=true`,
  min/max_time_factor=1/16, switch_cost=0.1 (swept), κ=0.1, actor_entropy_coef=0.01, opax, init_stddev default.
- **dt consistency**: the trainer's `base_dt = env.get_attr("dt")` resolves to `SwitchCostWrapper.self.dt`
  (same object in the stack), so `dt_ratio == num_repetitions` exactly — absolute dt cancels.
- **dt extraction FIXED 2026-06-30**: safe_adaptation_gym exposes NO Gym `dt`/`control_timestep`, so the
  old code silently fell back to 0.01. The REAL control dt = `robot.sim.model.opt.timestep (0.004) *
  _ROBOT_TO_CONTROL_FREQUENCY[robot]` → **point = 0.02 (50 Hz)**, car 0.04, doggo 0.048. New
  `_control_dt()` helper in the factory reads this and passes it EXPLICITLY to `SwitchCostWrapper(dt=...)`
  (new param). dm_control was already correct (cartpole control_timestep = 0.01, NOT 0.02 — verified
  against the live sim). Absolute dt cancels in the discounting math, but it's now physically correct for
  reporting control frequency in Hz (the paper's axis).
- **dt-head saturation diagnostic ALREADY EXISTS** (`epoch_summary.continuous_time_metrics`, logged at
  trainer.py:152): `train/ct/{mean_dt_ratio,std_dt_ratio,frac_dt_1,frac_dt_max}`. frac_dt_1/frac_dt_max≈1
  = saturated; std≈0 = collapsed. Watch these in the smoke test.
- **dt-head init scale (opt-in) ADDED**: `ContinuousActor.dt_init_stddev` (default None = byte-identical;
  the last/dt action dim can get its own initial exploration). NOT set for PointGoal (init_stddev=5.0
  already explores); uncomment `actor.dt_init_stddev` in safe_goal_tase.yaml only if the diagnostic shows
  the dt head pinned. This is the likely fix for the past cartpole CT failures (init_stddev=0.025).

SMOKE TEST (run this FIRST, ~20 min, catches integration crashes I can't hit locally — no jax here):
```
python train_actsafe.py +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal training.epochs=2 training.seed=0
```
Smoke-test checks (the OPEN VERIFICATION items): (b) the (4,64,64)→(3,64,64) image strip runs; (c) the
dt head is NOT saturated to one end — log/inspect `info['dt']` histogram (entropy coef should help; if
still pinned, give the dt head its own init scale); (d) `info['steps']`/`info['dt']` flow through
acting.py step-counting so variable-length episodes book-keep correctly.

THEN the overnight sweep (the key diagnostic = is the dt histogram non-degenerate?):
```
python train_actsafe.py -m +experiment=safe_goal_tase +hardware=4090_rtx hydra/launcher=slurm \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.switch_cost=0.02,0.1,0.5 training.seed=0,1,2
```

### TASE TASK ORDER (this week) — DONE; superseded by the block above
1. **T2** (raw-cost fix) — one-line correctness fix, unblocks every CT number. Do first.
2. **T1** (wire wrapper into PointGoal) + **T3** (CT budget guard) — makes `enabled=true` actually
   build the variable-dt env on the validated testbed with the right d=25 bar.
3. **T5** (config + smoke test) — prove the image/world-model path runs end-to-end; verify dt varies.
4. **T4** is a config choice (switch_cost in reward) already satisfied by ship-(a); no code beyond
   setting the value.
5. **T6** (entropy bonus) — do AFTER the image/world-model path runs clean (T5) but BEFORE the first
   real TASE run, so the contribution isn't measured on a collapse-prone actor. Quick: one line in
   the loss + a config coef; sanity-check on the fixed AR1/AR2 cells that collapsed seeds recover.
6. THEN launch a first TASE run at the d=25 bar and compare against the fixed-AR grid on the same
   axes: does the adaptive agent hold cost ≤ AR4-level WHILE reaching AR4-level reward at lower
   average control frequency than AR1? That comparison is the first evidence the method beats every
   fixed frequency.

OPEN VERIFICATION before trusting TASE safety numbers: (a) safe_adaptation_gym does NOT expose a
physical `dt` — RESOLVED, use `base_dt=1.0` atomic-env-step units (see T1); (b) confirm the
world-model 4-channel strip on the (4,64,64) image path (only the 1-D dm_control path is exercised
today — T5); (c) confirm the actor's dt head isn't saturating to one end (it shares init_stddev=5.0
with the motor actions — may need its own scale); (d) confirm `info['steps']`/`info['dt']` from
SwitchCostWrapper flow through acting.py:57 step-counting on the PointGoal path the same way they do
for dm_control (the episode-length bookkeeping for variable-dt episodes).

---

## ROOT CAUSE OF reward ≈ 0: the "Frankenstein" config (FIXED)

`actsafe/configs/experiment/safe_goal_ar_study.yaml` had imported hyperparameters from the
SPARSE/scarce Safety-Gym configs and applied them to the DENSE `go_to_goal` task at AR=1:
- `actor.init_stddev = 0.025` (sparse value) → near-deterministic actor → starves action
  exploration on the dense task → policy never finds reward. **This is the primary reward killer.**
- `exploration_steps = 850000`, `model_initialization_scale = 0.05` — also sparse-cell values.
- `action_repeat = 1` instead of the paper's dense default `2`.

The safety constraint was NOT the cause: the ActSafe paper reaches Ĵr ≈ 15–20 on dense
`go_to_goal` at the SAME budget d=25 (Fig 4). `constraint_pessimism=0.001` is negligible
(≈ the null/bayes regime), so pessimism was not the cause either.

**Fix applied (this session):** rewrote `safe_goal_ar_study.yaml` to match upstream's dense
`safety_gym.yaml` — removed `init_stddev=0.025` (→ agent default 5.0), set
`exploration_steps=500000`, `constraint_pessimism=0.001`, `action_repeat=2` (the swept anchor).
Early signal after the fix: epoch-0 per-env rewards spiked to +8.17/+5.72/+4.37 vs the old
flat ~0 — config fix appears to be working. **Still to confirm: the exploitation phase climbs
toward ~15–20.**

### Verified the reward path is upstream-faithful (no discrete-algo bug introduced)
- `MultiRewardBridge` (multi_reward.py) is unchanged from upstream; selects task reward index 0.
- All `world_model.py` `time_to_go` stripping is gated on `self.continuous_time` → inert for discrete.
- `safe_actor_critic.py` discrete path (uniform discount array) is numerically equivalent to
  upstream's scalar path.
- Phase logic in `actsafe.py:125-219` (offline → OPAX exploration → exploitation) is intact;
  the task actor-critic IS trained during exploration.

---

## CLUSTER CRASH: wandb/NFS shutdown fragility, NOT a training bug — and the decision

Symptom: the run showed as "crashed" in wandb; the `.err` log showed
`wandb: Network error (TransientError)` then a repeating `OSError: [Errno 116] Stale file
handle` in wandb's internal thread during `poll_exit`. Diagnosis: compute nodes have flaky
access to wandb.ai; during a network stall wandb's debug log (on NFS home) gets a stale NFS
handle, and the internal thread then loops forever trying to flush to the dead handle, hanging
the job at exit. **Not a training error, not gym-related.**

**DECISION (user, 2026-06-24): do NOT add launcher workarounds. `slurm.yaml` stays minimal
and is reverted to its committed form (no `WANDB_MODE=offline`, no scratch-dir env vars).**
Rationale: the run worked fine the day before; this was most likely an outlier network blip.
Offline mode "messes things up" (breaks live monitoring and adds a manual `wandb sync` step).
If the stale-handle hang recurs and is reproducible, revisit then — but do not pre-emptively
complicate the launcher for a one-off. The async wandb writer in `rl/logging.py` already drops
metrics on a full queue (deadlock-safe), so transient blips shouldn't be fatal.

## RENDER FLAG CRASHES RUNS (user-confirmed 2026-06-25)
Enabling the render flag caused PointGoal runs to FAIL outright; removing it let them complete.
Do NOT add render to any sweep command. (Likely MuJoCo EGL/offscreen headless rendering on the
compute node — the committed `slurm.yaml` already sets EGL but the render path is still fragile.)
The AR sweep commands below intentionally omit render.

## gym → gymnasium migration: DEFERRED (do not do now)
The deprecation warnings (`apply_api_compatibility`, "Gym unmaintained", NumPy-2.0 note) are
cosmetic and unrelated to the crash. The NumPy-2.0 warning only bites if numpy is upgraded,
which is pinned. Migrating is a high-risk refactor of `safe_adaptation_gym` + the API-compat
shim — i.e. the exact env wiring that just started producing correct reward. File under "later,
only if numpy is bumped or upstream forces it."

---

## NEXT ACTIONS (in order)

1. **Restart AR=1 seed=0 with the fixed config**, confirm `train/objective` climbs toward
   ~15–20 in the exploitation phase and the stale-handle hang does not recur. Reproduce the
   paper's exact dense cell first as a sanity check:
   ```bash
   python train_actsafe.py +experiment=safety_gym +hardware=4090_rtx hydra/launcher=slurm \
     +wandb.project=actsafe-ct-pointgoal training.seed=0
   ```
2. **GO/NO-GO — the fixed-AR sweep** (the money plot's prerequisite), anchored at AR=2:
   ```bash
   python train_actsafe.py -m +experiment=safe_goal_ar_study +hardware=4090_rtx hydra/launcher=slurm \
     +wandb.project=actsafe-ct-pointgoal training.action_repeat=1,2,4,8 training.seed=0,1,2
   ```
   Hypothesis: `train/cost_return` rises with AR and crosses budget=25 → GO. Flat at all AR
   (incl. 8) → thesis unsupported → honest re-scope with the supervisor. Plot:
   `docs/plot_ar_safety_study.py` (needs wandb auth). Spec: `docs/ar_safety_study.md`.
3. **CT correctness before any CT result** — fix the load-bearing cost-accounting leak, then
   run the budget-invariance test (below), then the CT money plot (dt/freq vs distance-to-hazard).
4. Ablations: `switch_cost=0` (headline: adaptation is safety-driven), exploration/deployment
   pessimism decoupling.

### LOAD-BEARING CT BUG (fix before trusting any CT safety number)
`wrappers.py:191-192` (`SwitchCostWrapper.step`):
`total_cost += (self.discounting ** current_step) * cost`. Safety cost is discounted WITHIN the
macro-hold window, so the same physical trajectory reports LESS cost when held longer → the
agent can game the budget by acting less frequently — the OPPOSITE of the thesis. `self.discounting`
is also independent of `agent.safety_discount=0.99`. Fix: accumulate safety cost as a RAW
undiscounted sum within the window (reward may stay discounted). Then run **budget invariance**:
a fixed policy forced to several dt values must report (near-)identical realized episode cost;
if cost changes with dt, CT can cheat the budget. This is the single most important CT check.

### RESOLVED (2026-06-25): budget × action_repeat scaling is a FIX, not a bug
`make_actor_critic.py` budget scaling: fork uses
`episode_safety_budget = (safety_budget/episode_steps)/(1-safety_discount)` with
`episode_steps = time_limit/action_repeat`, giving 5.0 at AR=2 vs upstream's naive 2.5.

**Verdict after full trace:** the scaling is the faithful generalization of upstream's OWN budget
translation to action_repeat≠1, and it is the CORRECT choice for the AR sweep. Derivation:
- Upstream's formula is already a translation, not a raw budget. The safety critic constrains a
  DISCOUNTED cost-to-go `J_c = E[Σ γ^k C_k]`. Upstream infers the per-step cost rate `ρ=d/time_limit`
  consistent with undiscounted episode budget d=25, then expresses it as `ρ/(1-γ)`.
- Changing action_repeat R changes the MDP: per-agent-step cost grows ~R× (ActionRepeat SUMS raw
  cost, wrappers.py:27) AND there are R× fewer agent steps (`N=time_limit/R`), with γ discounting
  per agent step. Re-deriving: `c=d/N`, `J_c≈c/(1-γ)` ⟹ threshold `=(d/(time_limit/R))/(1-γ)` =
  the fork's formula exactly. Inert at AR=1 (N=time_limit ≡ upstream).
- This holds the PHYSICAL episode budget (25 raw sim-step costs) invariant across AR — which the AR
  sweep REQUIRES. Upstream-naive (2.5) would instead tighten physical safety ~2× as AR grows,
  confounding the experiment (can't separate physics-driven violations from a tightened budget).
- VALIDITY DEPENDS ON RAW-SUM cost aggregation. Discrete AR sweep routes through `ActionRepeat`
  (raw sum) — PointGoal is NOT a SwitchCostWrapper env (episodic_async_env.py:201-209) — so the
  discrete path is SOUND. The CT path's `SwitchCostWrapper` DISCOUNTS within the window
  (wrappers.py:192) → breaks this translation → must fix the within-window discounting (the
  load-bearing CT bug above) before the budget translation is valid on the CT path.
- TODO (low priority): add an `assert episode_steps == time_limit` regression guard at AR=1 and a
  comment in make_actor_critic.py pointing at the raw-sum dependency.

### METRICS SEMANTICS (added 2026-06-25)
`epoch_summary.py:117` `_objective` = `rewards.sum(2).mean()` — RAW UNDISCOUNTED episode sum.
- `train/cost_return` = undiscounted per-episode cost sum → compare DIRECTLY against budget=25.
  The 5.0-at-AR=2 discounted threshold is INTERNAL to the safety critic, never logged.
- `train/objective` = undiscounted episode reward (PointGoal dense target ~15–20).
- Exploration-phase caveat: during OPAX exploration the deployed policy maximizes info-gain (σ),
  so `train/objective` (reward) is NOT meaningful. BUT `train/cost_return` IS still meaningful —
  ActSafe does SAFE exploration (the explorer is itself cost-UCB-constrained, exploration.py:48-50),
  so cost_return measures whether exploration stayed within budget. Do not dismiss it.

### WORLD MODEL vs OTACOS (audited 2026-06-25)
- dt fed as last ACTION dim → RSSM transition conditioned on hold duration; WM learns macro-dynamics
  from data. ✔ matches OTACOS variable-duration prediction.
- `time_to_go` STRIPPED before encoder + from reconstruction target (world_model.py:176,205) → never
  enters the latent.
- DIVERGENCE from OTACOS (deliberate, must be stated in the paper, not claimed as equivalence):
  OTACOS uses augmented state `(x, integrated-reward b, time-to-go t)` because it is FINITE-HORIZON
  made stationary by carrying t. This fork is INFINITE-HORIZON DISCOUNTED and carries neither t nor b
  in the latent (reward accumulated in wrapper, predicted per macro-step). Consistent for a stationary
  discounted policy, but the policy/value CANNOT be time-dependent (no different behavior near episode
  end). Defensible for the safety-frequency story; state it explicitly.
- RECOMMENDATION (2026-06-25): do NOT build the full OTACOS augmented-state `(x,b,t)` model. OTACOS
  needs it for a finite-horizon GP regret proof; we are infinite-horizon discounted with an RSSM
  ensemble that already supplies the calibrated epistemic uncertainty pessimism/optimism need, and
  dt-as-action already conditions the transition on hold duration (the one property that matters).
  Only optional tweak: stop stripping time_to_go from the encoder IF a time-dependent policy proves
  necessary near episode end. The real CT priority is the within-window cost-discounting bug, not the
  world-model architecture.

### Penalizer revert (offered, not executed)
Reverting "the penalizer" to upstream to minimize points of failure means reverting **6
entangled files** (lbsgd.py, augmented_lagrangian.py, dummy_penalizer.py, safe_actor_critic.py
step_scale plumbing, common/learner.py grad_step `scale=`, make_actor_critic.py). The discrete
path is already verified numerically equivalent to upstream, so this is low-urgency; only do it
if a safety-side bug is suspected.

### Success criteria
- `train/objective` clearly positive (PointGoal dense target ~15–20, paper-comparable).
- `train/cost_return` under episode budget; `agent/lbsgd/safe` ≈ 1.
- AR sweep produces a monotone (or U-shaped) cost-vs-frequency curve — the GO signal.

---

## EXPERIMENTS & PLATFORMS STRATEGY (added 2026-06-25)

### Provisional name for the approach
**TASE — Time-Adaptive Safe Exploration** (working name). Deeper one-liner for the paper's framing:
*safety-driven control-frequency adaptation* — the agent slows its decision rate (longer holds) where
the world model is confident and safe, and speeds up (shorter holds) near hazards where epistemic
uncertainty and cost risk spike. Alternatives floated: SAFR (Safety-Adaptive FRequency), CASE. Pick
later; TASE is the placeholder.

### Core hypothesis (the thing every experiment must isolate)
Control frequency is a *causal* factor in execution-time safety: at a FIXED physical safety budget,
LOWER control frequency (longer open-loop holds) → more violations, because the agent commits to an
action across a window where the state (and hazard proximity) changes faster than it can react. The
money plot is a U-shape / monotone rise of realized cost vs action-repeat (or vs dt in CT).

### Platforms — recommendation
PRIMARY: **stay in Safety-Gym**, ladder of difficulty, because (a) it is ActSafe's home task so the
WM/encoder/wrappers are already correct, (b) all the safe-RL baselines (CPO, LAMBDA, BSRP-Lag) report
numbers there, (c) hazards give a clean, dense cost signal, (d) momentum/non-holonomy is the physical
*mechanism* for the U-shape.
  Run BOTH PointGoal and CarGoal (not either/or — PointGoal already works, Car is the stronger test):
  1. **PointGoal** (current) — PRIMARY; runs already produce paper-level reward. Holonomic, gentle;
     may show only a weak U-shape, but it's the validated baseline env.
  2. **CarGoal / CarButton** — non-holonomic, real momentum & drift → a long hold overshoots into
     hazards. BEST candidate for a clean frequency-breaks-safety signal. Becomes the headline if
     PointGoal's U-shape is weak. Run alongside PointGoal, not instead of it.
  3. **DoggoGoal** — high-dim, unstable gait; stress test / generalization datapoint, not the lead.
SECONDARY (generalization, one datapoint, NOT the main result): **DM-Control + custom constraint**,
e.g. Walker/Hopper with a torso-height or impact-velocity safety cost. Higher implementation cost
(DM-Control has no native cost channel — must author the cost fn). Use only to show the effect isn't
Safety-Gym-specific.
DROPPED: cartpole (Appendix A) — cost stays under budget at every AR, U-shape physically impossible.

### Experiment matrix
Let X-axis be control frequency (discrete: action_repeat ∈ {1,2,4,8}; CT: swept dt / t_max). Y-axis:
realized undiscounted `cost_return` (vs budget=25) and `train/objective`.

A. **TASE vs unsafe exploration** (isolates: does SAFE adaptation matter?) — TASE vs uniform-random
   dt and vs greedy/entropy exploration with no safety constraint. Expect: unsafe baselines violate
   hard at low freq; TASE adapts freq to stay near budget.
B. **TASE vs fixed-discretization safe RL** (THE core contribution plot) — TASE (adaptive dt) vs
   ActSafe run at each FIXED action_repeat. Expect: every fixed-AR ActSafe sits on the U-shape
   (some AR too slow → violations, some too fast → wasteful/over-conservative); TASE traces the
   lower envelope by choosing dt per-state. THIS is the "adaptation buys you something" result.
C. **TASE vs external safe-RL baselines** — CPO, LAMBDA (Lagrangian MBRL), BSRP-Lag. Run at the
   standard fixed frequency to position absolute safety/return numbers against the literature.
D. **Ablations** — `switch_cost=0` (headline: adaptation is SAFETY-driven, not interaction-cost-driven);
   exploration- vs deployment-pessimism decoupling (Appendix A carry-over); fixed-dt vs adaptive-dt
   at matched mean frequency (isolates adaptivity from raw frequency).

### Novelty framing (ICLR 2027) — guard against "just a code merge"
Do NOT pitch as TACOS+ActSafe glued together. The contribution is a NOVEL PROBLEM SETTING + a finding:
- Setting: *safe exploration under agent-chosen control frequency* — the agent decides not just WHAT
  to do but HOW LONG to commit, under a hard safety budget, with epistemic uncertainty.
- Finding (the load-bearing empirical claim): control frequency is a first-class safety lever, and
  adapting it per-state — fast near hazards / under uncertainty, slow when confident — Pareto-dominates
  any fixed frequency on the safety-return frontier. Concept: modulating dt is the temporal dual of
  regulating the epistemic safety bound near hazards.
- This is publishable IF plot B shows TASE beating the fixed-AR envelope. That plot is the kill gate.

### Gate order (do not skip)
1. Discrete AR sweep on PointGoal (GO/NO-GO #2 above) → is there a U-shape at all?
2. If weak, repeat on CarGoal before touching CT.
3. Fix the CT within-window cost-discounting bug + budget-invariance test.
4. Only then: TASE (adaptive dt) and plot B.

---

## Appendix A — Cartpole investigation (SUPERSEDED 2026-06-23, kept for its findings)

Everything below predates the PointGoal pivot. The cartpole task (`safe_swingup_sparse_hard`)
was abandoned because cost stayed under budget at every action_repeat, so the safety-frequency
U-shape physically cannot appear there. Retained because several findings (data-fix semantics,
exploration-starvation diagnosis, exploration/deployment pessimism decoupling idea) carry over.

### ROOT CAUSE (2026-06-18): `interact()` stored ~10× too little data — FIXED
The CT rewrite of `acting.py::interact()` changed what `episodes_per_epoch` meant. Upstream:
one trajectory holds ALL parallel envs, `episode_count += 1` per synchronized `done.all()`
(one rollout batch), `observe()` gets `[num_envs, T]` → 5 batches × 10 envs = 50 episodes/epoch.
Fork (broken): per-env trajectories, `episode_count += 1` per env, breaks at `num_episodes` →
observes 5 of 10 envs, discards the other 5 → 5 episodes/epoch, while `step` still counted all
10 envs. Net: world model saw ~10× less data per epoch. **Fixed 2026-06-18:** `interact()` now
counts batches (upstream semantics) and observes all envs; CT staggered-`done` preserved via
`active_mask`. Verified the discrete path is mathematically identical to upstream.

### Cartpole sweep findings (kept for carry-over insight)
- std=5+pess=1 reliably reaches obj 535/553 at cost 7–8, but pess=1 WEAKENS the safety guarantee.
- The faithful paper cell std=0.025+pess=50 barely reproduces (1/12 seeds reached obj≈488);
  the rest stagnate at obj≈0 with LOW cost = **exploration starvation** (agent does nothing,
  trivially safe), from two stacked causes: (a) init_stddev=0.025 → near-deterministic actor;
  (b) pess=50 → the OPAX explorer's constraint (exploration.py:48-50 reuses constraint_pessimism)
  makes cost-UCB = mean + 50·unc exceed the seen budget → explorer repelled from novel-but-safe
  region → world model never populates it → reward critic never sees reward.
- **Principled fix (carries to PointGoal): decouple exploration-phase pessimism from deployment
  pessimism.** Add `agent.sentiment.exploration_pessimism` (default = constraint_pessimism),
  consume at exploration.py:48-50, sweep low (1–5) while the deployed safe-actor-critic stays
  pessimistic. Keeps the paper's deploy-time guarantee while letting OPAX explore.
- `opax.py` round→floor fixed 2026-06-22 to match `SwitchCostWrapper.compute_time` floor; CT-only.

### CT gradient design (verified correct, carries over)
dt-as-action-dim (WM learns macro dynamics from data); `stop_gradient` on the discount kills the
analytic discount-hack so dt only gets gradient through WM dynamics (STE on safe_actor_critic
line ~236 now redundant but harmless); OPAX `/dt_ratio` with stop_grad is correct per-unit-time
normalization. The real CT risk is the cost-accounting leak (wrappers.py:191-192), not gradients.
