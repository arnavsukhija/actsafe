# ActSafe-CT Implementation Plan (current as of 2026-06-24)

This is the single source of truth for "where we left off." Start here on any new device
or chat. The historical cartpole investigation is preserved verbatim in **Appendix A** at the
bottom — it is superseded by the 2026-06-23 PointGoal pivot but its findings are still cited.

Companion docs in this `handoff/` folder (mirror of the auto-memory): `MEMORY.md` (index),
`project_strategy_2026-06-23.md` (the decisions), `project_paper_direction.md`,
`project_bugs_fixed.md`, `project_ct_architecture.md`, `project_cluster_infra.md`,
`user_profile.md`, `feedback_style.md`.

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
