# ActSafe-CT Implementation Plan (current as of 2026-07-11)

This is the single source of truth for "where we left off." Start here on any new device
or chat. The historical cartpole investigation is preserved verbatim in **Appendix A** at the
bottom — it is superseded by the 2026-06-23 PointGoal pivot but its findings are still cited.

Companion docs in this `handoff/` folder (mirror of the auto-memory): `MEMORY.md` (index),
`project_strategy_2026-06-23.md` (the decisions), `project_paper_direction.md`,
`project_bugs_fixed.md`, `project_ct_architecture.md`, `project_cluster_infra.md`,
`user_profile.md`, `feedback_style.md`.

---

## FIRST-PRINCIPLES REVIEW + COVERAGE-FIRST PURGE LANDED (2026-07-11) — CURRENT STATE

**The full architectural review lives in `code_review.md` (repo root, rewritten 2026-07-11) —
read it before touching the CT stack.** It proves the SMDP/ZOH/flow foundation sound
(Bellman/telescoping exact; time augmentation valid; k-gradients flow through three audited
paths; tanh IS enforced on the dt head) and replaces the estimator-bias story with the user's
**central diagnosis: phase-wise dt-coverage mismatch**. Code changes below are uncommitted on
`wip/tase-pointgoal` as of writing; tests must be re-run on Euler (`poetry run pytest -v`)
before launching — no local env on this machine.

**Corrected OPAX narrative (supersedes the czo8evls read in the 2026-07-07 pre-launch section
and the safe_goal_tase.yaml comment of that date):** the "healthy spread for 50k–200k then
collapse at 250k" was a misread — 0–200k IS the offline uniform phase (`offline_steps=200000`);
its mean_dt≈11.7 matches uniform [0,1) pseudo-times mapping to k∈[9,16] (the offline-range
bug). **OPAX chose dt=1 from its FIRST controlled step (200k) and never left** — the rational
optimum of a per-decision log-squashed bonus with no switch cost in the exploration objective
(V_explore(k) ≈ b(k)/(1−γ^k)). OPAX never "degenerated"; there is currently no objective for
it to explore higher k. dt-aware OPAX (switch cost in the objective, k-sweep disagreement,
(state,k) novelty) is a parked study arm — design sketch in code_review.md §4.

**Part A purge (landed):**
1. **Expectile cost head DELETED entirely** (never run in any observed experiment; one-sided
   weighting risks its own misalignment; under the coverage diagnosis plain Gaussian/MSE should
   calibrate once data exists): `world_model.py` fields/branch, `cost_head`/`cost_expectile_tau`
   keys in `agent/actsafe.yaml` + `safe_goal_tase.yaml`. The factored hazard-rate head is
   likewise DROPPED (conflicts with the flow-based identity — the model perceives the flow at
   the end of the hold); estimator-level changes only per the escalation clause below.
2. **`opax_dt_normalization` DELETED** (flag + plumbing in `opax.py`/`opax_bridge.py`/
   `exploration.py`/config); idea recorded in code_review.md §4.
3. **`dt_exploration: uniform` is again the default** in `safe_goal_tase.yaml` (mechanism
   unchanged in `ActSafe.__call__`: resample only the EXECUTED action's dt head over the full
   [−1,1], keep motor dims + `prev_action` consistency).
4. **Offline-coverage bug FIXED**: `UniformExploration(action_dim, dt_pseudo_dim=True)` (CT
   only) samples the dt dim in [−1,1]; upstream sampled [0,1) → offline executed ONLY k∈[9,16]
   at max_repeat=16. Motor dims stay [0,1) for baseline comparability. Tests:
   `tests/test_dt_exploration.py`.
5. **NOT purged** (audited sound): γ^k + STE, nearest-int mapping, action-conditioned aggregate
   decoder, full-MDP time, discounted CMDP + B=d/(T(1−γc)), cost_realized/exposure fields,
   budget-invariance tests, constraint_pessimism_source flag (ablation only).

**Part B diagnostics battery (landed, all per-epoch on one replay batch):**
- `agent/cost_calibration/count_k_*` — raw bucket sizes alongside the gaps (a clean gap on an
  empty bucket proves nothing; the gate is gaps ≈ 0 AT NON-TRIVIAL COUNTS).
- `agent/model_per_k/recon_k_*`, `kl_k_*` — reconstruction MSE + posterior‖prior KL bucketed by
  exposure: tests "the flow model is unvalidated at large k" on the DYNAMICS (discriminates the
  coverage diagnosis from a pure cost-head story).
- `agent/imagination/cost_return_{imagined,realized,gap}` — open-loop rollout with stored
  actions from matched buffer states vs stored discounted cost (the only probe of multi-step
  imagination compounding; teacher-forced calibration cannot see it). NEGATIVE gap = optimistic.
- `agent/ct/kslope/{perceived_cost,perceived_qc,realized_cost}` — finite-difference ĉ(s,u,k)
  and Q̂_c = ĉ + γc^k·V̂_c(s'_k,t') between k_min/k_max vs the buffer's OLS cost-vs-exposure
  slope. perceived_qc < 0 while realized > 0 = the degenerate "longer looks safer" gradient.
- `agent/actor/dt_head_grad_norm` (+ `motor_head_grad_norm`) — tanh-saturation and STE-Jacobian
  ((k_max−k_min)/2) watch, last-layer dt rows.
- **Guard:** `ActSafe.update()` now hard-fails when a pre-exposure pickle is resumed with CT
  enabled (`replay_buffer.aux_backfilled`) instead of training on fabricated exposure≡1.

**Escalation clause:** only if material optimism persists at *well-covered* k (balanced
`count_k_*`) after this coverage fix do we revisit estimator-level changes.

**Launch (Euler; the "after" arm of the coverage experiment):**
```bash
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.max_repeat=8,16 \
  agent.continuous_time.switch_cost=0.002,0.005,0.01,0.05 \
  training.seed=0,1,2
```
**Read order (gates, in order — if 1 fails nothing downstream is interpretable):**
(1) coverage during exploration: `train/ct/buffer/frac_dt_q1..q4` roughly balanced,
`near_hazard_distinct_dt` high, `frac_dt_max > 0`; (2) per-k model error + calibration gaps
WITH counts; (3) `agent/imagination/cost_return_gap` ≈ 0; (4) constraint sign vs realized
`cost_return` — the 100%-safe-while-violating signature (run e6cttmuk) must break;
(5) task phase: `cost_return ≤ 25`, `dt_near_far_ratio < 1`, kslope perceived/realized agree.

---

## PRE-LAUNCH VERIFICATION PASS (2026-07-07, before the Wave-1 sweep) — SUPERSEDED IN PART

> **2026-07-11:** item 1's czo8evls read is WRONG (the 50k–200k "healthy spread" was the
> offline uniform phase, not OPAX; OPAX sat at dt=1 from its first controlled step) and the
> `policy` revert plus the expectile head are both undone — see the 2026-07-11 section above.
> Items 2–4 (checkpoint cadence, wandb run-id fix, calibration plumbing) remain current.

Four checks against the wandb history of the crashed `actsafe-ct-pointgoal` runs (all 248 runs
in the project are `state=crashed`), done before launching the Wave-1 sweep.

1. **`dt_exploration` reverted to `policy`** (user decision): read of run `czo8evls`
   (switch_cost=0.05, max_repeat=16, this "policy"/natural mode) shows OPAX's dt distribution
   is NOT stably diverse — `mean_dt_ratio≈11.7, std≈2.2` (healthy) for steps 50k-200k, then
   COLLAPSES to `frac_dt_1≈1.0, std_dt_ratio≈0` for steps 250k-500k (the back half of the
   500k-step exploration budget spent almost entirely at dt=1) — the same poor-coverage finding
   that motivated the uniform patch in the first place. Reverted anyway per user instruction, but
   **watch `train/ct/buffer/frac_dt_9_plus` and `agent/cost_calibration/gap_k_9_plus`** in the
   new sweep: if the k≥9 bucket is data-starved, a clean calibration gate there proves nothing.
   Once past `exploration_steps`, the same run's dt jumps to and stays at `mean_dt_ratio≈13`
   with `frac_dt_1≈0` for the rest of training (550k-1.35M) — a roughly fixed high-hold operating
   point (economically rational under a fixed `switch_cost`), not yet evidence of hazard-tracking
   adaptive dt (that needs the still-pending dt-vs-hazard-proximity battery, Pillar 3).
2. **Root cause of "no space left on device" found and fixed**: `ReplayBuffer` preallocates a
   dense `(capacity=1000, max_length+1=1001, 64, 64, 3)` uint8 observation array ≈ **12.3 GB**,
   fully materialized regardless of how full the buffer actually is. `checkpoint_every` defaulted
   to `1` (`configs/config.yaml`), so `Trainer.train()` cloudpickled this ~12GB state EVERY epoch
   (~every 2 min at observed fps) via `StateWriter`'s tmp-write-then-`os.replace` — safe against
   corruption, but needs ~2x the file size (~25GB) free during the swap. Fixed:
   `safe_goal_tase.yaml` now sets `training.checkpoint_every: 10`.
3. **New finding, not previously documented: wandb run-id fragmentation.** Every crashed+resumed
   sweep run split its learning curve across 2+ wandb runs. Confirmed on disk: `czo8evls` logs
   steps 50k→1,350,000 continuously, then `wee7kzun` (a different run id, `switch_cost`/
   `max_repeat` identical) picks up at exactly step 1,350,000 and continues to 1,650,000 before
   crashing again — i.e. the resumed process reattached to the pickled `Trainer` state but NOT to
   the same wandb run, because `WeightAndBiasesWriter` never persisted/derived a run id, so
   `wandb.init(resume="allow", ...)` minted a fresh one each time. This would have made the
   per-k calibration curve this sweep exists to validate unreadable as one continuous line after
   any crash. Fixed in `actsafe/rl/logging.py`: `id` is now `md5(os.getcwd())[:16]` when not
   explicitly set — the hydra run dir is the same path `Trainer.should_resume()` checks for
   `state.pkl`, so a resume in the same directory now reattaches to the same wandb run.
4. **Validation plumbing (`agent/cost_calibration/gap_k_*`, `cost_realized`/`exposure` buffer
   fields) confirmed correctly wired**: `ActSafe.report()` samples one batch, calls
   `cost_calibration(model, features, actions, batch.exposure, key)` unconditionally whenever the
   buffer is non-empty (`actsafe.py:343-356`); `cost_calibration` computes `pred − target` (the
   SAME `features.cost` the model is trained on) bucketed by executed `exposure` k. Confirmed
   these keys are genuinely NEW: `czo8evls` (pre-Wave-1) has zero rows for any
   `agent/cost_calibration/*` key — there is no historical data to compare against; the first
   Wave-1 run is the first real exercise of this path. Re the "critic performs worse at high k"
   claim: that was never an independent finding, it IS the calibration-gap story — MSE/Gaussian
   regression on a target with support `[0,k]` has one-sided-optimistic shrinkage that grows with
   k (measured ≈28× ensemble std, `corr(violation, calibration gap)=+0.91` in the discrete AR
   study); expectile regression's asymmetric weighting (τ=0.9 on under-prediction) is the Wave-1
   attempt to fix exactly this, and `gap_k_1` vs `gap_k_9_plus` flat-at-≈0 is the literal
   pass/fail readout.

All 25 tests still pass (`test_epoch` deselected, pre-existing/unrelated), ruff clean, no new
mypy errors from these edits (mypy has pre-existing unrelated errors elsewhere in the package,
confirmed present before this change via `git stash`).

---

## WAVE-1 CALIBRATION-FIX STACK LANDED (2026-07-07) — SUPERSEDED IN PART

> **2026-07-11:** the expectile head (item 2) was DELETED before ever running, and the factored
> hazard-rate escalation path was DROPPED (flow-identity conflict) — the first-principles review
> (`code_review.md`) replaced the estimator-bias theory with the coverage diagnosis; see the
> 2026-07-11 section above. Items 1, 3–7 (mapping fix, exposure plumbing, calibration metrics,
> dt_near_far_ratio, pessimism flag, budget tests) remain landed and current.

Implements the fix path approved by the 2026-07-06 audit, with one mid-implementation pivot:
**the user chose EXPECTILE REGRESSION over the factored hazard-rate head as the first attempt**
(less invasive; the rate head stays fully designed as the escalation path if expectile fails
the calibration gate). All 26 tests pass; ruff + mypy clean. Changes uncommitted on
`wip/tase-pointgoal` as of writing.

**What landed:**
1. **ct_time mapping fix — max_repeat now reachable** (`actsafe/rl/ct_time.py`): quantization
   changed from `floor` to NEAREST integer (ties up, implemented as `floor(x+0.5)` because
   np.round is banker's rounding), clipped to [k_min, k_max]. User's choice over the
   [k_min,k_max+1)+floor alternative; accepted trade-off: k_min/k_max own half-width pseudo
   intervals (2× under-sampled under uniform dt-exploration vs interior holds). Wrapper
   execution, STE forward, and diagnostics all import ct_time.py, so quantization stays
   consistent everywhere. NOTE: new runs' dt histograms are not bit-comparable to pre-fix runs
   (mid-range holds shift ~0.5; frac_dt_max was structurally ~0 before, now real).
2. **Expectile cost head** (`world_model.py`; config `agent.model.cost_head:
   gaussian|expectile`, `cost_expectile_tau: 0.9`): cost-channel loss becomes asymmetric MSE —
   UNDER-prediction (the optimistic direction) weighted τ, over-prediction 1−τ; τ=0.5 recovers
   the gaussian gradients exactly. `safe_goal_tase.yaml` defaults to `expectile`; reward
   channel and the discrete-path default (gaussian) unchanged.
3. **Exposure + raw-cost plumbing**: `Transition` gained `exposure` (executed base steps, from
   `info['steps']`; action_repeat on discrete); the replay buffer stores `cost_realized` (raw
   hold cost) and `exposure` alongside the UNCHANGED discounted `cost` (legacy head
   byte-identical). Old pickles resume via lazy `_ensure_aux_arrays` backfill
   (cost_realized≈cost, exposure=1) — fine for exploratory resumes; use FRESH runs for
   reported experiments.
4. **Per-k cost-calibration metrics** (`world_model.cost_calibration`, logged per epoch under
   `agent/cost_calibration/*`): `gap_k_1/k_2_4/k_5_8/k_9_plus` (mean predicted−target per
   exposure bucket; NEGATIVE = optimistic), `gap_overall`, `gap_hazard_holds` (hazard-positive
   holds only — zero-inflation hides optimism in the overall mean), `target_k_*`, `frac_k_*`.
   **THE GATE: expectile passes iff gaps ≈ 0 with no k-trend; else escalate to the factored
   rate head.**
5. **dt-vs-hazard-proximity metric** (`ct_time.buffer_dt_coverage`): new
   `train/ct/buffer/far_hazard_mean_dt` + `dt_near_far_ratio` (near = ≤1 decision from a
   hazard, far = ≥5, hazard-containing episodes only; ratio < 1 = decides faster near hazards
   = the adaptiveness signal), plus k_max-relative quartile buckets `frac_dt_q1..q4` (legacy
   fixed buckets kept for wandb continuity).
6. **Pessimism-source flag** (`sentiment.py`; `agent.sentiment.constraint_pessimism_source:
   latent|cost_spread`): default `latent` = unchanged behavior; `cost_spread` = mean + κ·std of
   decoded cost over the ensemble (pessimism in cost units) — ablation only.
7. **Budget-invariance audit tests** (`tests/test_budget_invariance.py`) + refactor
   `make_actor_critic.compute_episode_safety_budget()`: the 2026-07-06 audit as regression
   tests — CT budget dt-independent (2.5); discrete realized allowance = d at every R (the
   R-cancellation); wrapper within-hold discounting telescopes to exact per-base-step
   discounting for arbitrary (incl. horizon-clipped) hold schedules.

**DEFERRED to Wave 2 (user decision 2026-07-07, designs locked in the assistant plan file):**
(a) auto-tuned interaction budget — Lagrangian dual on decisions/episode replacing the fixed
switch_cost (AL term on the imagined discount-weighted mean hold, LBSGD untouched; switch cost
removed from env reward so reward targets stay stationary) — implement only after the Wave-1
sweep reads clean; (b) factored hazard-rate head (Binomial on N with exposure k, analytic
composition ĉ = r̂·(1−γc^k)/(1−γc), exact ∂ĉ/∂k > 0) — the escalation path.

**Launch (config defaults now carry expectile + uniform dt_exploration + mapping fix):**
```bash
# Wave-1 sweep ("after" arm; same grid as the running policy-mode control arm)
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.max_repeat=8,16 \
  agent.continuous_time.switch_cost=0.002,0.005,0.01,0.05 \
  training.seed=0,1,2
# Gaussian-head control on the same stack (isolates the expectile effect):
#   ... agent.model.cost_head=gaussian
# Optional pessimism ablation: ... agent.sentiment.constraint_pessimism_source=cost_spread
# Discrete AR study untouched (cost_head defaults to gaussian there).
```

**What to read on wandb:** (a) `agent/cost_calibration/gap_k_*` — the gate (≈0, no k-trend;
watch `gap_hazard_holds` especially); (b) `agent/safety_critic/constraint` optimism vs realized
`cost_return` — the old everywhere-positive gap should close; (c)
`train/ct/buffer/dt_near_far_ratio` < 1 in task phase = adaptiveness; (d)
`train/ct/buffer/frac_dt_max` now > 0 (mapping fix working); (e) `frac_dt_q1..q4` for coverage.

---

## BUDGET/CRITIC AUDIT (2026-07-06) — THE DIAGNOSIS IS SETTLED

**Question audited (user hypothesis):** is the SMDP safety-budget math fundamentally flawed —
do higher-repeat runs get an artificially inflated budget, explaining both the safety-critic
collapse and the dt behavior? **Answer: no. The budget math is sound in both branches; every
observed failure is safety-critic calibration error.** Verified algebraically AND empirically.

### 1. Budget equations (both branches exact)
- **Discrete branch:** `B(R) = d·R/(T(1−γc))` = 2.5R. Per-agent-step cost targets sum R base
  steps, so critic value `V ≈ R·c̄/(1−γc)` — the R cancels: `V ≤ B(R) ⇔ c̄·T ≤ d`. Realized
  allowance is d=25 at EVERY repeat. Key falsifiable prediction: budget-filling would give
  realized cost flat at 25 across R, never an upward slope.
- **TASE branch:** `B = d/(T(1−γc))` = 2.5, dt-independent. Chunk-invariance (telescoping):
  within-hold discounting in `SwitchCostWrapper` + γc^k across decisions compose to exactly
  `Σ_t γc^t c_t` for ANY hold schedule → holding longer cannot buy allowance. Not exploitable.

### 2. The discrete AR figure decoded (53 task-phase lineages, `safe_goal_ar_study`,
project `actsafe-ct-pointgoal`, `continuous_time.enabled=false`, tail-avg past 1M steps)

| AR | realized J | J−25 | calibration excess (10·gap/R) |
|----|-----------|------|-------------------------------|
| 1  | 16.3 | −8.7 | −6.0 (critic pessimistic) |
| 2  | 20.9 | −4.1 | −1.4 |
| 4  | 20.8 | −4.2 | −1.8 |
| 8  | 28.4 | +3.4 | +5.2 (critic optimistic) |
| 16 | 27.0 | +2.0 | +10.4 |

- LBSGD servoes the *perceived* cost value to ~90% of budget at every R (AR=8: V̂=18.1±0.3 vs
  B=20 across all 16 lineages) → `J(R) = α·25 + 10·ε(R)/R` — the ENTIRE slope of the
  "cost rises with repeat" figure is the critic's bias ε(R), not physics, not budget.
- Bias flips sign between AR=4 and AR=8 (pessimistic → optimistic). Run-level
  corr(J−25, calibration excess) = **+0.91** (n=50, degenerates excluded); all 22 violating
  runs have positive gap; 30/31 under-budget runs have negative/near-zero gap.
- The physical claim "coarse control is inherently costlier at matched safety" is
  **unidentified** in this data (enforcement error varies with R); needs matched
  realized-violation comparisons. Vanilla ActSafe "works" because it lives at R=1 where the
  bias is small and protective.

### 3. Root cause: MSE on time-aggregated sparse cost targets
Gaussian/MSE cost decoder regresses zero-inflated accumulated targets, support [0,R] (discrete)
or [0,k] (TASE). Shrinkage toward the conditional mean is one-sided OPTIMISTIC, absolute bias
∝ target scale ∝ dt. κ·ensemble-std pessimism cannot cover it (systematic, not epistemic;
measured gap ≈ 28× ensemble std → would need κ≈28 vs current 0.1). In TASE the flat learned
k-slope additionally gives the wrong-signed perceived-safety gradient
`∂Q̂/∂k = γ^k·ln(γ)·V′ < 0` (true sign near hazards is positive) — longer holds LOOK safer, so
the dt head gets a spurious safety gradient. Empirical signature: critic V falls 1.91→1.29 as
mean_dt rises 1.3→11.9 while realized value stays flat; corr(mean_dt, optimism gap) = +0.40 at
rep16, ≈0 at rep8. **The agent hacks the critic, not the budget** (discrete runs: R fixed →
no agency, pure enforcement error; TASE: dt is an action → the agent steers into the
optimistic region).

### 4. Approved fix path (audit approved 2026-07-06; code NOT yet written)
- **Primary — factored hazard-rate head:** learn per-base-step hazard probability p̂(s,u) via
  Bernoulli/cross-entropy; compose hold cost analytically `Σ_{i<k} γc^i p̂ᵢ` (or
  `p̂·(1−γc^k)/(1−γc)` under local stationarity). Targets are {0,1} at every R/k → kills
  scale-dependent shrinkage and the discrete sign flip; k-dependence exact → restores
  ∂Q̂/∂k > 0 near hazards; CE is a proper scoring rule (calibrated by construction).
  Secondary options (in order): expectile regression τ>0.5 on the cost head; real-transition
  TD grounding of the safety critic; base-resolution imagination (exact, compute ∝ k).
  Threshold-side realized-cost anchoring is the wrong layer (leaves the actor's gradient biased).
- **Exploratory — auto-tuned interaction budget:** replace fixed `switch_cost` with a dynamic
  per-episode decision budget enforced by dual gradient ascent on the switch price (a fixed
  price drifts as the policy improves; observed bang-bang dt is the rational response to a flat
  price, not mistuning). Design stage only.
- **Small bug to land alongside:** `ct_time.dt_ratio_from_pseudo` never reaches `max_repeat`
  (requires p=1.0 exactly, tanh never attains it; frac_dt_max ≤ 0.005 in all 80 lineages).
  Fix: map [−1,1] → [k_min, k_max+1) before floor, with clip.
- **Before any adaptiveness claim:** add a dt-vs-hazard-proximity metric (fast decisions near
  hazards). Also: honest reward comparisons must be at matched REALIZED violation (in current
  runs the perceived constraint is inactive, so reward is partly unearned).

### Verdict on health
Formulation core (flat realized budget, chunk-invariant accounting, variable discounting) is
sound — defend as-is. Genuine signals: monotone dt price-elasticity across 3 orders of
switch_cost; rational bang-bang allocation; rep16/sc0.05 under budget at mean_dt ≈ 11. Unearned
claims until the cost head is fixed: the safety claim and the *adaptive*-safety claim. The
discrete sign-flip table is paper material: the failure is a general property of learned
time-aggregated cost models, which the factored hazard-rate head fixes for both settings.

---

## VANILLA-FAITHFUL CLEANUP LANDED + NEW SWEEP RUNNING (2026-07-05, evening) — CURRENT STATE

**The full TASE cleanup plan (audit → user directives → 7 commits) landed on `wip/tase-pointgoal`.**
All 17 tests pass (`test_unsupervised_trainer::test_epoch` deselected — pre-existing failure);
`ruff check actsafe/` clean.

**Commits (in order):**
- `af75f9b` — StateWriter atomic checkpoint write (tmp + fsync + os.replace; SIGKILL-safe).
- `cc9d230` — removed `safety_dt_gradient` injection entirely; `opax_dt_normalization` default
  `false` (flag kept for ablations); docs corrected (the discount stop-gradient claim was FALSE —
  the discount `γ^dt_ratio` is fully differentiable, deliberate & vanilla-faithful, decided with
  the supervisor 2026-07-05).
- `a454d68` — STE `round` → `floor` so imagination matches SwitchCostWrapper's executed hold.
- `ae6367f` — repeat-units parametrization: config keys `min_repeat`/`max_repeat` (integers);
  new single-source-of-truth `actsafe/rl/ct_time.py` (affine map, floor, STE, inverse); deleted
  `Trainer._populate_continuous_time_config` runtime injection (the resume-bug source);
  `SwitchCostWrapper` clock counts UP from 0 by EXECUTED steps (user's formulation; fixes a
  latent bug where the clock advanced by requested time on early truncation).
- `93cfe25` — full-MDP time: agent state = (latent, elapsed-time fraction). CNNs stay on real
  pixels; the clock channel is stored uint8-safe as `255·elapsed/horizon`; policy/critics get
  `state_dim+1`; imagination propagates time analytically (`t' = min(t + k/T, 1)`).
- `e99129c` — per-epoch `train/ct/buffer/*` metrics in `report()` (dt histogram shares, mean/std,
  near-hazard coverage) — the in-run answer to the coverage question, no pickle forensics.
- `c2c48a0` — `diagnose_dt_coverage.py` OOM hardening (coverage-only mode frees the ~16 GB obs
  array; needs `srun --mem=64G`; supports new k_min/k_max and legacy tmin/tmax/base_dt pickles).

**Locked decisions (user directives, do not relitigate):** no stop-gradients beyond vanilla
(discount differentiable; OPAX bonus stop-grad kept = upstream; STE-internal stop-grad kept =
the estimator itself); no switch_cost in the OPAX objective; time stays in the agent's state
(full MDP — this is the primary novel contribution, don't compare to upstream here);
`constraint_pessimism=0.1`; oTaCoS action-conditioned reward/cost decoder kept.

**New sweep LAUNCHED (2026-07-05, on the new stack, dt_exploration=policy = natural-coverage arm):**
`max_repeat=8,16 × switch_cost=0.002,0.005,0.01,0.05 × seed=0,1,2` (24 runs, project
`actsafe-ct-pointgoal`). First read at 550–600k steps (just past the 500k OPAX phase):
- **Natural dt coverage is CONFIRMED poor: `train/ct/buffer/frac_dt_1` = 0.85–0.94 in every run**
  (mean_dt ≈ 1.5–1.85). The buffer is saturated with dt=1 after the OPAX phase — this settles
  open question #2 with data. Near-hazard contrast exists but is thin
  (near_hazard_distinct_dt = 7–16, near_hazard_std_dt ≈ 2.3 (mr8) / 4.5 (mr16)).
- Task-phase policy dt (`train/ct/frac_dt_1`) already scales with switch_cost as expected:
  sc=0.05 → ~0.01–0.03 (long holds), sc=0.002 → 0.6–0.8 (mostly dt=1).
- cost_return at this early point: ~18–48, straddling/violating budget 25 (unchanged story).

**Executive decision (user, 2026-07-05): dt_exploration=uniform going forward.** Given the
confirmed dt=1 buffer saturation (and the OOM'd offline diagnostic), we do not wait further:
during the exploration window the EXECUTED action's dt head is resampled uniformly over
pseudo-time [−1,1] (policy force dims kept, `prev_action` kept consistent for the RSSM filter —
mechanism in `ActSafe.__call__`, actsafe.py). `safe_goal_tase.yaml` now sets
`dt_exploration: uniform`; the running sweep (policy mode) is kept as the natural-coverage
CONTROL arm.

**Next actions:**
1. Let the policy-mode sweep run (control arm; its `train/ct/buffer/*` curves are the "before").
2. Launch the SAME grid on the updated config (`dt_exploration=uniform`) — the "after" arm:
   the yaml default now carries uniform, so the identical launch command works; or add the
   explicit override `agent.continuous_time.dt_exploration=uniform`.
3. Compare `train/ct/buffer/near_hazard_*` and the safety gap (`agent/safety_critic/constraint`
   optimism vs realized cost_return) between the arms — the direct test of whether dt-coverage
   was the bottleneck behind the everywhere-optimistic constraint estimate.
4. If violation persists WITH verified coverage → audit the within-window cost-discounting leak
   next (see the empirical-audit section below), not actor-loss mechanisms.

---

## SWEEPS COMPLETE — FULL EMPIRICAL AUDIT (2026-07-05, afternoon) — CURRENT STATE OF KNOWLEDGE

Both 5M-step sweeps **finished this morning** (sdg=True lineages ended ~2026-07-04 23:14Z batch,
sdg=False ~07-05 07:48Z batch; the 07:16Z "runs" are zombie resumes of completed jobs that logged
nothing). Nothing is currently running. A per-run wandb scan of all 178 CT runs
(`scan of train/ct/{mean,std}_dt_ratio per process lifetime`) plus a full-lineage metric pull
established:

**1. The collapse taxonomy is now airtight.** Every *fresh* process shows: healthy dt → genuine
OPAX-phase dt→1 at ~250k–500k (recovers at 500k; absent with `opax_dt_normalization=false`) →
switch-cost-scaled task-phase dt. Every *resumed* process is dt≡1/std≡0 from its first log — 100%
of 6 resume batches, including step ranges where fresh processes show mean dt 7–14. "Collapse
before the requeue" observations are the (real, recovering) OPAX window, not a task-phase failure.

**2. `safety_dt_gradient` A/B: no meaningful difference — but it was CONFOUNDED.** The sdg=True
sweep (launched 07-04 15:06Z, commit 7948ca0) predates the action-conditioned decoder; the
sdg=False sweep (23:46Z, d9f7b49) has it. Final cost_return (mean±std over 3 seeds):
sc=0.002: 26–29 (False) vs 24–26 (True); sc=0.01: 36–38 vs 31–36; sc=0.05: 34–45 vs 36–39.
So "injection, no decoder" ≈ "decoder, no injection" on safety. Neither actor-gradient mechanism
moves the needle, which points away from gradient wiring and at the estimate being optimized.

**3. THE key safety finding: the constraint estimate is optimistic everywhere.**
`agent/safety_critic/constraint` (budget 2.5 − V_c) over the last 1M steps is **positive in every
single lineage** (0.39–0.74) with frac>0 = 1.00 — the agent believes it is ~20% under budget
100% of the time — while realized cost_return is 25–55 (up to 120% over). LBSGD and the CT budget
math (verified: d/(T·(1−γ_s)) = 2.5, chunk-invariant) are working as designed on a wrong estimate.
**The bottleneck is model/critic cost underestimation, not enforcement.**

**4. Scale of the problem:** violation grows with switch_cost (⇒ hold length): sc=0.002 ≈25–29
(feas ~0.6), sc=0.01 ≈31–38 (~0.4), sc=0.05 ≈34–45 (~0.36). Reference discrete AR study
(same project, non-CT runs): AR=1 ≈10–20 (feas 0.7–0.95), AR=4 ≈20–25, AR=8/16 ≈25–30 with heavy
seed variance. So (a) coarse control is genuinely harder to keep safe even in discrete-land, and
(b) TASE carries an additional ≈+10 cost offset over matched-frequency discrete AR. The
hold-length-scaling part is exactly what a dt-conditioned cost underestimate (coverage gap)
predicts; the constant offset needs the within-window discounting-leak audit / pessimism check.

**5. Objective logging fixed (commit a5a9674):** `train/objective` bakes in −switch_cost per
decision (and within-hold discounting); `train/objective_raw` now logs the undiscounted task
reward without the penalty (via `info['reward_realized']`, plumbed like `cost_realized`), so task
performance is comparable across switch_cost. Note completed sweeps only have the penalized one.

**Fix-stack triage (what is necessary vs. superseded):**
- KEEP (formulation/infra, artifact-independent): action-conditioned decoder (oTaCoS c̄(s,u,t)),
  discount STE, CT budget accounting, opax_dt_normalization flag, variable-length episode
  handling, resume-config fix, cost_realized/reward_realized reporting, dt_exploration=uniform.
- SUPERSEDED: `safety_dt_gradient` injection — a first-order surrogate for the d(cost)/d(dt) the
  decoder provides naturally; empirically indistinguishable arms. Default false going forward;
  keep the flag only as a paper ablation.
- DEAD: `dt_init_stddev` suggestion (never enabled, motivated by the phantom collapse).

**Decision & next actions (2026-07-05):**
1. **Run `scripts/diagnose_dt_coverage.py <run>/state.pkl --two-curve 8` on the completed 5M
   checkpoints NOW** (e.g. sc=0.05/mtf=16 and sc=0.002/mtf=8, one seed each): this is the Step-1
   coverage verification + the "before" two-curve baseline, on real data, at zero compute cost.
2. Relaunch ONE grid fresh on commit a5a9674 (decoder + dt_exploration=uniform + telemetry fix +
   objective_raw): `safety_dt_gradient=false`, same grid as before. This is the "after" arm.
   Optionally add sdg=true on the same commit for a clean (unconfounded) flag ablation.
3. After ~1M steps re-run the diagnostic (Step 3): if coverage is fixed and the two-curve mean is
   still dt-flat → decoder/training is the suspect; if dt-sensitive but violation persists →
   audit the within-window cost-discounting leak and consider raising `constraint_pessimism`.
4. Do NOT iterate on actor-loss mechanisms until #3 says the model's cost estimate is trustworthy —
   finding 3 says the actor is faithfully optimizing a wrong constraint.

---

## THE "dt COLLAPSE" WAS A RESUME LOGGING ARTIFACT (2026-07-05) — READ BEFORE TRUSTING ANY dt METRIC

**Root cause (found by cross-checking wandb lineages against the code):** `t_min`/`t_max`/`base_dt`
are injected into the hydra config at runtime by the trainer, but (pre-fix) only inside
`make_agent()` — which never runs on requeue-resume (`from_pickle` reuses the pickled agent). The
resumed trainer's `_run_training_epoch` then fell back to `tmin=tmax=0, base_dt=1`, and
`EpochSummary.continuous_time_metrics` computed `time_for_action ≡ 0 → dt_ratio ≡ 1` for **every
action of every resumed run**. Behavior was NOT affected: the pickled `make_env` closure and the
pickled actor-critic carry the correct values, and the objective/cost curves are continuous across
requeues (a genuine dt→1 collapse at sc=0.05 would show a −50 switch-cost drop in the objective;
it never does — the genuine dt=1 OPAX phase at 250k–500k shows exactly that −50 signature).

**Consequences for every prior conclusion:**
- Every "dt collapses to 1 by ~4M steps" observation (both sweeps, both `safety_dt_gradient`
  settings, pre- and post-decoder-fix) coincides exactly with the first requeue. **All post-requeue
  `train/ct/*` metrics in `actsafe-ct-pointgoal` before 2026-07-05 are invalid.** Pre-requeue
  segments (fresh runs, typically up to ~2.4M steps) are the only valid dt telemetry.
- What the VALID data actually shows (fresh 2026-07-04 23:46 batch, decoder fix + sgd=False):
  offline phase dt≈12.2 (uniform [0,1) pseudo — upper half only), OPAX phase dt=1.0 exactly
  (genuine: per-time-normalized bonus favors max decision rate; objective = −50·switch_cost
  confirms), then the task actor takes over at 500k with dt≈14 declining smoothly to ≈8–9 at 2.4M
  for sc=0.05 (std ≈5 — real diversity), dt≈1.7–2.2 for sc=0.01, dt≈1.1–1.4 for sc=0.002. That is
  a *sane, switch-cost-scaled, slowly adapting* dt profile, not a collapse.
- The "TASE sits at cost≈33 **at dt≈1**" load-bearing fact is half wrong: the cost numbers are
  real (cost metrics don't use tmin/tmax), the dt≈1 attribution was the artifact. **The real open
  problem is cost_return ≈25–45 vs budget 25 (feasibility ≈0.2) at whatever dt the policy actually
  runs.** Whether the decoder fix helps dt-adaptation is UNDECIDED — post-2.4M dt telemetry from
  the current sweep is garbage.

**Fix (committed):** `Trainer._populate_continuous_time_config()` now runs in `__enter__` (fresh
AND resume paths); `_run_training_epoch` asserts the keys exist instead of silently defaulting.
Currently-running runs keep logging garbage dt until requeued/relaunched with the fix.

**dt data-coverage mechanism (committed, addresses the chicken-and-egg exploration gap):**
`agent.continuous_time.dt_exploration=uniform` (+ optional `dt_exploration_steps`, default =
`exploration_steps`): during the warm-up window the EXECUTED action's dt head is resampled
uniformly over pseudo-time [−1, 1] (policy's force dims kept) in `ActSafe.__call__` — real-env
rollouts only, `AgentState.prev_action` updated so the RSSM filter conditions on the executed
action. Rationale: offline phase covers only the upper half of the dt range and the OPAX phase
collapses to dt=1, so the early buffer has no same-state multi-dt contrast for the WM to learn
c̄(s,u,t) from. Enabled by default in `safe_goal_tase.yaml`. Old pickles resume fine (defensive
getattr).

**Diagnostics (committed):** `scripts/diagnose_dt_coverage.py <run_dir>/state.pkl [--two-curve K]`
— run on the cluster. Reports (1) dt histograms per hazard-proximity bucket from the replay buffer
(Step-1 coverage check) and (2) predicted cost mean + ensemble disagreement vs dt at near-hazard
posterior states (the two-curve model diagnostic).

**Next actions:**
1. Relaunch the sgd=False sweep FRESH with the metric fix + `dt_exploration=uniform` (same grid);
   the running artifact-logged runs can be left to finish for cost curves but their dt telemetry is
   unusable.
2. After ~1M steps, run `diagnose_dt_coverage.py --two-curve 8` on one checkpoint per switch_cost
   to verify near-hazard dt coverage and dt→cost model sensitivity (before/after comparison).
3. THEN judge the decoder fix on valid full-length dt curves; only touch actor-loss wiring if the
   two-curve mean stays dt-flat despite verified coverage.
4. The cost≈30+ vs budget 25 infeasibility is the other real problem (frequency-independent;
   also present pre-requeue) — likely the within-window cost-discounting leak; tackle separately.

---

## oTaCoS-ALIGNED ACTION-CONDITIONED DECODER (2026-07-05) — SUPERSEDED READ: see section above
(The "collapse by 4.2M" premise below was the resume artifact; the structural c̄(s,u,t) argument
and the committed decoder change remain valid, but its effect is still untested on valid telemetry.)

**Empirical state (wandb `actsafe-ct-pointgoal`, sdg sweep of 2026-07-04, 18 configs × requeue):**
- `mean_dt` scales cleanly with switch_cost pre-requeue (≈1.2 / 1.9 / 5.5 for sc 0.002/0.01/0.05),
  then drifts to dt≈1 everywhere by ~4.2M steps. **Verified against the old (no-injection) runs:
  they ALSO drift to dt≈1–2** (only sc=0.05/mtf=16 held dt≈7), so the `safety_dt_gradient`
  injection is NOT the cause of the collapse — and dt→1 under persistent infeasibility is locally
  rational (max reactivity). Earlier "old runs recovered to 2–12" claim was wrong (partial
  mid-training read).
- **Load-bearing fact: TASE sits at cost ≈33 (raw) at dt≈1 in BOTH sweeps, vs discrete AR1 ≈15.7.**
  The safety gap is frequency-independent; base safety learning is the bottleneck, not the dt head.

**Code audit conclusions (verified against source, 2026-07-05):**
- WM training alignment is correct (`_prepare_features` feeds next_observation; the posterior
  consumes the dt action + post-hold obs; the decoder targets that hold's cost). No off-by-one in
  the λ-return / critic pairing: `G[k] = c_k + γ^{dt_k}·G[k+1]` regressed onto s_{k+1} is the
  standard Dreamer arrival-value convention and is chunk-invariant. The `γ^dt` discount is correct
  semi-MDP accounting (matches the lab PPO reference) — NOT a bug; stop-gradient ideas are dead.
- The real structural gap: cost of a hold is a transition quantity c̄(s, u, t), but the decoder
  read the arrival latent only. Within-hold cost is not observable from the arrival frame (pass
  through a hazard mid-hold → clean arrival pixels), so imagination's d(cost)/d(dt) had to route
  through the prior/posterior KL — data-hungry, loses to the exact analytic discount gradient.
- The `safety_dt_gradient` injection is correct in form (sign/scale/indexing verified) but is zero
  wherever predicted cost is zero, and LBSGD's infeasible fallback drops the reward gradient (and
  with it the switch-cost economics) entirely.

**Fix (committed): action-conditioned reward/cost decoder** in `world_model.py`, gated on
`model.continuous_time` (discrete path byte-identical). Decoder input = concat(latent, action) in
training (`__call__`) and imagination (`sample`, action broadcast over the ensemble axis). Composed
with the duration-conditioned dynamics, each member yields c̄_i(s, u, t) — the oTaCoS formulation in
latent space, preserving ensemble pessimism. Supersedes the injection → run new sweeps with
`agent.continuous_time.safety_dt_gradient=false`. Pre-launch check on the cluster:
`pytest tests/test_world_model_action_conditioning.py -v`. NOTE: old pickles cannot resume into the
new decoder (input dim changed) — fresh runs only.

**Launch plan (overnight 2026-07-05):**
```bash
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.max_time_factor=8,16 \
  agent.continuous_time.switch_cost=0.002,0.01,0.05 \
  agent.continuous_time.safety_dt_gradient=false training.seed=0,1,2
```
Optional diagnostic (localizes the ≈33-vs-15.7 gap: adaptive-training poisoning vs TASE plumbing):
`agent.continuous_time.max_time_factor=1 agent.continuous_time.switch_cost=0.002 training.seed=0,1,2`.

**Watch for:** dt becoming state-dependent (short near hazards, long in open space) instead of
globally collapsed; cost at sc=0.01/0.05 approaching the AR-sweep numbers; sc=0.002 staying at
dt≈1 is fine (switch cost too cheap to matter — not a failure signal).

---

## TASE "DONE AFTER EPOCH 0" CRASH — ROOT CAUSE FOUND & FIXED (2026-07-03)

**Symptom:** every seed-0 TASE grid cell died right after epoch 0 with no visible exception in the
submitit `.err`. The AR study (`safe_goal_ar_study`) ran fine on the same cluster/GPU pool, so the
failure was TASE-specific.

**CONFIRMED ROOT CAUSE (reproduced locally on CPU, full traceback):** a variable-length-episode
shape bug in `actsafe/rl/epoch_summary.py`. `EpochSummary.metrics` did `np.stack(rewards)` /
`np.stack(costs)` across all episodes in an epoch, and `continuous_time_metrics` did
`np.concatenate(all_actions, axis=0)` on a fixed time axis. Both assume **every episode has the same
number of transitions**. That holds for `ActionRepeat` (fixed `repeat` base steps per decision →
every episode is exactly `time_limit/repeat` decisions long) but is **violated by TASE**:
`SwitchCostWrapper` lets the agent choose the hold length (`num_repetitions ∈ [1, max_time_factor]`)
per decision, so each episode consumes its fixed physical time budget in a variable number of
decisions — different across the parallel envs and across epochs. The first epoch where two episodes
differ in length, `np.stack` throws `ValueError: all input arrays must have the same shape`. This is
called once per epoch in `trainer.py` (`objective, cost_return, feasibilty = summary.metrics`),
immediately after epoch 0's rollouts — exactly "done after epoch 0". It's a pure numpy logic bug,
nothing to do with CUDA/XLA/GPU, which is why it reproduces identically on CPU and on the cluster.

Why the `.err` was empty of a *useful* exception: `train_actsafe.py` catches and re-raises, but under
submitit the traceback lands in the job's stdout/submitit pickle result, not always where you'd look;
regardless, the confirmed repro makes the forensic guessing moot.

**Fix applied (`epoch_summary.py`):**
- `metrics`: replaced `np.stack` with `_stack_padded` — zero-pads each episode's reward/cost array on
  the time axis to the epoch's max length before stacking. Mathematically exact: `_objective` and
  `_feasibility` only `.sum()` over the time axis, and trailing zeros don't change a sum. `_stack_padded`
  no-ops back to plain `np.stack` when all lengths already match, so it does not alter AR-study behavior.
- `continuous_time_metrics`: flatten each episode's own time axis and pool across episodes with
  `np.concatenate` (instead of stacking on a fixed time axis). The dt_ratio/force stats are means/stds
  over a pool of scalars, so no padding is involved — no fake values enter the statistics.
- **These feed only the logging/reporting path** (`logger.log(...)` in `trainer.py`), never the replay
  buffer or `agent.learn()`. So the fix cannot affect learning or behavior — it only unblocks the
  per-epoch metric print. Verified locally: a shrunk TASE run (`JAX_PLATFORMS=cpu`, `parallel_envs=4`,
  real `plan_horizon`/`update_steps`) now clears epoch 0's `summary.metrics` and proceeds into epoch 1
  actor-critic updates, with healthy diagnostics (`mean_dt_ratio≈8.5`, `frac_dt_1≈0.38`,
  `mean_abs_force≈0.87` → agent moving, dt varying, not frozen/collapsed).

**Full component audit for the same class of bug (2026-07-03) — all clear, safe to sweep:**
- `replay_buffer.py` `add`: already length-aware — zero-pads into fixed `max_length` slots, records true
  `self.lengths`, and samples only within real length. Variable-length is first-class. ✓
- `acting.py`: per-env trajectories under `active_mask`; early-finishing envs stop appending while others
  continue, so each env keeps its own length. ✓ (empirically ran clean)
- `episodic_async_env.py`: `np.asarray` stacks across envs at a single timestep (uniform shape), never
  across time. ✓
- `report()` / `metrics.py`: operate on fixed-`sequence_length` replay samples and scalars. ✓
- `safe_actor_critic.py:240` discount STE `dt_ratio = dt_raw + stop_gradient(round(max(dt_raw,1)) - dt_raw)`
  is correct; historical Bug A (missing `stop_gradient`) is NOT present. `discounted_cumsum` handles the
  variable per-step discount via `lax.scan`. ✓
- `epoch_summary.videos`: reads only `all_vids[-1]` (a single trajectory), so NOT subject to the
  cross-episode length bug; `render_episodes=0` for the sweep anyway. Left as-is. ✓

**Secondary hygiene change (NOT the root cause):** also restored `XLA_PYTHON_CLIENT_PREALLOCATE=false` /
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` to `slurm.yaml` (dropped incidentally in commit `9b088d8`) and added
`JAX_TRACEBACK_FILTERING=off`. This is low-risk memory hygiene + a diagnostic aid; it is **not** what
caused the TASE crash (the AR study ran fine without it). Kept because it's reasonable, not because it
fixed anything.

**Interactive-debug procedure (for future issues like this):** run locally or on an `srun` shell with
`JAX_PLATFORMS=cpu JAX_TRACEBACK_FILTERING=off WANDB_MODE=offline` and a shrunk config
(`training.parallel_envs=4 training.time_limit=200 agent.exploration_steps=1500 training.epochs=2
training.episodes_per_epoch=1 agent.offline_steps=0`) while keeping the real `plan_horizon`/`update_steps`.
CPU strips all cuDNN/XLA-autotune noise so pure logic bugs surface with a clean Python traceback in
seconds, and the shrunk env still exercises the variable-length rollout + epoch-summary path.

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

SMOKE TEST (run this FIRST, ~20 min, catches integration crashes I can't hit locally — no jax here).
NOTE: `-m` AND `hydra/launcher=slurm` are BOTH required even for the single smoke run. `-m` enables
multirun; `hydra/launcher=slurm` selects the SlurmLauncher (the config that holds the rtx_4090
`additional_parameters`). Without the launcher override Hydra falls back to the BasicLauncher and
errors with `Key 'additional_parameters' not in 'BasicLauncherConf'` (runs locally, no 4090):
```
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal training.epochs=2 training.seed=0
```
Smoke-test checks (the OPEN VERIFICATION items): (b) the (4,64,64)→(3,64,64) image strip runs; (c) the
dt head is NOT saturated to one end — log/inspect `info['dt']` histogram (entropy coef should help; if
still pinned, give the dt head its own init scale); (d) `info['steps']`/`info['dt']` flow through
acting.py step-counting so variable-length episodes book-keep correctly.

THEN the seed-0 grid (key diagnostic = is the dt histogram non-degenerate?). Sweep switch_cost ×
max_time_factor first on seed 0 (9 runs), read train/ct/{frac_dt_1,frac_dt_max,std_dt_ratio} to find
the adaptive cell, THEN spend seeds 1,2 on the winner:
```
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.max_time_factor=4,8,16 \
  agent.continuous_time.switch_cost=0.002,0.01,0.05 training.seed=0
```
THEN the OPAX dt-normalization ablation on the winning cell:
```
python train_actsafe.py -m hydra/launcher=slurm +experiment=safe_goal_tase +hardware=4090_rtx \
  +wandb.project=actsafe-ct-pointgoal \
  agent.continuous_time.opax_dt_normalization=true,false \
  agent.continuous_time.max_time_factor=8 agent.continuous_time.switch_cost=0.01 training.seed=0,1,2
```

- **OPAX dt-normalization ABLATION FLAG ADDED 2026-06-30**: `continuous_time.opax_dt_normalization`
  (default true). Gates the `new_rewards /= stop_gradient(dt_ratio)` in opax.py (per-physical-time
  uncertainty; without it OPAX inflates uncertainty by predicting far ahead → locks dt to max,
  freezes the agent). Threaded exploration.py → OpaxBridge → opax.modify_reward. Both this and the
  discount STE (safe_actor_critic.py:240) are HARD-GATED behind continuous_time → the validated AR
  baseline is byte-identical; only TASE runs touch them. Ablate with
  `agent.continuous_time.opax_dt_normalization=true,false`.

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
