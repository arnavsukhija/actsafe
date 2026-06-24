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

### Open reconciliation
`make_actor_critic.py` budget scaling: fork uses
`episode_safety_budget = (safety_budget/episode_steps)/(1-safety_discount)` with
`episode_steps = time_limit/action_repeat`, giving 5.0 at AR=2 vs upstream's 2.5 (always divides
by time_limit). The AR sweep is internally fair (same "episode cost ≤ 25" at every AR), but
reconcile this before comparing ABSOLUTE safety numbers to the paper.

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
