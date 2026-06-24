---
name: project-strategy-2026-06-23
description: "Decided direction (Safety-Gym PointGoal, all-in ICLR 2027) and the load-bearing CT cost-accounting finding, as of 2026-06-23"
metadata: 
  node_type: memory
  type: project
  originSessionId: 95708d4e-d445-499b-bc2e-07ab11aaa4d3
---

# ActSafe-CT Strategy Decision (2026-06-23)

## Decisions made
- **Task pivot: move off dm_control cartpole to Safety-Gym PointGoal** for the whole story.
  Cartpole `safe_swingup_sparse_hard` cannot prove the thesis: across the AR∈{1,2,4,8} sweep
  realized cost stayed under budget at EVERY frequency, so the safety-frequency U-shape (the
  money plot) physically cannot appear there, and it barely reproduces (1/12 seeds). PointGoal
  is ActSafe's home task (more faithful reproduction) and its hazards+momentum should make a
  long open-loop hold actually cause violations. See [[project-paper-direction]].
- **User is all-in on ICLR 2027** (~Sept 2026 deadline, ~10 weeks). Build full pipeline but
  keep the kill gate: if the AR violation U-shape doesn't appear on PointGoal by ~week 4, that
  is the signal to rescope to a workshop (PointGoal is the most favorable task; if not there,
  nowhere).

## Critical path
1. Wire `SwitchCostWrapper` into `safe_adaptation_gym` (currently dm_control only) + reproduce
   discrete ActSafe on PointGoal at AR=1.
2. GO/NO-GO: fixed-AR sweep {1,2,4,8,16}, plot violation rate vs AR — need the U-shape.
3. CT: budget-invariance test, then money plot (dt/freq vs distance-to-hazard).
4. Ablate switch_cost=0 (headline: adaptation is safety-driven), exploration_pessimism decoupling.

## Verified this session
- Upstream `yardenas/actsafe` added as git remote `upstream`. merge-base == upstream HEAD, so
  every fork diff is the user's own change (no upstream drift to chase).
- Discrete path is faithful to upstream (no discrete-algo bug introduced). acting.py data-fix is
  semantically equivalent to upstream for discrete; CT code paths all gated on continuous_time.

## LOAD-BEARING CT FINDING — safety cost is discounted within the macro-window
`wrappers.py:191-192` (SwitchCostWrapper.step): `total_cost += (self.discounting ** current_step)
* cost`. The safety cost is discounted WITHIN the hold window, so the same physical trajectory
reports LESS cost when held longer → the agent can game the budget by acting less frequently, and
this is the OPPOSITE of the paper's thesis (low freq should mean MORE violations). `self.discounting`
is also independent of `agent.safety_discount=0.99`. **Fix before trusting any CT safety number:**
accumulate safety cost as a RAW undiscounted sum within the window (reward may stay discounted), then
run the budget-invariance test (fixed policy at forced dt values → realized episode cost must not
change with dt). This IS the "budget invariance" check flagged but never run. See [[project-ct-architecture]].

## GO/NO-GO experiment SET UP (2026-06-23)
- Config `actsafe/configs/experiment/safe_goal_ar_study.yaml`: discrete ActSafe on
  Safety-Gym `go_to_goal` (PointGoal, dense reward, 9 hazards+10 vases — verified via
  env introspection `env._world.task.obstacles=[9,10,0,1]`). Dense reward chosen on
  purpose to avoid the cartpole do-nothing confound. constraint_pessimism left null
  (=bayes/mean cost, paper-faithful Safety-Gym default; null/0 → bayes via sentiment.py:13).
- Discrete AR sweep is FAIR by construction: TimeLimit wraps base env then ActionRepeat
  (episodic_async_env.py:193-209) → same 1000 base steps/episode at every AR; ActionRepeat
  sums RAW cost (wrappers.py:26-27, no within-window discounting) → realized cost
  frequency-invariant; budget formula scales with AR → same "episode cost ≤ 25" constraint.
- Fixed slurm.yaml `max_num_timeout: 0 → 100` (the uncommitted 0 would truncate every run
  at the 4h timeout_min=240 wall; requeue+pickle-resume needed for ~5M-step runs).
- Launch: `+experiment=safe_goal_ar_study +hardware=4090_rtx hydra/launcher=slurm
  +wandb.project=actsafe-ct-pointgoal training.action_repeat=1,2,4,8 training.seed=0,1,2`
  (run AR=1 seed=0 alone FIRST as reproduction sanity). Docs: docs/ar_safety_study.md,
  plot: docs/plot_ar_safety_study.py (needs wandb auth, absent in Claude sandbox).
- wandb NOT queryable from the Claude env (no api_key/.netrc); rely on user to read runs
  or the plot script. Hypothesis: cost_return rises with AR & crosses budget=25 → GO;
  flat at all AR incl 8 → thesis unsupported → honest re-scope with supervisor.

## User state (2026-06-23): demoralized, considering scrapping; has been avoiding the
supervisor until there is a result. Agreed plan: this single clean experiment IS the
result+clarity to bring to the supervisor. Per-run discipline: don't launch a run unless
its one-sentence hypothesis is crisp. See [[feedback-style]].

## CT gradient design is otherwise correct
dt-as-action-dim (WM learns macro dynamics from data); discount stop_gradient kills the analytic
discount-hack so dt only gets gradient through WM dynamics (the STE on line 236 is now redundant but
harmless); OPAX /dt_ratio stop_grad normalization is correct. The risk is the cost-accounting leak
above, not the gradients.
