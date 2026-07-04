---
name: project-dt-gradient-trace-2026-07-04
description: "Traced the safety-cost→dt gradient path; corrects the 2026-07-03 claim that Bug A is absent. dt collapse is driven by artifactual discount gradients, not a severed path."
metadata:
  node_type: memory
  type: project
---

# dt gradient-path trace (2026-07-04)

Symptom under investigation: `frac_dt_1 → 1`, `std_dt_ratio → 0` (dt collapses to 1) while
`cost_return ≈ 30–45 > budget 25`. User hypothesis was "a stop_gradient on the variable
discount severed the safety→dt path." **The trace shows the OPPOSITE is the actual state.**

## What the code actually does (safe_actor_critic.py `evaluate_actor`, lines 230-296)
- dt head = last action dim (`pseudo_time = actions[..., -1]`). STE at line 240:
  `dt_ratio = dt_raw + stop_gradient(max(round(dt_raw),1) - dt_raw)` → gradient of dt_ratio
  w.r.t. pseudo_time is identity (correct; this is NOT a severing stop_gradient).
- `discount = base_discount ** dt_ratio`, `safety_discount = base_safety_discount ** dt_ratio`
  — **fully differentiable w.r.t. dt_ratio.** There is NO `stop_gradient(dt_ratio)` on the
  discount. Git: it was added (9621d85) then DELIBERATELY REMOVED — `66ea7fe` (reward discount)
  and `fd6fe3b` (safety discount), both 2026-05-05. So historical "Bug A" (differentiable
  discount → dt-hacking) is **PRESENT**, contradicting implementation_plan.md's 2026-07-03 note
  which conflated the STE stop_gradient with the discount stop_gradient. See [[project-bugs-fixed]].

## Two channels carry dt into the safety cost / LBSGD barrier
Constraint = `budget - safety_lambda_values.mean()`; LBSGD barrier `-eta·log(constraint)` and the
fallback `grad(-constraint)` both propagate its gradient to the actor. So ∂constraint/∂dt reaches
the dt head. It is a SUM of:
1. **Channel 1 (model dynamics, right sign, weak):** pseudo_time → action → `cell.predict` →
   state → `reward_cost_decoder` → `trajectories.cost`. Longer hold → model predicts more cost →
   pushes dt DOWN in violation. Real but depends on the model having learned cost-vs-dt, and the
   target is the within-hold *discounted* cost (wrappers.py:205) which saturates in dt (mild at
   γ=0.99 over dt≤16, ~8%).
2. **Channel 2 (discount, wrong sign, strong):** dt_ratio → `safety_discount = γc^dt` →
   compounded through `compute_lambda_values`. Larger dt → smaller γc^dt → smaller
   safety_lambda → LARGER constraint (looks safer) → pushes dt UP (classic cost-hacking).

**Verdict:** ∂(accumulated cost)/∂(dt_logits) is NONZERO but is a right-signed-weak (Ch1) minus
wrong-signed-strong (Ch2) mixture → the safety constraint cannot reliably pull dt down. Meanwhile
the identical differentiable-discount artifact on the REWARD side pushes dt→1 (smaller dt = less
discount = higher discounted value), and that reward artifact dominates → observed collapse to
dt=1. The collapse is driven by the *removed* stop_gradients, not a severed path.

## Accounting audit (physical-time vs step)
- Budget (make_actor_critic.py CT path): `25/(time_limit·(1-safety_discount)) = 2.5`;
  time_limit=1000 base steps, discount per base step (γ^dt_ratio) → **physical-time based. OK.**
- Reward/cost discount = γ^dt_ratio per agent step, dt_ratio = #base steps → **physical. OK**
  (the elegant CT property; the discrete AR study's per-decision discount confound does NOT
  apply here).
- Switch cost = flat per-DECISION reward penalty (wrappers.py:216); no budget denominator. Intended
  cost-of-control (more decisions → more penalty). Not a bug.
- **Reporting bug:** `info['cost']` (→ Transition → `train/cost_return`) is the within-hold
  DISCOUNTED sum, not the raw physical sum. `info['cost_realized']` holds the raw sum but is
  NOT wired to the log (wrappers.py:234 TODO). So the reported cost under-counts long holds →
  makes TASE look safer than it physically is (the gaming concern). Fix = log cost_realized.

## Proposed fix direction (pending user decision on accounting semantics)
Reinstate `stop_gradient` on both discounts (kill Ch2 hacking + the dt→1 reward artifact) AND add
an explicit, differentiable, dt-scaling pessimistic-cost term so safety shapes dt through physics,
not the discount. Add config `agent.continuous_time.safety_dt_gradient: bool` to detach that term
for the ablation. Surgical; no hyperparameter tuning.
