---
name: project-bugs-fixed
description: Bugs introduced by the user into the CT ActSafe implementation that have been identified and fixed
metadata: 
  node_type: memory
  type: project
  originSessionId: 9686672c-25cf-4aa9-82a2-e74562b76208
---

# Confirmed Bugs Fixed in ActSafe CT Implementation

## Bug 1: Safety Budget Formula Missing action_repeat Factor
**File**: `actsafe/actsafe/make_actor_critic.py`  
**Status**: FIXED (previous session)

Old: `(safety_budget / time_limit) / (1 - safety_discount)` → budget = 10 (wrong)  
Fixed: `(safety_budget / (time_limit / action_repeat)) / (1 - safety_discount)` → budget = 20

This was the root cause of LBSGD being permanently in fallback for discrete AR=2.

---

## Bug 2: floor vs round Mismatch in CT Straight-Through Estimator
**File**: `actsafe/actsafe/safe_actor_critic.py` ~line 236  
**Status**: FIXED (previous session)

STE must use `jnp.floor` to match `SwitchCostWrapper.compute_time()`. Was using `jnp.round`.

---

## Bug 3: safety_discount=1.0 Breaks Bellman Contraction
**Status**: NEVER use — kept as anti-pattern warning

---

## Bug A: Missing stop_gradient on Discount (CRITICAL — CT primary failure cause)
**File**: `actsafe/actsafe/safe_actor_critic.py` ~line 239  
**Status**: FIXED (2026-06-17)

Without stop_gradient: actor gradients flow through `base_discount ** dt_ratio`.
Gradient ∂(γ^dt)/∂dt = log(γ) × γ^dt × cost < 0 → actor learns to increase dt to
shrink γ^dt → 0 and hide future safety costs in imagination. LBSGD never gets a
real corrective gradient. Real env costs accumulate normally. Permanent disconnect.

This was the **primary cause** of CT runs being stuck at cost_return 600–900 with
no improvement across 3M steps.

Fixed:
```python
_dt_sg = jax.lax.stop_gradient(dt_ratio)
discount = base_discount ** _dt_sg
safety_discount = base_safety_discount ** _dt_sg
```

---

## Bug B: round vs floor in OPAX CT Normalization
**File**: `actsafe/actsafe/opax.py` line 33  
**Status**: FIXED (2026-06-17)

OPAX normalized epistemic reward by `jnp.round(time_for_action / base_dt)` instead of
`jnp.floor(...)`. Inconsistency with SwitchCostWrapper's floor semantics partially
defeated the anti-dt-exploitation guard.

---

## Bug C: round vs floor in CT Diagnostic Logging
**File**: `actsafe/rl/epoch_summary.py` lines 103, 110  
**Status**: FIXED (2026-06-17)  
Diagnostic only — does not affect training.
