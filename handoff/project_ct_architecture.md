---
name: project-ct-architecture
description: "Continuous-time ActSafe implementation details — key design decisions, data flow, and invariants"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9686672c-25cf-4aa9-82a2-e74562b76208
---

# Continuous-Time ActSafe Architecture

## Key CT Design Decisions

### SwitchCostWrapper (actsafe/rl/wrappers.py)
- `compute_time()` uses `floor(time_for_action / base_dt) * base_dt` — FLOOR, not round
- `step()` does `num_repetitions = int(round(time_for_action / base_dt))` but since compute_time already floors, this is effectively floor too
- Costs are accumulated per ENV step: `total_cost += info.get("cost", 0.0)` — sum of `action_repeat` base-step costs

### Straight-Through Estimator in safe_actor_critic.py (line ~236)
Must use `jnp.floor` to match SwitchCostWrapper:
```python
dt_ratio = dt_raw + jax.lax.stop_gradient(jnp.maximum(jnp.floor(dt_raw), 1.0) - dt_raw)
```
Forward: discrete floor value. Backward: identity gradient through dt_raw → actor learns pseudo_time via backprop.

### Stop Gradient on Discount
`discount = base_discount ** dt_ratio` uses `stop_gradient` on dt_ratio to prevent the actor from hacking safety by increasing dt to shrink `γ^dt → 0` and hide future costs.

### time_to_go Stripping
The world model CNN strips the scalar `time_to_go` channel before convolution — it's a homogeneous scalar with no spatial structure that degraded image reconstruction when fed into conv layers.

## CT Safety Budget
In CT mode, `action_repeat=1` (default), so the budget formula in `make_actor_critic.py`:
```python
episode_steps = cfg.training.time_limit / cfg.training.action_repeat
episode_safety_budget = (cfg.training.safety_budget / episode_steps) / (1.0 - cfg.agent.safety_discount)
```
gives the same result as dividing by `time_limit` directly (since action_repeat=1). No special CT handling needed.

## LBSGD Safety Constraint
- `constraint = safety_budget - safety_lambda_values.mean()`
- LBSGD "happy case" when `constraint > _EPS` (agent is safe, optimize normally with safety penalty)
- LBSGD "fallback" when `constraint ≤ _EPS` (agent is unsafe, take emergency safety step)
- `safe` metric = fraction of time in happy case (want this = 1.0 for safe agents)
- `cost_value` logged = `safety_lambda_values.mean()` — should be << `safety_budget` for safe agents

## Key Config Values (safe_sparse_cartpole experiment)
- `training.action_repeat = 2`
- `training.safety_budget = 100`
- `training.time_limit = 1000` (base sim steps)
- `agent.safety_discount = 0.99` (default, must NOT be overridden to 1.0)
- `agent.sentiment.constraint_pessimism = 50.0`
- `agent.exploration_strategy = opax`
- Computed `episode_safety_budget = (100/500)/0.01 = 20`
