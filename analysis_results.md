# Continuous-Time ActSafe: Full Code Review

> Comparing `19df27ff` (Yarden's last commit) → `30b526f` (HEAD)
> **56 commits · 28 files · +756 / −156 lines**

---

## Summary Verdict

Your core architectural changes (continuous-time discounting, SwitchCostWrapper, variable-length replay buffer, async step counting, LBSGD fallback fix, discount hacking fix) are **structurally sound**. The design decisions are well-reasoned. However, I found **4 bugs** (2 critical, 2 moderate) and **2 code smells** that should be addressed.

---

## 🔴 Critical Bugs

### Bug 1: `observe_transition` uses stale `transition` variable from last loop iteration

**File:** [acting.py](file:///home/arnav/workspace/actsafe/actsafe/rl/acting.py#L77)

```python
# Line 77 — OUTSIDE the per-env for-loop
agent.observe_transition(transition, sim_steps=sim_steps_this_call)
```

The `transition` variable here is set *inside* the `for i in range(environment.num_envs)` loop (line 67-74) but `observe_transition` is called **after** the loop exits (line 77). This means `transition` always refers to the **last active env's transition**, not a batched or aggregate transition.

**Impact:** Functionally this "works" because `observe_transition()` only uses the `sim_steps` kwarg (it ignores the transition data itself). But it's semantically wrong and fragile — if `observe_transition` ever starts using the transition data, it'll silently use only env N-1's data.

**Fix:** Move the call out of depending on `transition` at all, or pass `None`:
```python
# After the per-env loop
agent.observe_transition(None, sim_steps=sim_steps_this_call)  # only sim_steps matters
```

Or better: refactor `observe_transition` to accept `sim_steps` as a standalone method.

---

### Bug 2: `compute_lambda_values` uses scalar indexing on `discount` without guarding dimensionality

**File:** [safe_actor_critic.py](file:///home/arnav/workspace/actsafe/actsafe/actsafe/safe_actor_critic.py#L185)

```python
def compute_lambda_values(
    next_values: jax.Array, rewards: jax.Array, discount: jax.Array, lambda_: float
) -> jax.Array:
    tds = rewards + (1.0 - lambda_) * discount * next_values
    tds = tds.at[-1].add(lambda_ * discount[-1] * next_values[-1])  # ← line 185
    return discounted_cumsum(tds, lambda_ * discount)
```

When `discount` is a **scalar** (which happens in the discrete-time `else` branch after `eqx.filter_vmap` operates on uniform arrays), `discount[-1]` will index into a 0-d array. In JAX, `jnp.array(0.99)[-1]` doesn't error but returns the scalar itself, so this **works by accident**. However:

- The type signature says `discount: jax.Array` but the behavior depends on `ndim`.
- `discounted_cumsum` has an explicit `if discount.ndim > 0` guard, but `compute_lambda_values` does not.
- After `eqx.filter_vmap`, the discount arrays passed in are 1-D (shape `(horizon-1,)`), so the `[-1]` indexing works correctly for that case.

**Verdict:** This is **not currently broken** because `eqx.filter_vmap` ensures discount is always 1-D by the time it reaches this function. But the lack of a guard is fragile. Consider adding a comment or assertion.

---

## 🟡 Moderate Bugs

### Bug 3: `critic_loss_fn` receives per-batch discount arrays but calls `compute_discount` unbatched

**File:** [safe_actor_critic.py](file:///home/arnav/workspace/actsafe/actsafe/actsafe/safe_actor_critic.py#L189-L202)

```python
def critic_loss_fn(
    critic: Critic,
    trajectories: jax.Array,
    lambda_values: jax.Array,
    discount: jax.Array,   # shape: [batch, horizon-1]
    horizon: int,
) -> jax.Array:
    planning_discount = compute_discount(discount, horizon - 1)  # ← called WITHOUT vmap
```

In `evaluate_actor` (line 272), `compute_discount` is called with `eqx.filter_vmap`:
```python
planning_discount = eqx.filter_vmap(compute_discount)(discount_current, horizon - 1)
```

But in `critic_loss_fn` (line 196), it's called **without vmap**:
```python
planning_discount = compute_discount(discount, horizon - 1)
```

The `compute_discount` function handles the vector case via `if factor.ndim > 0`, but when `discount` is shape `[batch, horizon-1]`, `factor.ndim == 2`, which triggers the vector branch. The `jnp.cumprod(factor[..., :length-1], axis=-1)` will work along the last axis, which is correct. So this **works by accident** due to the `...` broadcasting in `compute_discount`.

**Verdict:** Functionally correct but should be explicitly vmapped for clarity and safety:
```python
planning_discount = eqx.filter_vmap(compute_discount)(discount, horizon - 1)
```

---

### Bug 4: `EpochSummary.__init__` conflicts with `@dataclass` field defaults

**File:** [epoch_summary.py](file:///home/arnav/workspace/actsafe/actsafe/rl/epoch_summary.py#L10-L17)

```python
@dataclass
class EpochSummary:
    _data: list[list[Trajectory]] = field(default_factory=list)
    cost_boundary: float = 25.0

    def __init__(self, cost_boundary: float = 25.0):  # ← manual __init__ overrides @dataclass
        self._data = []
        self.cost_boundary = cost_boundary
```

You manually wrote `__init__` which **overrides** the one generated by `@dataclass`. This works but defeats the purpose of using `@dataclass`. Either remove the `@dataclass` decorator, or remove the manual `__init__` and rely on the generated one. The generated one already handles `_data` via `default_factory=list` and `cost_boundary=25.0`.

**Verdict:** Not broken, but a code smell. The `@dataclass` decorator is now misleading.

---

## 🟠 Code Smells & Potential Issues

### Smell 1: LBSGD `Penalizer` protocol return type doesn't match implementations

**File:** [safe_actor_critic.py](file:///home/arnav/workspace/actsafe/actsafe/actsafe/safe_actor_critic.py#L34-L43)

The `Penalizer` protocol declares:
```python
def __call__(...) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array], float | None]:
```

But `LBSGDPenalizer.__call__` (lbsgd.py line 94) declares:
```python
def __call__(...) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
```

The actual return on line 122 is `return updates, state, rest, metrics, step_scale` (5 elements), matching the new protocol. But the **type annotation** on line 94 only shows 4 elements. This is a minor annotation bug.

---

### Smell 2: `DummyPenalizer` and `AugmentedLagrangianPenalizer` return `None` for `step_scale`

Both return 5-tuples with `None` as the last element. In `update_safe_actor_critic`, this `None` is passed to `actor_learner.grad_step(..., scale=step_scale)`. The learner handles `None` correctly (line 33: `if scale is not None`), so this works. But it couples three files to an implicit `None`-means-no-scaling convention.

---

## ✅ Verified Correct: Your 3 Core Fixes

### 1. Discount Hacking Fix — ✅ Correct

[safe_actor_critic.py L243](file:///home/arnav/workspace/actsafe/actsafe/actsafe/safe_actor_critic.py#L243):
```python
dt_ratio_nograd = jax.lax.stop_gradient(dt_ratio)
discount = base_discount ** dt_ratio_nograd
safety_discount = base_safety_discount ** dt_ratio_nograd
```

This correctly prevents the actor from backpropagating through the discount to manipulate `dt_ratio`. The STE estimator on line 236 still allows gradients through the *world model predictions*, which is exactly what you want.

### 2. Lagrangian Amnesia Fix — ✅ Correct

[lbsgd.py L51-52](file:///home/arnav/workspace/actsafe/actsafe/actsafe/lbsgd.py#L51-L52):
```python
def fallback():
    return grad_f_1, LBSGDState(eta_t), (0.0, 0.0, 0.0), (backup_lr / base_lr)
```

`LBSGDState(eta_t)` — eta is preserved during fallback instead of being decayed. This is correct: when in safety violation, the multiplier should not relax.

### 3. LBSGD Gradient Decomposition — ✅ Correct

The change from `g * lr / base_lr` to returning `(g, lr/base_lr)` separately and applying `scale` after Adam is correct. This lets Adam track momentum/variance of the actual gradient `g`, with LBSGD only controlling the step size. This is a better decomposition.

---

## 🧠 Evaluation: Does the Opax Normalization Fix Make Sense?

**TL;DR: Yes, the proposed fix is mathematically correct and should work.** But the implementation needs care.

### The Exploit (Your Diagnosis)

The Opax reward is `epistemic_uncertainty(distributions)`. The world model's epistemic uncertainty naturally scales with prediction horizon: predicting 4 steps ahead means more divergence among ensemble members than 1 step. So:

```
opax_reward(dt=4) >> opax_reward(dt=1)
```

Opax maximizes total reward → picks `dt_ratio=4` → safety critic blocks all motion at `dt=4` → frozen agent.

### Why the Agent Froze: Step-by-Step Causal Chain

To understand the freeze, trace the three competing objectives the agent must satisfy simultaneously:

**Step 1: Opax discovers that `dt_ratio` is free curiosity.**

The `normalized_epistemic_uncertainty` in `opax.py` computes the variance across the RSSM ensemble's predictions. When the actor outputs `dt_ratio=4`, the world model calls `jax.lax.scan` for 4 steps, each compounding the stochastic transitions. The ensemble members diverge more over 4 steps than over 1 step simply because randomness accumulates. The epistemic uncertainty at `dt=4` is roughly ~4× larger than at `dt=1`, not because the model is actually uncertain about that region of state space, but because predictions are inherently noisier over longer horizons.

So Opax learns: **"I get 4× the reward for the same state if I just increase `dt_ratio`. I never need to move."**

**Step 2: The safety critic vetoes motion at `dt_ratio=4`.**

The pessimistic safety critic adds `constraint_pessimism × epistemic_uncertainty` to the predicted cost. With `dt_ratio=4`:
- The base epistemic uncertainty is already inflated (per Step 1)
- Multiplied by the pessimism coefficient, the predicted cost for any non-zero force becomes astronomically high
- The LBSGD constraint `safety_budget - safety_lambda_values.mean() > 0` is violated → the agent enters **fallback mode**, where only the cost-minimizing gradient `grad_f_1` is followed

So the safety critic says: **"If you commit to a force for 4 steps, I predict you will crash. Do nothing."**

**Step 3: The agent finds the Nash equilibrium — freeze.**

The actor must satisfy both Opax (maximize uncertainty) and the safety critic (stay safe). The only action that simultaneously:
- Maximizes `dt_ratio` (to inflate Opax reward) ✅
- Has zero predicted cost (to satisfy the safety critic) ✅

...is `force = 0.0, dt_ratio = 4.0`. The cart sits still, doing nothing, while Opax collects massive "free" curiosity reward from the long-horizon prediction noise. The agent gently bounces against the safety boundary at low velocity but never actually explores.

**Why the discrete agent didn't freeze:** In discrete mode, there is no `dt_ratio` dimension to exploit. The epistemic uncertainty is fixed at `dt=action_repeat` for every action. Opax can *only* increase uncertainty by visiting novel states, which is the intended behavior.

```
┌─────────────────────────────────────────────────┐
│            The Freeze Equilibrium               │
│                                                 │
│  Opax says:  "Maximize dt_ratio → free reward"  │
│  Safety says: "dt_ratio=4 + force ≠ 0 → crash"  │
│  Compromise:  dt_ratio=4, force=0 → frozen      │
│                                                 │
│  Fix: Normalize reward by dt_ratio              │
│       → No incentive to inflate duration        │
│       → Opax must move to get reward            │
└─────────────────────────────────────────────────┘
```

### The Proposed Fix

Normalize by time: `opax_reward / dt_ratio`. This changes the Opax objective to "**maximize uncertainty per unit time**", destroying the incentive to inflate duration.

### Why It's Correct

1. **Information-theoretic justification**: You want to maximize the *rate* of information gain per unit of physical time. Dividing by `dt_ratio` converts total information gain to a rate.
2. **Aligns exploration with exploitation**: The task actor already works in per-step terms. Making Opax also reason per-step ensures the two don't fight each other.
3. **Doesn't destroy exploration signal**: `uncertainty(dt=1) / 1` can still be large where the model is genuinely uncertain. It just removes the artificial amplification from duration.

### Implementation Guidance

In [opax.py](file:///home/arnav/workspace/actsafe/actsafe/opax.py), `modify_reward` receives a `trajectory: Prediction`. Since you added `trajectory.action` to the `Prediction` NamedTuple, you can extract `pseudo_time`:

```python
def modify_reward(
    trajectory: Prediction,
    distributions: ShiftScale,
    scale: float = 1.0,
    epistemic_scale: float = 1.0,
    stop_grad: bool = True,
    continuous_time: bool = False,
    tmin: float | None = None,
    tmax: float | None = None,
    base_dt: float | None = None,
) -> tuple[Prediction, ShiftScale]:
    new_rewards = (
        normalized_epistemic_uncertainty(distributions, scale=epistemic_scale) * scale
    )
    
    if continuous_time and tmin is not None and tmax is not None and base_dt is not None:
        pseudo_time = trajectory.action[..., -1]
        time_for_action = ((tmax - tmin) / 2.0 * pseudo_time) + (tmax + tmin) / 2.0
        dt_ratio = jnp.maximum(jnp.round(time_for_action / base_dt), 1.0)
        # Normalize: reward per unit time
        new_rewards = new_rewards / jax.lax.stop_gradient(dt_ratio)
    
    if stop_grad:
        new_rewards = jax.lax.stop_gradient(new_rewards)
    return Prediction(
        trajectory.action,
        trajectory.next_state,
        new_rewards,
        trajectory.cost,
    ), distributions
```

> [!IMPORTANT]
> The `stop_gradient` on `dt_ratio` in the normalization is critical. Without it, Opax could learn to *minimize* `dt_ratio` to inflate the denominator-normalized reward, creating the opposite pathology.

You'll also need to thread `continuous_time`, `tmin`, `tmax`, `base_dt` through `OpaxBridge.sample()` → `opax.modify_reward()`. This means adding those params to `OpaxBridge.__init__` or making them `eqx.field(static=True)` attributes.

---

## 🔧 Hyperparameter Audit: Discrete Defaults vs. Continuous-Time Changes

I traced every config change from Yarden's original to your current `continuous_time_cartpole.yaml` + `actsafe.yaml` defaults. Here's the full comparison:

### Changes in `continuous_time_cartpole.yaml` (experiment-level overrides)

| Parameter | Discrete Original | Current Continuous | Commit | Sound? |
|-----------|------------------|--------------------|--------|--------|
| `action_repeat` | `2` | `1` | initial | ✅ **Correct.** The SwitchCostWrapper handles repetition internally via `dt_ratio`. Setting action_repeat=1 ensures 1:1 base sim steps. |
| `epochs` | `100` | `300` | initial | ✅ **Fine.** Continuous-time needs more epochs because the agent explores at effectively variable rates. |
| `offline_steps` | `200000` (default) | `10000` | `cb6e335` | ⚠️ **Aggressive but correct direction.** 200k let the world model perfectly memorize the safe center, killing Opax's signal. 10k is a 20× reduction. **Recommendation:** Try `25000-50000` as a middle ground — you still want a reasonable world model before Opax kicks in, just not a perfect one. |
| `exploration_steps` | `1000000` | `1000000` | `cb6e335` (reverted from 1.25M) | ✅ **Matches discrete.** Good — same exploration budget in sim-steps terms. |
| `constraint_pessimism` | `50.0` | `2.0` | `30b526f` via `fb42f69`→`30b526f` | ⚠️ **The most impactful change. Needs care.** See detailed analysis below. |
| `init_stddev` | `0.025` | `0.025` | `3ab03bf` (reverted from 5.0) | ✅ **Correct to revert.** 5.0 creates near-uniform initial policy, which is appropriate for discrete but causes the continuous-time actor to output wildly random `dt_ratio` values early on, destabilizing training. 0.025 concentrates initial actions near zero, giving conservative early behavior. |
| `max_time_factor` | N/A | `4` | `dcc90c7` (was 8) | ✅ **Good reduction.** `dt_ratio ∈ [1,4]` means the agent commits to at most 4× the base timestep. 8× was too aggressive — 8 steps into the future is too uncertain for the safety critic to reason about. |
| `switch_cost` | N/A | `0.001` | initial | ✅ **Tiny but non-zero.** Just enough to discourage gratuitous action switching without dominating the reward signal. |
| `sequence_length` | `50` (default) | `50` | `c49a318` (reverted from 16) | ✅ **Correct to revert.** 16 was too short for the world model to learn temporal dynamics. 50 matches discrete. |
| `backup_lr` in penalizer | `1e-2` (default) | `1e-2` (default, removed override) | `dcc90c7` | ✅ **Correct to remove override.** The 8e-5 override was making fallback steps way too small. |

### Changes in `actsafe.yaml` (global defaults affecting ALL experiments)

| Parameter | Yarden's Original | Your Current | Commit | Sound? |
|-----------|-------------------|-------------|--------|--------|
| `actor.initialization_scale` | `0.01` | `0.1` | `bd31a2a` | ⚠️ **Questionable.** This is a **global default** that affects discrete experiments too. 10× larger init scale means larger initial policy weights. This was done in the "updating lbsgd fallback" commit. **Recommendation:** If this was only needed for continuous-time, move it into `continuous_time_cartpole.yaml` as an override rather than changing the global default. Otherwise you risk regressing discrete experiments. |
| `model.continuous_time` | not present | `false` | `ec34278` | ✅ **Correct.** Defaults to false, overridden in the experiment YAML. |
| `continuous_time.*` | not present | added | various | ✅ **Clean addition** with sensible defaults. |

### Changes in `safe_sparse_cartpole.yaml` (discrete experiment)

| Parameter | Yarden's Original | Your Current | Commit | Sound? |
|-----------|-------------------|-------------|--------|--------|
| `constraint_pessimism` | `50.0` | `5.0` | (in diff) | 🔴 **This changes the DISCRETE baseline!** Yarden's original used `50.0` for the discrete experiment. You changed it to `5.0`. This means if you re-run the discrete agent, it will behave differently from the published/validated results. **Recommendation:** Revert this to `50.0` unless you have evidence that 5.0 works better for discrete. |

---

### Deep Dive: `constraint_pessimism` — The Most Critical Knob

This went through the following journey:
```
50.0 (Yarden's original, proven for discrete)
 → 5.0 (your first reduction for continuous-time)
 → 2.0 (your latest "lower pessimism for testing" commit)
```

**Why lowering is necessary for continuous-time:**
- In discrete mode, epistemic uncertainty is bounded (fixed horizon, fixed dt)
- In continuous-time at `dt_ratio=4`, the base epistemic uncertainty is ~4× larger
- `pessimism × uncertainty` scales multiplicatively: `50 × 4× = 200×` the effective pessimism vs discrete
- This completely paralyzes the agent

**Is 2.0 the right value?**

Probably too low. At `pessimism=2.0`, the safety critic only adds 2× the epistemic uncertainty to cost predictions. This means:
- ✅ The agent won't be paralyzed
- ⚠️ The agent might not be pessimistic *enough* — it could underestimate costs near the wall and violate the safety budget

> [!IMPORTANT]
> **Recommendation:** After implementing the Opax normalization fix, you should be able to **increase pessimism back toward 5.0**. The reason you need 2.0 right now is partly because Opax forces `dt_ratio=4`, which inflates uncertainty. Once Opax stops doing that and settles on `dt_ratio≈1-2`, the base uncertainty drops, and higher pessimism becomes tolerable. Start with `pessimism=2.0` + Opax fix, then sweep `[2.0, 3.0, 5.0]` in parallel.

---

### Hyperparameter Recommendations Summary

| Parameter | Current | Recommendation | Priority |
|-----------|---------|---------------|----------|
| `constraint_pessimism` (continuous) | `2.0` | Keep for now, increase to 3-5 after Opax fix | 🟡 |
| `constraint_pessimism` (discrete) | `5.0` | **Revert to `50.0`** to preserve baseline | 🔴 |
| `offline_steps` | `10000` | Try `25000-50000` as middle ground | 🟡 |
| `actor.initialization_scale` (global) | `0.1` | Move to experiment YAML or revert global to `0.01` | 🟡 |
| All others | — | Sound, keep as-is | ✅ |

---

## 📋 Summary of Action Items

| # | Severity | File | Issue | Status |
|---|----------|------|-------|--------|
| 1 | 🔴 Critical | `acting.py:77` | `transition` variable refers to last env only | Fix needed |
| 2 | 🟡 Minor | `safe_actor_critic.py:185` | `discount[-1]` on potentially scalar — works by accident | Add comment |
| 3 | 🟡 Moderate | `safe_actor_critic.py:196` | `compute_discount` called without vmap in `critic_loss_fn` | Works but fragile |
| 4 | 🟡 Minor | `epoch_summary.py:10-17` | `@dataclass` + manual `__init__` conflict | Cleanup |
| 5 | 🟠 Smell | `lbsgd.py:94` | Return type annotation mismatch | Fix annotation |
| 6 | ✅ | `opax.py` | Implement Opax time-normalization fix | **Next step** |

---

## Architecture Assessment

The overall continuous-time extension is well-designed:

- **SwitchCostWrapper** cleanly separates the macro-action logic from the base environment.
- **Variable-length replay buffer** with `lengths` tracking is the right approach.
- **Sim-step-based training counters** ensure apples-to-apples comparison between discrete and continuous agents.
- **STE for dt_ratio rounding** allows gradient flow through the world model while maintaining integer steps.
- **Stopped-gradient discount** is the correct solution to the discount hacking problem.
- **LBSGD gradient/scale decomposition** preserves Adam's adaptive learning rate properties.

The pending Opax normalization is the right next step and should break the curiosity exploit.
