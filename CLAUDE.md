# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ActSafe is a model-based RL research implementation for safe exploration with active learning (ICLR 2025). The core idea: use a learned world model (RSSM) to plan safe trajectories while actively reducing uncertainty. Written in JAX + Equinox.

This fork extends ActSafe toward a **continuous-time / control-frequency safety** research story
(targeting ICLR 2027).

## Current Project State (as of 2026-07-19)

**OPEN-LOOP WORLD-MODEL REVAMP + INTERACTION BUDGET (2026-07-19) — CURRENT.** The eta/LBSGD
hypothesis is closed (2026-07-13/14 sweep: constraint > 0 in 12/12 configs while 8/12 violate
the realized budget; no eta/eta_rate setting calibrates the critic). The diagnostics battery
relocated the failure to the **one-shot variable-k dynamics prediction**: teacher-forced per-k
cost gaps are ≈ 0 at real counts (coverage fixed), but `agent/imagination/cost_return_gap` is
negative in 9/12 runs and per-k reconstruction error grows 30–70% with k. Response
(approved plan, top of `handoff/implementation_plan.md`): the world model executes a hold as
k open-loop micro-predictions (latent-space SwitchCostWrapper) behind
`agent.continuous_time.dynamics: flow | openloop` (`flow` kept as the A/B arm); per-base-step
cost/reward targets stored from the wrapper; imagination aggregates per-decision via a
fractional mask (imagination API contract frozen — `safe_actor_critic`/LBSGD/λ-returns/budget
untouched); plus the auto-tuned **Interaction Budget** (dual ascent on a switch price,
env-side switch cost zeroed when enabled). The SMDP decision layer is unchanged — variable
frequency remains the agent's action. The paragraphs below describe the 2026-07-11 state and
are superseded where they conflict.

**FIRST-PRINCIPLES REVIEW + COVERAGE-FIRST PURGE LANDED (2026-07-11)** — a strict SMDP
architectural review (`code_review.md`, repo root — read it before touching the CT stack)
proved the ZOH/flow foundation sound and replaced the estimator-bias theory with the central
diagnosis: **phase-wise dt-coverage mismatch** (offline explored only k∈[9,16] via a [0,1)
sampling bug; OPAX sits at dt=1 from its FIRST controlled step — the rational optimum of its
objective, not degeneration; the task phase then operates at k the flow model has never seen,
and the imagination-trained safety critic reads optimistically off-distribution). Landed:
**expectile cost head DELETED** (never run; the hazard-rate head likewise DROPPED — flow
identity conflict); `opax_dt_normalization` deleted; `dt_exploration: uniform` default again;
`UniformExploration` dt dim fixed to [−1,1]; diagnostics battery added (`count_k_*`,
`agent/model_per_k/{recon,kl}_k_*`, `agent/imagination/cost_return_*`, `agent/ct/kslope/*`,
`agent/actor/dt_head_grad_norm`); hard-fail guard on pre-exposure pickle resume under CT.
Launch command + read-order gates: top of `handoff/implementation_plan.md`. The paragraphs
below describe the 2026-07-07 state and are superseded where they conflict (notably: expectile
is gone, and the czo8evls "collapse at 250k" read was wrong — 0–200k was the offline phase).

**WAVE-1 CALIBRATION-FIX STACK LANDED (2026-07-07)** — the implemented response to the
2026-07-06 audit below, with a user pivot: **expectile regression** on the cost channel
(`agent.model.cost_head: expectile`, τ=0.9; asymmetric MSE weighting cost UNDER-prediction τ,
over-prediction 1−τ) is the first-attempt fix, chosen over the factored hazard-rate head as
less invasive; the rate head stays fully designed as the escalation path. Also landed:
- **ct_time nearest-integer mapping** (`floor(x+0.5)`, ties up, clipped): fixes the
  "max_repeat unreachable" bug; wrapper/STE/diagnostics all follow `actsafe/rl/ct_time.py`.
  New runs' dt histograms are not bit-comparable to pre-fix runs.
- **Exposure + raw-cost plumbing**: buffer now stores `cost_realized` (raw hold cost) and
  `exposure` (executed base steps) alongside the unchanged discounted `cost`; old pickles
  resume via lazy backfill, but use fresh runs for reported experiments.
- **Per-k calibration metrics** `agent/cost_calibration/gap_k_*` — THE GATE for any cost-head
  fix (pass: gaps ≈ 0, no k-trend; watch `gap_hazard_holds`).
- **dt-adaptiveness metric** `train/ct/buffer/dt_near_far_ratio` (< 1 = decides faster near
  hazards) + k_max-relative coverage quartiles `frac_dt_q1..q4`.
- **Pessimism-source flag** `agent.sentiment.constraint_pessimism_source: latent (default) |
  cost_spread` (κ·ensemble-std of decoded cost; ablation only).
- **Budget-invariance audit tests** (`tests/test_budget_invariance.py`) + extracted
  `compute_episode_safety_budget()`.
DEFERRED to Wave 2 (after the sweep): the auto-tuned interaction budget (dual ascent replacing
fixed switch_cost) and the factored rate head. Launch commands + wandb read guide: top section
of `handoff/implementation_plan.md`.

**The 2026-07-06 budget/critic audit is the headline: the safety-budget math is theoretically
sound in BOTH branches, and every observed safety failure — including the discrete AR study's
"cost rises with action repeat" figure — is a safety-critic CALIBRATION error, not a budget
flaw.** Full audit (equations + the 53-run discrete table): `handoff/implementation_plan.md`.

**Audit findings (2026-07-06) — treat these as settled:**
1. **Discrete budget is sound.** `B(R) = d·R/(T(1−γc))` is an exact unit conversion (per-agent-step
   cost targets also scale with R; the R cancels → realized allowance is d=25 at every repeat).
   Budget-filling would predict realized cost FLAT at 25 across R; the observed slope is instead
   the critic's calibration error: LBSGD servoes the *perceived* cost value to ~90% of budget at
   every R (e.g. V̂=18.1±0.3 vs B=20 across all 16 AR=8 lineages), and the critic's bias flips
   sign between AR=4 and AR=8 (pessimistic → optimistic). Run-level corr(violation, calibration
   gap) = **+0.91**; all 22 violating runs have a positive gap. The high-repeat collapse is purely
   critic miscalibration. The "coarse control is physically costlier" claim is UNIDENTIFIED in
   current data (needs matched realized-violation comparisons). The TASE budget
   `d/(T(1−γc)) = 2.5` is dt-independent and chunk-invariant (telescoping identity) — not
   exploitable by holding longer.
2. **Root cause: MSE regression on time-aggregated sparse costs.** The Gaussian/MSE cost decoder
   regresses zero-inflated accumulated targets with support [0, R] (or [0, k] in TASE); shrinkage
   toward the conditional mean is one-sided OPTIMISTIC with absolute bias growing with target
   scale, i.e. with dt. κ·ensemble-std pessimism cannot cover it (systematic, not epistemic;
   measured gap ≈ 28× std). In TASE this also flattens the learned k-slope, giving the
   wrong-signed perceived-safety gradient ∂Q̂/∂k < 0 (longer holds *look* safer) — the agent
   hacks the critic, not the budget (corr(mean_dt, optimism gap) = +0.40 at rep16).
3. **Planned principled fix — factored hazard-rate head:** replace the learned time-aggregate
   with a per-base-step hazard probability p̂(s,u) trained via Bernoulli/cross-entropy, and
   compose the hold cost analytically (Σ_{i<k} γc^i p̂ᵢ, or p̂·(1−γc^k)/(1−γc) under local
   stationarity). Targets have support {0,1} at every R/k (kills the scale-dependent shrinkage
   and the discrete sign flip), the k-dependence becomes exact (restores ∂Q̂/∂k > 0 near
   hazards), and CE on Bernoulli targets is a proper scoring rule (calibrated by construction).
   NOT yet implemented — audit approved, code pending.
4. **Exploratory: auto-tuned interaction budget.** Shift from a fixed `switch_cost` to a dynamic
   per-episode interaction (decision) budget enforced via dual gradient ascent on the switch
   price — motivated by the finding that a fixed price drifts (becomes relatively cheaper as the
   policy improves) and produces rational bang-bang dt rather than mistuning. Design stage only.

**Previous milestone — the vanilla-faithful TASE cleanup** landed on `wip/tase-pointgoal`
(7 commits: `af75f9b` … `c2c48a0`); the 24-run sweep on the new stack settled the coverage
question. Full landing record + sweep read: top section of
[`handoff/implementation_plan.md`](handoff/implementation_plan.md).

**What the cleanup changed (audit → user directives, 2026-07-05):**
- `safety_dt_gradient` injection REMOVED; `opax_dt_normalization` default `false` (flag kept for
  ablations only). Discount `γ^dt_ratio` is FULLY DIFFERENTIABLE — deliberate, vanilla-faithful
  (supervisor decision 2026-07-05); the only stop-gradients are upstream's OPAX-bonus one and the
  STE-internal one (the estimator mechanism itself).
- **Repeat-units parametrization:** config keys are `agent.continuous_time.min_repeat`/`max_repeat`
  (integer hold bounds in base control steps); physical seconds and the `t_min/t_max/base_dt`
  runtime injection are GONE (with the resume bug they caused). Single source of truth for the
  pseudo-time → hold-length map, STE, and buffer-coverage metrics: `actsafe/rl/ct_time.py`.
- **Full-MDP time:** agent state = (latent, elapsed-time fraction). The clock channel (uint8-safe,
  `255·elapsed/horizon`, counts UP by EXECUTED steps) stays out of the CNNs but is carried as a
  scalar into the policy/critics (`state_dim+1`) and propagated analytically in imagination.
  This is the primary novel contribution — do NOT strip time from the agent's state.
- STE uses `floor` (matches `SwitchCostWrapper` execution); `StateWriter` checkpoint writes are
  atomic; per-epoch `train/ct/buffer/*` metrics report the replay buffer's dt-coverage in-run.

**Open questions #1–#3 from the previous investigation — resolved:**
1. OPAX dt-normalization: default OFF (non-vanilla; the dt≡1 OPAX optimum is correct for an
   uncosted info-rate objective, not a failure mode). Kept only as an ablation flag.
2. Natural dt coverage: **measured, and it IS poor.** The 2026-07-05 sweep (dt_exploration=policy)
   shows `train/ct/buffer/frac_dt_1` = 0.85–0.94 in all 24 runs at 550–600k steps.
   **Executive decision (user): `dt_exploration: uniform` is now the default in
   `safe_goal_tase.yaml`** — during the exploration window the executed action's dt head is
   resampled uniformly over pseudo-time [−1,1] (mechanism in `ActSafe.__call__`). The running
   policy-mode sweep is kept as the natural-coverage control arm.
3. Discrete repeat-counter redesign: dropped — the repeat-units refactor already removed the
   scaling machinery; the continuous head + floor is behaviorally equivalent and vanilla-shaped.

**Current sweep (running):** `max_repeat=8,16 × switch_cost=0.002,0.005,0.01,0.05 × seed=0,1,2`,
project `actsafe-ct-pointgoal`. Next: launch the identical grid with the updated config
(uniform dt exploration) as the "after" arm and compare `train/ct/buffer/near_hazard_*` and the
constraint-optimism gap. The cost-underestimation problem (`agent/safety_critic/constraint`
positive everywhere while realized cost_return violates the budget) is now **diagnosed** —
see audit findings #1–#2 above — with the hazard-rate head (#3) as the approved fix path.
Known small bug to land alongside: the pseudo-time → hold map never reaches `max_repeat`
(needs `p=1.0` exactly after tanh; fix: map to `[k_min, k_max+1)` before floor, with clip).
Also pending before any adaptiveness claim: a dt-vs-hazard-proximity metric.

## Handoff & Current Status — READ THIS FIRST

**All project documentation, decisions, progress, and the running plan live in [`handoff/`](handoff/).**
It is a checked-in mirror of the assistant's auto-memory so any device or new chat can resume
where work left off. Start with:

- [`handoff/implementation_plan.md`](handoff/implementation_plan.md) — single source of truth for
  "where we are" and "what's next" (current task: Safety-Gym PointGoal AR safety-frequency study).
- [`handoff/project_strategy_2026-06-23.md`](handoff/project_strategy_2026-06-23.md) — the decisions
  (PointGoal pivot, all-in ICLR 2027, the load-bearing CT cost-accounting finding).
- [`handoff/project_paper_direction.md`](handoff/project_paper_direction.md),
  [`handoff/project_bugs_fixed.md`](handoff/project_bugs_fixed.md),
  [`handoff/project_ct_architecture.md`](handoff/project_ct_architecture.md),
  [`handoff/project_cluster_infra.md`](handoff/project_cluster_infra.md) — supporting context.
- [`handoff/MEMORY.md`](handoff/MEMORY.md) — index of the above.

When you make a meaningful decision or finish a milestone, update `handoff/implementation_plan.md`
so the handoff stays current.

## Commands

```bash
# Install
poetry install

# Lint / format
ruff check actsafe/
ruff format actsafe/

# Type check
mypy actsafe/

# Tests
pytest -v
pytest tests/test_trainer.py -v   # single test file

# Training (Hydra-based)
python train_actsafe.py                                        # default: SafetyGym + ActSafe
python train_actsafe.py experiment=safety_gym                  # named experiment preset
python train_actsafe.py training.epochs=50 agent.plan_horizon=15  # override hyperparams
python train_actsafe.py --help                                 # show all Hydra options
```

## Architecture

### Core Training Flow

```
train_actsafe.py  →  Trainer (rl/trainer.py)  →  ActSafe agent (actsafe/actsafe/actsafe.py)
                                                         ├── WorldModel (world_model.py)
                                                         │     └── RSSM (rssm.py)
                                                         ├── SafeModelBasedActorCritic (safe_actor_critic.py)
                                                         │     └── ActorCritic (actor_critic.py)
                                                         ├── ReplayBuffer (replay_buffer.py)
                                                         └── Exploration (exploration.py)
```

Each epoch: `Trainer` calls `acting.interact()` (collect real experience) → `agent.learn()` (world model + policy updates via imagination rollouts).

### Key Modules

**`actsafe/actsafe/actsafe.py`** — Agent entry point. `policy()` maps observations to actions; `observe()` stores transitions; `learn()` triggers world model and actor-critic gradient steps; `report()` logs metrics and videos.

**`actsafe/actsafe/world_model.py`** — Probabilistic RSSM ensemble. `variational_step()` computes the ELBO loss. Encodes pixel/proprioceptive observations and predicts future states, rewards, and costs.

**`actsafe/actsafe/safe_actor_critic.py`** — Constrained actor-critic. `imagine()` rolls out trajectories in the learned model. Maintains a separate safety-critic for cost estimation; combines with `lbsgd.py` (Lagrangian-based constrained optimizer) for safety constraint satisfaction.

**`actsafe/actsafe/rssm.py`** — Recurrent State Space Model. Maintains stochastic + deterministic latent state. Core of the world model.

**`actsafe/rl/trainer.py`** — Training orchestration. `Trainer` is a context manager that serializes full training state via pickle for resumption. `from_pickle()` restores a run. Handles dynamic `dt` extraction for continuous-time experiments.

**`actsafe/rl/acting.py`** — Environment interaction. `interact()` rolls out the current policy, supporting parallel envs via `EpisodicAsyncEnv`. Records videos and collects trajectories.

**`actsafe/actsafe/exploration.py`** — Exploration strategy (entropy-based or OPAX) for active uncertainty reduction, a core contribution of ActSafe.

**`actsafe/actsafe/sentiment.py`** — Ensemble-based epistemic uncertainty quantification used for the active exploration bonus.

**`actsafe/actsafe/augmented_lagrangian.py`** + **`lbsgd.py`** — Safety constraint enforcement via augmented Lagrangian penalty and constrained SGD.

### Configuration System (Hydra)

All configs live in `actsafe/configs/`. Hierarchy:
- `config.yaml` — training loop, logging, JIT settings
- `agent/actsafe.yaml` — model sizes, LRs, safety budget, plan horizon
- `environment/*.yaml` — env selection (SafetyGym, CartPole, DMControl, Humanoid)
- `experiment/*.yaml` — predefined experiment templates (e.g. `safety_gym`, `continuous_time_cartpole`)
- `hardware/*.yaml` — GPU-specific presets (RTX 2080/3090/4090)

Key safety hyperparameters: `budget=25` (cost constraint), `lambda_=0.95` (Lagrangian multiplier), `plan_horizon=15`, `safety_discount=0.99`.

### Benchmark Suites

Environment wrappers in `actsafe/benchmark_suites/`: `safe_adaptation_gym/` (primary SafetyGym integration), `dm_control/`, `humanoid_bench/`. Each exposes a standard Gymnasium interface.

### Common Utilities

`actsafe/common/`: `learner.py` (Adam wrapper), `mixed_precision.py` (half-precision), `double_buffer.py` (data sync between JAX devices).

## Tech Stack

- **JAX + Equinox** — all neural networks are functional (no mutable state); JIT-compiled training steps
- **Optax + custom `opax.py`** — gradient-based optimization with Equinox-compatible transforms
- **Distrax** — probabilistic distributions (Gaussian, categorical) for RSSM
- **Hydra** — experiment configuration and multi-run sweeps
- **W&B / TensorBoard** — logging (`rl/logging.py`)
- **Poetry** — dependency management (`pyproject.toml`)
- **Ruff** (line-length=88) + **mypy** (with numpy plugin) for linting/typing

# Fork Changes (continuous-time RL) — see `handoff/` for the full record

This fork adds continuous-time / control-frequency machinery on top of upstream ActSafe. The
**full, current documentation of every change, decision, bug fix, and the running plan lives in
[`handoff/`](handoff/)** (see "Handoff & Current Status" above). The themes, in brief:

1. **Continuous-time RL & env wrappers** — `SwitchCostWrapper` (agent outputs an action plus a
   pseudo-time mapped to an integer hold length in `[min_repeat, max_repeat]` base steps — see
   `actsafe/rl/ct_time.py`, the single source of truth); variable-length episode handling in the
   replay buffer; a count-up elapsed-time clock channel (uint8-safe `255·elapsed/horizon`) that is
   kept OUT of the CNNs but carried as a scalar in the agent's state (full-MDP time).
2. **Discounting & optimization** — variable discount `base_discount ** dt_ratio` with a
   straight-through estimator (floor, matching wrapper execution). The discount is FULLY
   DIFFERENTIABLE w.r.t. the dt head — this is deliberate and vanilla-faithful (decided with the
   supervisor, 2026-07-05). The STE's internal `stop_gradient` is the estimator mechanism itself
   (floor has zero derivative a.e.), not a gradient block. Earlier versions of this doc claimed a
   `stop_gradient` on `dt_ratio` in the discount; that was removed in `66ea7fe`/`fd6fe3b` and is
   intentionally absent.
3. **Safety & LBSGD** — world-model `sample()` shape-bug fix; action-repeat-aware safety-budget
   scaling; the LBSGD fallback step scaled downstream of Adam (so the safety-recovery step isn't
   normalized away); `jnp.maximum(constraint, 1e-12)` log-barrier guard.
4. **Euler cluster / MLOps** — async wandb writer (deadlock-safe, drops on full queue); MuJoCo EGL
   headless rendering; JAX memory-preallocation limits; requeue + pickle resume.

> Note: items 1–4 above are a high-level index only. For exact file/line references, current
> status, known-open bugs (e.g. the load-bearing within-window cost-discounting leak), and the
> next-actions list, **read [`handoff/implementation_plan.md`](handoff/implementation_plan.md)** —
> do not treat this summary as authoritative on its own.

