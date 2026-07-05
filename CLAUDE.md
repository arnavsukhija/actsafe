# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ActSafe is a model-based RL research implementation for safe exploration with active learning (ICLR 2025). The core idea: use a learned world model (RSSM) to plan safe trajectories while actively reducing uncertainty. Written in JAX + Equinox.

This fork extends ActSafe toward a **continuous-time / control-frequency safety** research story
(targeting ICLR 2027).

## Current Project State (as of 2026-07-04)

**Live investigation: OPAX dt-collapse during TASE exploration, and whether three CT design
choices are actually necessary.** Not yet resolved — read this before continuing the thread.

**Background.** TASE agent outputs action `u` + pseudo-time `t ∈ [0,1]`, mapped affinely to hold
duration `dt`. During the OPAX exploration phase (first `exploration_steps=500000` steps), fresh
(non-resumed) wandb segments show a clean, reproducible collapse: steps 50–200k have `mean_dt≈12`
(near max), but steps 250–500k show `mean_dt≡1.00` and `frac_dt_1≡1.00` exactly — 100% of decisions
pick `dt=1` — before partially recovering by 550k+. Separately, `agent/safety_critic/constraint` is
positive everywhere across 178 CT runs (0.39–0.74, `frac>0=1.00`) while realized `cost_return`
violates the budget (25–55 vs budget 2.5) — the model/critic is optimistic about cost in dt-space;
this is not an enforcement failure.

**Bugs found and fixed this session:**
- Resume-path telemetry loss: `t_min`/`t_max`/`base_dt` were injected only in `make_agent()` on
  fresh start, not on pickle resume, so post-resume `dt_ratio` logging silently fell back to
  uninitialized 0/0/1 (mechanically confirmed via `std≡0`). Fixed by moving injection into
  `Trainer.__enter__` so it runs on both paths. This means: **the 250–500k collapse window is real
  OPAX behavior, not a resume artifact** — it's only visible pre-fix in fresh segments because
  resumed segments showed `dt≡1` throughout for the wrong (logging) reason.
- Objective conflation: `train/objective` was penalized (included `-switch_cost`), not comparable
  across `switch_cost` sweep values. Fixed by plumbing `reward_realized` (undiscounted, unpenalized)
  through `SwitchCostWrapper → Transition → Trajectory.as_numpy() → EpochSummary → Trainer`, now
  logged separately as `train/objective_raw`. Committed as `a5a9674`.
- `StateWriter` (`actsafe/rl/logging.py:247`) pickle corruption: wrote directly to `state.pkl` in
  `"wb"` mode (truncates before write), so a SIGKILL mid-write (SLURM requeue or manual kill)
  destroys the checkpoint with no fallback. **Fixed and already applied** (see current file
  contents at lines 266–278): writes to `state.pkl.tmp`, `fsync`s, then `os.replace()`s atomically.

**Open, unresolved questions — user is explicitly skeptical of the current design and wants
evidence, not theory, before accepting any of these:**
1. Is `opax_dt_normalization` (dividing the OPAX bonus by `stop_gradient(dt_ratio)`) actually
   necessary? The theoretical claim (`actsafe/actsafe/opax.py` around line 32) is that without it
   the actor inflates uncertainty by predicting far ahead and locks `dt` to max. No empirical
   evidence has been shown for this failure mode specifically.
2. Is the explicit `dt_exploration=uniform` warm-up phase needed, or does OPAX naturally explore
   the dt head on its own? Unknown — **the replay buffer's actual dt composition during/after the
   250–500k collapse has never been directly inspected** (would need `diagnose_dt_coverage.py` on
   a completed checkpoint's buffer + a two-curve cost-head dt-sensitivity check). No verified pickle
   checkpoint has been available yet to run this — this is the highest-priority next step.
3. Should `dt` be a continuous action at all (pseudo-time → affine-mapped hold duration, requiring
   min/max time-repeat scaling), or would a discrete repeat-counter design (wrapper just receives
   the repeat value directly, no need to scale `t_min`/`t_max`) be simpler with similar
   interpolation behavior? Raised by the user, not yet investigated — no code changes made.

**Do not commit to an ablation sweep or redesign before verifying #2** (buffer dt composition and
cost-head dt-sensitivity) on a real checkpoint — that data should settle whether #1 and #3 are
worth pursuing, per the user's explicit instruction to check data before agreeing with or proposing
fixes.

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

1. **Continuous-time RL & env wrappers** — `SwitchCostWrapper` + adaptive `ActionRepeat` (agent
   outputs an action plus a duration `dt`); variable-length episode handling in the replay buffer;
   step counting via `base_dt`; stripping the scalar `time_to_go` channel before the world-model CNN.
2. **Discounting & optimization** — variable discount `base_discount ** dt_ratio` with a
   straight-through estimator. The discount is FULLY DIFFERENTIABLE w.r.t. the dt head — this is
   deliberate and vanilla-faithful (decided with the supervisor, 2026-07-05). The STE's internal
   `stop_gradient` is the estimator mechanism itself (round has zero derivative a.e.), not a
   gradient block. Earlier versions of this doc claimed a `stop_gradient` on `dt_ratio` in the
   discount; that was removed in `66ea7fe`/`fd6fe3b` and is intentionally absent.
3. **Safety & LBSGD** — world-model `sample()` shape-bug fix; action-repeat-aware safety-budget
   scaling; the LBSGD fallback step scaled downstream of Adam (so the safety-recovery step isn't
   normalized away); `jnp.maximum(constraint, 1e-12)` log-barrier guard.
4. **Euler cluster / MLOps** — async wandb writer (deadlock-safe, drops on full queue); MuJoCo EGL
   headless rendering; JAX memory-preallocation limits; requeue + pickle resume.

> Note: items 1–4 above are a high-level index only. For exact file/line references, current
> status, known-open bugs (e.g. the load-bearing within-window cost-discounting leak), and the
> next-actions list, **read [`handoff/implementation_plan.md`](handoff/implementation_plan.md)** —
> do not treat this summary as authoritative on its own.

