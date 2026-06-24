# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ActSafe is a model-based RL research implementation for safe exploration with active learning (ICLR 2025). The core idea: use a learned world model (RSSM) to plan safe trajectories while actively reducing uncertainty. Written in JAX + Equinox.

This fork extends ActSafe toward a **continuous-time / control-frequency safety** research story
(targeting ICLR 2027).

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
   straight-through estimator; `stop_gradient` on `dt_ratio` so the actor can't game the discount
   to hide future cost by stretching `dt`.
3. **Safety & LBSGD** — world-model `sample()` shape-bug fix; action-repeat-aware safety-budget
   scaling; the LBSGD fallback step scaled downstream of Adam (so the safety-recovery step isn't
   normalized away); `jnp.maximum(constraint, 1e-12)` log-barrier guard.
4. **Euler cluster / MLOps** — async wandb writer (deadlock-safe, drops on full queue); MuJoCo EGL
   headless rendering; JAX memory-preallocation limits; requeue + pickle resume.

> Note: items 1–4 above are a high-level index only. For exact file/line references, current
> status, known-open bugs (e.g. the load-bearing within-window cost-discounting leak), and the
> next-actions list, **read [`handoff/implementation_plan.md`](handoff/implementation_plan.md)** —
> do not treat this summary as authoritative on its own.

