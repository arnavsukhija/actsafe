---
name: project-cluster-infra
description: ETH Euler cluster Slurm/submitit configuration state and fixes for ActSafe training jobs
metadata: 
  node_type: memory
  type: project
  originSessionId: 9686672c-25cf-4aa9-82a2-e74562b76208
---

# Euler Cluster Infrastructure State

## Current slurm.yaml State
File: `actsafe/configs/hydra/launcher/slurm.yaml`

Key settings:
- `timeout_min: 240` (4-hour wall time)
- `max_num_timeout: 100` — CORRECTED 2026-07-03: `git log -S` shows this has been 100 since the file
  was created; an earlier version of this note claimed it was set to 0, which was never actually
  committed. 100 means Slurm DOES resubmit on timeout via `'#SBATCH --requeue'` — this is the
  mechanism behind the 2026-07-03 TASE silent-crash diagnosis (see `implementation_plan.md`): jobs
  that hit the 4h wall time get resubmitted, and the resubmitted attempt is where crashes were landing.
- `'#SBATCH --requeue'` kept — enables preemption requeue (desired behavior, NOT triggered on error/completion)
- `mem_per_cpu: 10240` (10 GB per CPU, 10 CPUs = 100 GB total)
- `additional_parameters: {"gpus": "rtx_4090:1", "account": "ls_krausea"}`
- EGL headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
- JAX compilation cache: `JAX_COMPILATION_CACHE_DIR=/cluster/scratch/${oc.env:USER}/.jax_cache`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` — RESTORED 2026-07-03 as
  low-risk memory hygiene. Added in `c2352c6` ("stop jax from preallocating all memory"), then
  accidentally dropped in `9b088d8` (an unrelated feature commit). **CORRECTION (2026-07-03):** this was
  initially hypothesized as the cause of the TASE grid's silent crashes, but that was WRONG — the real,
  reproduced root cause is a variable-length-episode `np.stack` bug in `actsafe/rl/epoch_summary.py`
  (see `implementation_plan.md`). These XLA vars are kept because they're sensible, not because they
  fixed the crash. The AR study ran fine on the same GPU pool without them.
- `JAX_TRACEBACK_FILTERING=off` — added 2026-07-03, pure diagnostic aid (uncut JAX tracebacks if a
  catchable exception does occur).

**max_num_timeout=100 (kept):** every job that hits the 4-hour wall limit is requeued via submitit;
Slurm `--requeue` handles preemption separately. (An earlier version of this note claimed it was/should
be 0; the committed file has always been 100 per `git log -S`.)

## mujoco Version Fix
File: `pyproject.toml`

Added `mujoco = ">=3.1.3,<3.9.0"` constraint.

**Why:** mujoco 3.9.0 removed `flex_bandwidth` from MjModel Python binding. dm-control 1.0.41 still expects this field in its index.py. Running with mujoco 3.9.0 causes:
```
AttributeError: 'MjModel' object has no attribute 'flex_bandwidth'
```
Pinning `<3.9.0` fixes this. Must also run `pip install "mujoco<3.9.0"` in the active venv on the cluster.

## Standard Training Command (discrete CartPole baseline)
```bash
python train_actsafe.py -m \
  +experiment=safe_sparse_cartpole \
  +hardware=4090_rtx \
  hydra/launcher=slurm \
  +wandb.project=actsafe-ct-cartpole \
  training.seed=0,1,2
```
Note: NO `agent.safety_discount` override — use default 0.99.
