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
- `max_num_timeout: 0` — prevents timeout-based resubmission loop (was 100, caused infinite requeue)
- `'#SBATCH --requeue'` kept — enables preemption requeue (desired behavior, NOT triggered on error/completion)
- `mem_per_cpu: 10240` (10 GB per CPU, 10 CPUs = 100 GB total)
- `additional_parameters: {"gpus": "rtx_4090:1", "account": "ls_krausea"}`
- EGL headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
- JAX compilation cache: `JAX_COMPILATION_CACHE_DIR=/cluster/scratch/${oc.env:USER}/.jax_cache`

**Why max_num_timeout=0:** With 100, every job that hit the 4-hour wall limit was requeued up to 100 times. submitit resubmits on timeout; Slurm `--requeue` handles preemption separately. Setting to 0 means jobs die cleanly after timeout instead of looping.

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
