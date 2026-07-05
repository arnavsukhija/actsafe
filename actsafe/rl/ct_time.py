"""Single source of truth for the pseudo-time -> hold-length (dt_ratio) mapping.

The agent's dt head outputs a pseudo-time p in [-1, 1]. It maps affinely to a
hold length in REPEAT UNITS (k in [k_min, k_max] base control steps) and is
floored to an integer, exactly matching SwitchCostWrapper's execution. Physical
seconds never enter the agent; dt = k * base_dt exists only for reporting.

The jax variants import jax lazily so that environment worker processes (which
import actsafe.rl.wrappers) stay jax-free.
"""

import numpy as np


def dt_raw_from_pseudo(pseudo, k_min: float, k_max: float):
    """Affine map [-1, 1] -> [k_min, k_max]. Module-agnostic (numpy or jax)."""
    return (k_max - k_min) / 2.0 * pseudo + (k_max + k_min) / 2.0


def dt_ratio_from_pseudo(pseudo, k_min: float, k_max: float):
    """Executed hold length in base steps (numpy)."""
    return np.maximum(np.floor(dt_raw_from_pseudo(pseudo, k_min, k_max)), 1.0)


def dt_ratio_from_pseudo_jnp(pseudo, k_min: float, k_max: float):
    """Executed hold length in base steps (jax)."""
    import jax.numpy as jnp

    return jnp.maximum(jnp.floor(dt_raw_from_pseudo(pseudo, k_min, k_max)), 1.0)


def ste_dt_ratio(pseudo, k_min: float, k_max: float):
    """Floor straight-through estimator: forward = the executed integer hold,
    backward = identity through the affine map. The internal stop_gradient IS
    the straight-through mechanism (floor has zero derivative a.e.); it is not
    a gradient block — the discount and cost paths stay fully differentiable
    w.r.t. the dt head.
    """
    import jax
    import jax.numpy as jnp

    dt_raw = dt_raw_from_pseudo(pseudo, k_min, k_max)
    return dt_raw + jax.lax.stop_gradient(
        jnp.maximum(jnp.floor(dt_raw), 1.0) - dt_raw
    )


def pseudo_from_dt_ratio(dt_ratio, k_min: float, k_max: float):
    """Inverse of the affine map (before flooring)."""
    return (2.0 * dt_ratio - (k_max + k_min)) / (k_max - k_min)


def buffer_dt_coverage(
    pseudo: np.ndarray,
    cost: np.ndarray,
    lengths: np.ndarray,
    k_min: float,
    k_max: float,
    prefix: str = "train/ct/buffer/",
) -> dict[str, float]:
    """dt-coverage statistics of the replay buffer's EXECUTED holds.

    Answers, per epoch and without pickle forensics, whether the world model's
    training data has the same-state multi-dt contrast that a dt-sensitive
    cost head c̄(s, u, t) needs — in particular NEAR HAZARDS, where it matters.

    pseudo:  [episodes, max_len] dt-head column of stored actions.
    cost:    [episodes, max_len] per-decision cost.
    lengths: [episodes] true (unpadded) episode lengths in decisions.
    """
    valid_eps = np.where(lengths > 0)[0]
    if len(valid_eps) == 0:
        return {}
    all_dt, near_dt = [], []
    for ep in valid_eps:
        length = int(lengths[ep])
        dt = dt_ratio_from_pseudo(pseudo[ep, :length], k_min, k_max)
        all_dt.append(dt)
        hazard_idx = np.where(cost[ep, :length] > 0)[0]
        if len(hazard_idx) > 0:
            idx = np.arange(length)
            proximity = np.abs(idx[:, None] - hazard_idx[None, :]).min(1)
            near_dt.append(dt[proximity <= 1])
    dt = np.concatenate(all_dt)
    metrics = {
        f"{prefix}mean_dt": float(dt.mean()),
        f"{prefix}std_dt": float(dt.std()),
        f"{prefix}frac_dt_1": float(np.mean(dt == 1.0)),
        f"{prefix}frac_dt_2_4": float(np.mean((dt >= 2.0) & (dt <= 4.0))),
        f"{prefix}frac_dt_5_8": float(np.mean((dt >= 5.0) & (dt <= 8.0))),
        f"{prefix}frac_dt_9_plus": float(np.mean(dt >= 9.0)),
        f"{prefix}frac_dt_max": float(np.mean(dt == float(k_max))),
    }
    if near_dt:
        near = np.concatenate(near_dt)
        metrics |= {
            f"{prefix}near_hazard_count": float(len(near)),
            f"{prefix}near_hazard_mean_dt": float(near.mean()),
            f"{prefix}near_hazard_std_dt": float(near.std()),
            f"{prefix}near_hazard_distinct_dt": float(len(np.unique(near))),
        }
    else:
        metrics[f"{prefix}near_hazard_count"] = 0.0
    return metrics
