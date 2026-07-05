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
