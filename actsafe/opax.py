import jax
import jax.numpy as jnp
from actsafe.actsafe.rssm import ShiftScale
from actsafe.rl import ct_time
from actsafe.rl.types import Prediction

_EPS = 1e-5


def modify_reward(
    trajectory: Prediction,
    distributions: ShiftScale,
    scale: float = 1.0,
    epistemic_scale: float = 1.0,
    stop_grad: bool = True,
    continuous_time: bool = False,
    k_min: float | None = None,
    k_max: float | None = None,
    dt_normalization: bool = False,
) -> tuple[Prediction, ShiftScale]:
    new_rewards = (
        normalized_epistemic_uncertainty(distributions, scale=epistemic_scale) * scale
    )

    if (
        continuous_time
        and dt_normalization
        and k_min is not None
        and k_max is not None
    ):
        # NON-VANILLA, default OFF: normalize the Opax bonus by the hold length
        # ("uncertainty per base step"). Kept only as an opt-in ablation flag
        # (continuous_time.opax_dt_normalization); the dt->max failure it was
        # meant to guard against was never observed with it disabled.
        pseudo_time = trajectory.action[..., -1]
        dt_ratio = ct_time.dt_ratio_from_pseudo_jnp(pseudo_time, k_min, k_max)
        new_rewards = new_rewards / jax.lax.stop_gradient(dt_ratio)

    if stop_grad:
        new_rewards = jax.lax.stop_gradient(new_rewards)
    return Prediction(
        trajectory.action,
        trajectory.next_state,
        new_rewards,
        trajectory.cost,
    ), distributions


def normalized_epistemic_uncertainty(
    distributions: ShiftScale, axis: int = 0, scale: float = 1.0
) -> jnp.ndarray:
    epistemic_uncertainty = distributions.shift.var(axis)
    aleatoric_uncertainty = (distributions.scale**2).mean(axis)
    return 0.5 * jnp.log(
        1.0
        + (
            scale
            * epistemic_uncertainty.mean(-1)
            / (aleatoric_uncertainty.mean(-1) + _EPS)
        )
    )
