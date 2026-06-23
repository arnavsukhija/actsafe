import jax
import jax.numpy as jnp
from actsafe.actsafe.rssm import ShiftScale
from actsafe.rl.types import Prediction

_EPS = 1e-5


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
        # Normalize Opax reward by the time factor (dt_ratio) to convert the
        # objective from "maximize total uncertainty" to "maximize uncertainty
        # per unit of physical time".  Without this, the actor exploits the
        # dt_ratio dimension: predicting further into the future inflates
        # epistemic uncertainty for free, causing Opax to lock dt_ratio to its
        # maximum and freeze the agent (force=0, dt=max).
        pseudo_time = trajectory.action[..., -1]
        time_for_action = ((tmax - tmin) / 2.0 * pseudo_time) + (tmax + tmin) / 2.0
        dt_ratio = jnp.maximum(jnp.floor(time_for_action / base_dt), 1.0)
        # stop_gradient on dt_ratio is critical: without it Opax could learn to
        # *minimize* dt_ratio to inflate the normalized reward, creating the
        # opposite pathology.
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
