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
    dt_normalization: bool = True,
) -> tuple[Prediction, ShiftScale]:
    new_rewards = (
        normalized_epistemic_uncertainty(distributions, scale=epistemic_scale) * scale
    )

    if (
        continuous_time
        and dt_normalization
        and tmin is not None
        and tmax is not None
        and base_dt is not None
    ):
        # Normalize Opax reward by the time factor (dt_ratio) -> "uncertainty per unit
        # physical time". Without it the actor inflates uncertainty by predicting far
        # ahead, locking dt_ratio to max and freezing the agent (force=0, dt=max).
        # Ablatable via continuous_time.opax_dt_normalization.
        pseudo_time = trajectory.action[..., -1]
        time_for_action = ((tmax - tmin) / 2.0 * pseudo_time) + (tmax + tmin) / 2.0
        # floor matches SwitchCostWrapper's executed-hold quantization exactly.
        dt_ratio = jnp.maximum(jnp.floor(time_for_action / base_dt), 1.0)
        # stop_gradient is critical: else Opax minimizes dt_ratio to inflate the reward.
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
