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
) -> tuple[Prediction, ShiftScale]:
    # NOTE (CT): the bonus is per DECISION, not per base step, and the switch
    # cost never enters this objective — so dt=1 is the rational optimum of
    # exploration under a hold (V_explore(k) ≈ b(k)/(1−γ^k) with a
    # log-squashed bonus). A dt-aware OPAX (switch cost inside the objective,
    # k-sweep disagreement, (state, k) novelty) is a parked study arm; see
    # code_review.md §4. The former opax_dt_normalization flag (divide the
    # bonus by dt) pushed the WRONG way for coverage and was removed.
    new_rewards = (
        normalized_epistemic_uncertainty(distributions, scale=epistemic_scale) * scale
    )
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
