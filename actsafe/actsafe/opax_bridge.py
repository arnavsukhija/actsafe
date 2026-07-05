import jax
import equinox as eqx
from actsafe import opax
from actsafe.actsafe.rssm import ShiftScale, State
from actsafe.actsafe.world_model import WorldModel
from actsafe.rl.types import Policy, Prediction


class OpaxBridge(eqx.Module):
    model: WorldModel
    reward_scale: float = eqx.field(static=True)
    reward_epistemic_scale: float = eqx.field(static=True)
    continuous_time: bool = eqx.field(static=True)
    k_min: float | None = eqx.field(static=True)
    k_max: float | None = eqx.field(static=True)
    dt_normalization: bool = eqx.field(static=True)

    def sample(
        self,
        horizon: int,
        initial_state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, ShiftScale]:
        samples: tuple[Prediction, ShiftScale] = self.model.sample(
            horizon, initial_state, key, policy
        )
        trajectory = Prediction(
            samples[0].action,
            samples[0].next_state,
            samples[0].reward,
            samples[0].cost,
        )
        distributions = samples[1]
        return opax.modify_reward(
            trajectory,
            distributions,
            self.reward_scale,
            self.reward_epistemic_scale,
            continuous_time=self.continuous_time,
            k_min=self.k_min,
            k_max=self.k_max,
            dt_normalization=self.dt_normalization,
        )
