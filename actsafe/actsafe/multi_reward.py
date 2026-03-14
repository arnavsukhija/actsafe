import jax
import equinox as eqx
from actsafe.actsafe.rssm import ShiftScale, State
from actsafe.actsafe.world_model import WorldModel
from actsafe.rl.types import Policy, Prediction


class MultiRewardBridge(eqx.Module):
    model: WorldModel
    reward_index: int = eqx.field(static=True)

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
        trajectory, distributions = samples
        rewards = trajectory.reward[..., self.reward_index]
        trajectory = Prediction(
            samples[0].action,
            samples[0].next_state,
            rewards,
            samples[0].cost,
        )
        return trajectory, distributions
