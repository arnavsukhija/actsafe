"""Shape and gradient checks for the action-conditioned (oTaCoS) reward/cost decoder.

In continuous-time mode the reward/cost of a hold are transition quantities
r̄(s, u, t) / c̄(s, u, t): the within-hold accumulation is not observable from the
arrival frame alone, so the decoder is conditioned on the action (incl. the dt
head). These tests check both model paths (training inference + imagination) in
both modes, and that imagination now has a direct nonzero d(cost)/d(dt) path.

Run on the cluster before launching a sweep: pytest tests/test_world_model_action_conditioning.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from actsafe.actsafe.rssm import Features
from actsafe.actsafe.world_model import WorldModel


ACTION_DIM = 3  # 2 force dims + 1 dt (pseudo-time) dim
HORIZON = 12
BATCH = 2
ENSEMBLE = 5


def make_model(continuous_time: bool) -> WorldModel:
    # CT observations carry an extra time_to_go channel that the encoder strips.
    channels = 4 if continuous_time else 3
    return WorldModel(
        image_shape=(channels, 64, 64),
        action_dim=ACTION_DIM,
        deterministic_size=32,
        stochastic_size=8,
        hidden_size=32,
        ensemble_size=ENSEMBLE,
        initialization_scale=1.0,
        num_rewards=1,
        continuous_time=continuous_time,
        key=jax.random.PRNGKey(0),
    )


@pytest.mark.parametrize("continuous_time", [True, False])
def test_training_and_imagination_shapes(continuous_time):
    model = make_model(continuous_time)
    channels = 4 if continuous_time else 3
    key = jax.random.PRNGKey(1)

    obs = jnp.zeros((BATCH, HORIZON, channels, 64, 64))
    features = Features(
        obs,
        jnp.zeros((BATCH, HORIZON, 1)),
        jnp.zeros((BATCH, HORIZON)),
        jnp.zeros((BATCH, HORIZON, 1)),
    )
    actions = jnp.zeros((BATCH, HORIZON, ACTION_DIM))
    result = jax.vmap(lambda f, a: model(f, a, key))(features, actions)
    assert result.reward_cost.shape == (BATCH, HORIZON, 2)

    # Imagination with a callable policy (actor-critic path).
    policy = lambda state, k: jnp.zeros((ACTION_DIM,))
    prediction, _ = model.sample(7, model.cell.init, key, policy)
    assert prediction.reward.shape == (ENSEMBLE, 7, 1)
    assert prediction.cost.shape == (ENSEMBLE, 7)

    # Imagination with a raw action array (evaluate_model path).
    prediction, _ = model.sample(7, model.cell.init, key, jnp.zeros((7, ACTION_DIM)))
    assert prediction.cost.shape == (ENSEMBLE, 7)


def test_imagined_cost_has_direct_dt_gradient():
    model = make_model(continuous_time=True)
    key = jax.random.PRNGKey(2)

    def summed_cost(dt_value):
        actions = jnp.zeros((5, ACTION_DIM)).at[:, -1].set(dt_value)
        prediction, _ = model.sample(5, model.cell.init, key, actions)
        return prediction.cost.sum()

    grad = jax.grad(summed_cost)(0.3)
    # With the action-conditioned decoder the dt dim reaches the cost head
    # directly (not only through the KL-trained dynamics), so even an untrained
    # model has a nonzero derivative.
    assert jnp.isfinite(grad)
    assert jnp.abs(grad) > 0.0
