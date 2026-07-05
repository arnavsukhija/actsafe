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


def make_model(
    continuous_time: bool,
    k_min: float = 1.0,
    k_max: float = 1.0,
    horizon_steps: float = 1.0,
) -> WorldModel:
    # CT observations carry an extra elapsed-time clock channel that the CNN
    # skips (it re-enters the agent state as an exact scalar).
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
        k_min=k_min,
        k_max=k_max,
        horizon_steps=horizon_steps,
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


def test_imagination_time_recurrence_matches_wrapper():
    from actsafe.rl import ct_time

    horizon_steps = 100.0
    model = make_model(True, k_min=1.0, k_max=16.0, horizon_steps=horizon_steps)
    key = jax.random.PRNGKey(3)

    holds = [3, 7, 16, 1, 5]
    # k + 0.5 floors to exactly k under the shared affine map.
    pseudos = [ct_time.pseudo_from_dt_ratio(k + 0.5, 1.0, 16.0) for k in holds]
    actions = jnp.zeros((len(holds), ACTION_DIM)).at[:, -1].set(jnp.asarray(pseudos))

    prediction, _ = model.sample(
        len(holds), model.cell.init, key, actions, initial_time=jnp.asarray(0.1)
    )
    times = prediction.next_state[:, -1]
    expected = 0.1 + jnp.cumsum(jnp.asarray(holds, jnp.float32)) / horizon_steps
    assert jnp.allclose(times, jnp.minimum(expected, 1.0), atol=1e-6)


def test_flat_initial_state_roundtrips_time():
    model = make_model(True, k_min=1.0, k_max=16.0, horizon_steps=100.0)
    key = jax.random.PRNGKey(4)
    state_dim = 32 + 8
    flat = jnp.zeros((state_dim + 1,)).at[-1].set(0.25)
    policy = lambda state, k: jnp.zeros((ACTION_DIM,))
    prediction, _ = model.sample(4, flat, key, policy)
    # dt head 0 -> affine(0) in [1,16] is 8.5 -> floor 8 -> +0.08 per step.
    expected = 0.25 + 0.08 * jnp.arange(1, 5, dtype=jnp.float32)
    assert jnp.allclose(prediction.next_state[:, -1], expected, atol=1e-6)


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
