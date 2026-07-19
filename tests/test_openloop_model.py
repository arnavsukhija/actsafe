"""Openloop world-model training path (2026-07-19 revamp, commit 2).

The RSSM becomes a BASE-STEP model (motor action only) and a hold is the
k-fold composition of micro-predictions; per-micro-step reward/cost decodes
are supervised by the raw per-base-step targets and composed analytically
into whole-hold aggregates. The flow path must stay byte-identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from actsafe.common.learner import Learner
from actsafe.actsafe.rssm import Features, State
from actsafe.actsafe.world_model import WorldModel, variational_step

ACTION_DIM = 3  # 2 motor dims + 1 dt (pseudo-time) dim
K_MAX = 4
HORIZON = 5
BATCH = 2
ENSEMBLE = 3
GAMMA, GAMMA_C = 0.97, 0.99


def make_model(dynamics="openloop"):
    return WorldModel(
        image_shape=(4, 64, 64),  # 3 pixels + clock channel
        action_dim=ACTION_DIM,
        deterministic_size=32,
        stochastic_size=8,
        hidden_size=32,
        ensemble_size=ENSEMBLE,
        initialization_scale=1.0,
        num_rewards=1,
        continuous_time=True,
        k_min=1.0,
        k_max=float(K_MAX),
        horizon_steps=100.0,
        dynamics=dynamics,
        hold_discount=GAMMA,
        hold_cost_discount=GAMMA_C,
        key=jax.random.PRNGKey(0),
    )


def _batch(key):
    keys = jax.random.split(key, 5)
    features = Features(
        jax.random.uniform(keys[0], (BATCH, HORIZON, 4, 64, 64)) - 0.5,
        jax.random.normal(keys[1], (BATCH, HORIZON, 1)),
        jax.random.uniform(keys[2], (BATCH, HORIZON)),
        jnp.zeros((BATCH, HORIZON, 1)),
    )
    actions = jax.random.uniform(
        keys[3], (BATCH, HORIZON, ACTION_DIM), minval=-1.0, maxval=1.0
    )
    exposure = jax.random.randint(
        keys[4], (BATCH, HORIZON), 1, K_MAX + 1
    ).astype(jnp.float32)
    return features, actions, exposure


def test_openloop_cell_is_base_step_motor_only():
    model = make_model()
    # The cell's action input excludes the dt dim: k is structural.
    assert model.cell.priors.encoder.weight.shape[-1] == 8 + (ACTION_DIM - 1)
    # Flow keeps the full action input.
    flow = make_model("flow")
    assert flow.cell.priors.encoder.weight.shape[-1] == 8 + ACTION_DIM


def test_openloop_inference_shapes_and_micro():
    model = make_model()
    features, actions, exposure = _batch(jax.random.PRNGKey(1))
    key = jax.random.PRNGKey(2)
    result = jax.vmap(lambda f, a, e: model(f, a, key, exposure=e))(
        features, actions, exposure
    )
    assert result.reward_cost.shape == (BATCH, HORIZON, 2)
    assert result.micro_reward_cost.shape == (BATCH, HORIZON, K_MAX, 2)
    assert result.state.stochastic.shape == (BATCH, HORIZON, 8)
    assert result.posteriors.shift.shape == (BATCH, HORIZON, 8)
    assert result.priors.shift.shape == (BATCH, HORIZON, 8)


def test_openloop_aggregates_compose_micro_steps():
    # reward_cost must be exactly the within-hold discounted, exposure-masked
    # composition of the per-micro-step decodes (the wrapper identity, in
    # latent space).
    model = make_model()
    features, actions, exposure = _batch(jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(4)
    result = jax.vmap(lambda f, a, e: model(f, a, key, exposure=e))(
        features, actions, exposure
    )
    micro = np.asarray(result.micro_reward_cost)
    steps = np.arange(K_MAX)
    mask = steps[None, None, :] < np.asarray(exposure)[..., None]
    reward = (micro[..., 0] * GAMMA**steps * mask).sum(-1)
    cost = (micro[..., 1] * GAMMA_C**steps * mask).sum(-1)
    np.testing.assert_allclose(
        np.asarray(result.reward_cost[..., 0]), reward, rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(result.reward_cost[..., 1]), cost, rtol=1e-5)


def test_first_micro_step_matches_single_prior_step():
    # Composition sanity: micro-step 0 of the unroll is exactly one base-step
    # prior prediction (same member, same state, same action).
    model = make_model()
    state = State(
        jax.random.normal(jax.random.PRNGKey(5), (8,)),
        jax.random.normal(jax.random.PRNGKey(6), (32,)),
    )
    u = jnp.array([0.3, -0.2])
    prior_member = jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
        model.cell.priors,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    micro_states, micro_priors = model._unroll_hold_member(
        prior_member, state, u, jax.random.PRNGKey(7)
    )
    direct_shift_scale, direct_det = prior_member(state, u)
    np.testing.assert_allclose(
        np.asarray(micro_priors.shift[0]), np.asarray(direct_shift_scale.shift),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(micro_states.deterministic[0]), np.asarray(direct_det), rtol=1e-5
    )


def test_ensemble_unroll_shapes():
    model = make_model()
    state = State(jnp.zeros((8,)), jnp.zeros((32,)))
    micro_states, micro_priors = model._unroll_hold(
        state, jnp.zeros((2,)), jax.random.PRNGKey(8)
    )
    assert micro_states.stochastic.shape == (ENSEMBLE, K_MAX, 8)
    assert micro_priors.shift.shape == (ENSEMBLE, K_MAX, 8)


def test_variational_step_openloop_trains():
    model = make_model()
    learner = Learner(model, {"lr": 1e-3})
    features, actions, exposure = _batch(jax.random.PRNGKey(9))
    cost_steps = jax.random.uniform(jax.random.PRNGKey(10), (BATCH, HORIZON, K_MAX))
    reward_steps = jax.random.normal(jax.random.PRNGKey(11), (BATCH, HORIZON, K_MAX))
    (new_model, _), (loss, aux) = variational_step(
        features,
        actions,
        model,
        learner,
        learner.state,
        jax.random.PRNGKey(12),
        reward_steps=reward_steps,
        cost_steps=cost_steps,
        exposure=exposure,
    )
    assert jnp.isfinite(loss)
    assert "openloop_agg_residual" in aux
    # The decoder must actually receive gradient from the micro-step loss.
    delta = jax.tree_util.tree_map(
        lambda a, b: float(jnp.abs(a - b).max()),
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(
                lambda x: x,
                new_model.reward_cost_decoder.layers[0].weight,
            )
        ),
        jax.tree_util.tree_leaves(model.reward_cost_decoder.layers[0].weight),
    )
    assert np.max(np.asarray(delta)) > 0.0


def test_flow_path_unchanged():
    # The flow branch must run exactly as before (no exposure required).
    model = make_model("flow")
    features, actions, _ = _batch(jax.random.PRNGKey(13))
    key = jax.random.PRNGKey(14)
    result = jax.vmap(lambda f, a: model(f, a, key))(features, actions)
    assert result.reward_cost.shape == (BATCH, HORIZON, 2)
    assert result.micro_reward_cost is None


def test_openloop_sample_keeps_prediction_contract():
    from actsafe.rl import ct_time

    model = make_model()
    key = jax.random.PRNGKey(15)
    # Callable policy (actor-critic path).
    policy = lambda state, k: jnp.zeros((ACTION_DIM,))
    prediction, priors = model.sample(7, model.cell.init, key, policy)
    assert prediction.reward.shape == (ENSEMBLE, 7, 1)
    assert prediction.cost.shape == (ENSEMBLE, 7)
    assert prediction.next_state.shape == (7, 8 + 32 + 1)  # latent + time
    assert priors.shift.shape == (ENSEMBLE, 7, 8)
    # Array policy (evaluate_model / imagined_vs_realized path) + exact time
    # recurrence by the EXECUTED holds.
    holds = [2, 4, 1]
    pseudos = [ct_time.pseudo_from_dt_ratio(k, 1.0, float(K_MAX)) for k in holds]
    actions = jnp.zeros((3, ACTION_DIM)).at[:, -1].set(jnp.asarray(pseudos))
    prediction, _ = model.sample(
        3, model.cell.init, key, actions, initial_time=jnp.asarray(0.1)
    )
    times = prediction.next_state[:, -1]
    expected = 0.1 + jnp.cumsum(jnp.asarray(holds, jnp.float32)) / 100.0
    assert jnp.allclose(times, jnp.minimum(expected, 1.0), atol=1e-6)


def test_openloop_imagined_cost_has_dt_gradient():
    # The STE-masked composition gives the dt head a direct path into the
    # imagined hold cost: backward is the boundary micro-step's discounted
    # decode (the structural k-slope).
    model = make_model()
    key = jax.random.PRNGKey(16)

    def summed_cost(dt_value):
        actions = jnp.zeros((4, ACTION_DIM)).at[:, -1].set(dt_value)
        prediction, _ = model.sample(4, model.cell.init, key, actions)
        return prediction.cost.sum()

    grad = jax.grad(summed_cost)(0.1)
    assert jnp.isfinite(grad)
    assert jnp.abs(grad) > 0.0


def test_openloop_infer_state():
    model = make_model()
    observation = jax.random.uniform(jax.random.PRNGKey(17), (4, 64, 64)) - 0.5
    action = jnp.array([0.2, -0.4, 0.5])
    state = model.infer_state(
        model.cell.init, observation, action, jax.random.PRNGKey(18)
    )
    assert state.stochastic.shape == (8,)
    assert state.deterministic.shape == (32,)


def test_openloop_diagnostics_run():
    from actsafe.actsafe.world_model import (
        imagined_vs_realized_cost,
        k_slope_diagnostics,
        openloop_micro_calibration,
    )

    model = make_model()
    features, actions, exposure = _batch(jax.random.PRNGKey(19))
    cost_steps = jax.random.uniform(jax.random.PRNGKey(20), (BATCH, HORIZON, K_MAX))
    key = jax.random.PRNGKey(21)
    safety_critic = lambda s: s.sum()

    k_slope = k_slope_diagnostics(
        model, safety_critic, features, actions, exposure, GAMMA_C, key
    )
    for name in ("perceived_cost", "perceived_qc", "realized_cost"):
        assert jnp.isfinite(k_slope[f"agent/ct/kslope/{name}"])

    micro = openloop_micro_calibration(
        model, features, actions, exposure, cost_steps, key
    )
    assert "agent/openloop/micro_gap_i_0" in micro
    assert "agent/openloop/micro_gap_i_3" in micro
    assert "agent/openloop/micro_gap_i_7" not in micro  # k_max == 4
    # Every position executes micro-step 0, so its count is the full batch.
    assert int(micro["agent/openloop/micro_count_i_0"]) == BATCH * HORIZON
    assert jnp.isfinite(micro["agent/openloop/agg_residual"])

    imagination = imagined_vs_realized_cost(
        model, features, actions, exposure, GAMMA_C, key
    )
    assert jnp.isfinite(imagination["agent/imagination/cost_return_gap"])


def test_openloop_rejects_discrete_or_multireward():
    with pytest.raises(AssertionError):
        WorldModel(
            image_shape=(3, 64, 64),
            action_dim=ACTION_DIM,
            deterministic_size=8,
            stochastic_size=4,
            hidden_size=8,
            ensemble_size=1,
            initialization_scale=1.0,
            num_rewards=1,
            continuous_time=False,
            dynamics="openloop",
            key=jax.random.PRNGKey(0),
        )
