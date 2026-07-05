import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box

from actsafe.rl import ct_time
from actsafe.rl.wrappers import ConstantSwitchCost, SwitchCostWrapper


@pytest.mark.parametrize("k_min,k_max,base_dt", [(1, 16, 0.02), (1, 8, 0.01), (2, 50, 0.002)])
def test_repeat_units_map_matches_legacy_seconds_map(k_min, k_max, base_dt):
    pseudo = np.linspace(-1.0, 1.0, 4001)
    # Legacy: pseudo -> seconds in [k_min*base_dt, k_max*base_dt] -> /base_dt -> floor.
    t_min, t_max = k_min * base_dt, k_max * base_dt
    seconds = ((t_max - t_min) / 2.0 * pseudo) + (t_max + t_min) / 2.0
    legacy = np.maximum(np.floor(seconds / base_dt), 1.0)
    direct = ct_time.dt_ratio_from_pseudo(pseudo, k_min, k_max)
    np.testing.assert_array_equal(legacy, direct)


def test_ste_forward_matches_numpy_and_backward_is_affine():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    pseudo = jnp.linspace(-1.0, 1.0, 101)
    forward = np.asarray(ct_time.ste_dt_ratio(pseudo, 1.0, 16.0))
    np.testing.assert_array_equal(
        forward, ct_time.dt_ratio_from_pseudo(np.asarray(pseudo), 1.0, 16.0)
    )
    grad = jax.grad(lambda p: ct_time.ste_dt_ratio(p, 1.0, 16.0))(0.3)
    assert np.isclose(float(grad), (16.0 - 1.0) / 2.0)


def test_pseudo_roundtrip():
    for k in range(1, 17):
        pseudo = ct_time.pseudo_from_dt_ratio(float(k), 1.0, 16.0)
        assert int(ct_time.dt_ratio_from_pseudo(pseudo, 1.0, 16.0)) == k


def test_buffer_dt_coverage_metrics():
    k_min, k_max = 1.0, 16.0
    holds_ep0 = [1, 1, 8, 16]
    holds_ep1 = [2, 4]
    pseudo = np.zeros((3, 4))
    pseudo[0, :] = [ct_time.pseudo_from_dt_ratio(k + 0.5, k_min, k_max) for k in holds_ep0]
    pseudo[1, :2] = [ct_time.pseudo_from_dt_ratio(k + 0.5, k_min, k_max) for k in holds_ep1]
    cost = np.zeros((3, 4))
    cost[0, 2] = 1.0  # hazard at decision 2 of episode 0
    lengths = np.array([4, 2, 0])  # episode 2 is an empty slot

    m = ct_time.buffer_dt_coverage(pseudo, cost, lengths, k_min, k_max)
    assert m["train/ct/buffer/mean_dt"] == pytest.approx(np.mean([1, 1, 8, 16, 2, 4]))
    assert m["train/ct/buffer/frac_dt_1"] == pytest.approx(2 / 6)
    assert m["train/ct/buffer/frac_dt_max"] == pytest.approx(1 / 6)
    # Near-hazard (<=1 decision away) in episode 0: decisions 1..3 -> dts {1, 8, 16}.
    assert m["train/ct/buffer/near_hazard_count"] == 3.0
    assert m["train/ct/buffer/near_hazard_distinct_dt"] == 3.0


class _CountingEnv(gymnasium.Env):
    """Deterministic stub: reward 1 and cost 0.5 per base step, never terminates."""

    def __init__(self, max_steps=100):
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = Box(-1.0, 1.0, (2,), np.float32)
        self._max_episode_steps = max_steps
        self.base_steps = 0
        self.dt = 0.02

    def reset(self, *args, **kwargs):
        self.base_steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.base_steps += 1
        return np.zeros(3, np.float32), 1.0, False, False, {"cost": 0.5}


def _make_wrapped(k_max=16, max_steps=100):
    env = _CountingEnv(max_steps)
    return env, SwitchCostWrapper(
        env,
        min_repeat=1,
        max_repeat=k_max,
        switch_cost=ConstantSwitchCost(0.0),
        discounting=1.0,
    )


def test_wrapper_executes_requested_repeats_and_counts_up():
    base, env = _make_wrapped()
    obs, _ = env.reset()
    assert obs[-1] == 0.0  # elapsed clock starts at zero
    # pseudo-time for k=7: any p with floor(affine(p)) == 7.
    pseudo = ct_time.pseudo_from_dt_ratio(7.5, 1, 16)
    obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0, pseudo]))
    assert info["steps"] == 7
    assert base.base_steps == 7
    # Clock channel = 255 * elapsed/horizon (uint8-safe fraction encoding).
    assert obs[-1] == pytest.approx(255.0 * 7 / 100)
    assert reward == pytest.approx(7.0)  # undiscounted sum, zero switch cost
    assert info["cost_realized"] == pytest.approx(3.5)
    assert not (done or truncated)


def test_wrapper_truncates_exactly_at_horizon_without_overshoot():
    base, env = _make_wrapped(k_max=16, max_steps=20)
    env.reset()
    pseudo_max = 1.0  # request k=16 every time
    total = 0
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(np.array([0.0, 0.0, pseudo_max]))
        total += info["steps"]
        assert total <= 20
    assert total == 20  # horizon consumed exactly: 16 + clamped 4
    assert base.base_steps == 20


def test_wrapper_clock_advances_by_executed_steps_on_early_truncation():
    class _EarlyTruncEnv(_CountingEnv):
        def step(self, action):
            obs, r, done, trunc, info = super().step(action)
            # Inner env truncates itself after 5 base steps.
            return obs, r, done, self.base_steps >= 5, info

    base = _EarlyTruncEnv(100)
    env = SwitchCostWrapper(
        base, min_repeat=1, max_repeat=16, switch_cost=ConstantSwitchCost(0.0)
    )
    env.reset()
    obs, _, _, truncated, info = env.step(np.array([0.0, 0.0, 1.0]))  # request 16
    assert info["steps"] == 5  # only 5 executed
    # Clock reflects EXECUTED steps (5), not the request (16).
    assert obs[-1] == pytest.approx(255.0 * 5 / 100)
    assert truncated
