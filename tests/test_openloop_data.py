"""Per-base-step data plumbing for the openloop world model (2026-07-19 revamp).

The SwitchCostWrapper emits raw per-base-step cost/reward sequences
(`info['cost_steps']`/`['reward_steps']`, k_max-padded) and the replay buffer
stores them alongside the per-decision aggregates. These are model-side
supervision targets only — the decision-level SMDP data layout is unchanged.
"""

import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box

from actsafe.actsafe.replay_buffer import ReplayBuffer
from actsafe.rl import ct_time
from actsafe.rl.trajectory import TrajectoryData
from actsafe.rl.wrappers import ConstantSwitchCost, SwitchCostWrapper


class _ScriptedEnv(gymnasium.Env):
    """Emits prescribed per-base-step cost and reward sequences."""

    def __init__(self, costs, rewards, max_steps):
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = Box(-1.0, 1.0, (2,), np.float32)
        self._max_episode_steps = max_steps
        self._costs = np.asarray(costs, dtype=np.float64)
        self._rewards = np.asarray(rewards, dtype=np.float64)
        self.base_steps = 0
        self.dt = 0.02

    def reset(self, *args, **kwargs):
        self.base_steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        cost = float(self._costs[self.base_steps])
        reward = float(self._rewards[self.base_steps])
        self.base_steps += 1
        return np.zeros(3, np.float32), reward, False, False, {"cost": cost}


def test_wrapper_emits_per_step_sequences():
    gamma, gamma_c, switch_cost, k_max = 0.97, 0.99, 0.1, 8
    horizon = 20
    rng = np.random.default_rng(1)
    costs = (rng.random(horizon) < 0.4).astype(np.float64)
    rewards = rng.random(horizon)
    env = SwitchCostWrapper(
        _ScriptedEnv(costs, rewards, horizon),
        min_repeat=1,
        max_repeat=k_max,
        switch_cost=ConstantSwitchCost(switch_cost),
        discounting=gamma,
        cost_discounting=gamma_c,
    )
    env.reset()
    elapsed = 0
    for k in (5, 8, 1, 8):  # last hold horizon-clipped: only 6 steps remain
        pseudo = ct_time.pseudo_from_dt_ratio(float(k), 1, k_max)
        _, reward, _, _, info = env.step(np.array([0.0, 0.0, pseudo]))
        executed = info["steps"]
        cs, rs = info["cost_steps"], info["reward_steps"]
        assert cs.shape == (k_max,) and rs.shape == (k_max,)
        # Raw per-step values in executed positions, zero padding beyond.
        np.testing.assert_allclose(
            cs[:executed], costs[elapsed : elapsed + executed], rtol=1e-6
        )
        np.testing.assert_allclose(
            rs[:executed], rewards[elapsed : elapsed + executed], rtol=1e-6
        )
        assert not np.any(cs[executed:]) and not np.any(rs[executed:])
        # Aggregates are exactly the discounted sums of the step sequences.
        steps = np.arange(k_max)
        assert info["cost"] == pytest.approx(
            float((gamma_c**steps * cs).sum()), rel=1e-6
        )
        assert reward == pytest.approx(
            float((gamma**steps * rs).sum()) - switch_cost, rel=1e-6
        )
        assert info["cost_realized"] == pytest.approx(float(cs.sum()), rel=1e-6)
        assert info["reward_realized"] == pytest.approx(float(rs.sum()), rel=1e-6)
        elapsed += executed
    assert elapsed == horizon


def _make_buffer(step_dim):
    return ReplayBuffer(
        observation_shape=(1, 2, 2),
        action_shape=(3,),
        max_length=10,
        seed=0,
        capacity=3,
        batch_size=1,
        sequence_length=3,
        num_rewards=1,
        step_dim=step_dim,
    )


def _trajectory(length, step_dim, rng):
    obs = rng.integers(0, 255, (1, length, 1, 2, 2)).astype(np.uint8)
    cost_steps = rng.random((1, length, step_dim)).astype(np.float32)
    reward_steps = rng.random((1, length, step_dim)).astype(np.float32)
    return TrajectoryData(
        obs,
        obs,
        rng.random((1, length, 3)).astype(np.float32),
        rng.random((1, length)).astype(np.float32),
        rng.random((1, length)).astype(np.float32),
        exposure=np.full((1, length), 2.0, np.float32),
        cost_steps=cost_steps,
        reward_steps=reward_steps,
    )


def test_buffer_step_array_roundtrip():
    step_dim = 4
    buffer = _make_buffer(step_dim)
    assert buffer.has_step_arrays
    rng = np.random.default_rng(2)
    trajectory = _trajectory(6, step_dim, rng)
    buffer.add(trajectory)
    np.testing.assert_allclose(
        buffer.cost_steps[0, :6], trajectory.cost_steps[0], rtol=1e-6
    )
    assert not np.any(buffer.cost_steps[0, 6:])
    batch = next(buffer.sample(1))
    assert batch.cost_steps.shape == (1, 3, step_dim)
    assert batch.reward_steps.shape == (1, 3, step_dim)
    # Sampled windows must align with the stored episode (find the offset via
    # the sampled action row).
    offset = next(
        t
        for t in range(6 - 3 + 1)
        if np.allclose(buffer.action[0, t], batch.action[0, 0])
    )
    np.testing.assert_allclose(
        batch.cost_steps[0], trajectory.cost_steps[0, offset : offset + 3], rtol=1e-6
    )
    np.testing.assert_allclose(
        batch.reward_steps[0],
        trajectory.reward_steps[0, offset : offset + 3],
        rtol=1e-6,
    )


def test_buffer_without_step_dim_yields_none():
    buffer = _make_buffer(None)
    assert not buffer.has_step_arrays
    rng = np.random.default_rng(3)
    trajectory = _trajectory(6, 4, rng)._replace(cost_steps=None, reward_steps=None)
    buffer.add(trajectory)
    batch = next(buffer.sample(1))
    assert batch.cost_steps is None and batch.reward_steps is None


def test_legacy_pickle_lacks_step_arrays():
    buffer = _make_buffer(4)
    del buffer.cost_steps  # simulate un-pickling a pre-openloop checkpoint
    assert not buffer.has_step_arrays
