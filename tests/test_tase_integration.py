"""End-to-end smoke test for the TASE (continuous-time) agent plumbing.

Exercises the full (latent, elapsed-time) agent-state path on a stub pixel env:
SwitchCostWrapper -> ActSafe.policy() (time scalar from the clock channel) ->
replay buffer -> update_model() (time appended to posterior states) ->
task + OPAX actor-critic updates (state_dim + 1) -> imagination rollouts with
the analytic time recurrence.
"""

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from actsafe.actsafe.actsafe import ActSafe
from actsafe.rl.trajectory import TrajectoryData
from actsafe.rl.wrappers import ConstantSwitchCost, SwitchCostWrapper
from tests import make_test_config


class _PixelEnv(gymnasium.Env):
    def __init__(self, max_steps=64):
        self.observation_space = Box(0, 255, (3, 64, 64), np.float32)
        self.action_space = Box(-1.0, 1.0, (2,), np.float32)
        self._max_episode_steps = max_steps
        self._steps = 0
        self.dt = 0.02
        self._rs = np.random.RandomState(0)

    def reset(self, *args, **kwargs):
        self._steps = 0
        return self._obs(), {}

    def _obs(self):
        return self._rs.randint(0, 255, (3, 64, 64)).astype(np.float32)

    def step(self, action):
        self._steps += 1
        cost = 1.0 if self._steps % 7 == 0 else 0.0
        return self._obs(), 0.1, False, False, {"cost": cost}


def test_tase_agent_end_to_end_smoke():
    cfg = make_test_config(
        [
            "training.safe=true",
            "training.time_limit=64",
            "training.action_repeat=1",
            "training.parallel_envs=1",
            "agent.model.stochastic_size=8",
            "agent.model.deterministic_size=16",
            "agent.model.hidden_size=32",
            "agent.model.continuous_time=true",
            "agent.sentiment.ensemble_size=2",
            "agent.sentiment.constraint_pessimism=0.1",
            "agent.plan_horizon=5",
            "agent.update_steps=1",
            "agent.replay_buffer.batch_size=2",
            "agent.replay_buffer.sequence_length=4",
            "agent.continuous_time.enabled=true",
            "agent.continuous_time.min_repeat=1",
            "agent.continuous_time.max_repeat=8",
            "agent.exploration_strategy=opax",
            "agent.exploration_steps=1000000",
            "agent.offline_steps=0",
        ]
    )
    env = SwitchCostWrapper(
        _PixelEnv(max_steps=64),
        min_repeat=1,
        max_repeat=8,
        switch_cost=ConstantSwitchCost(0.01),
    )
    agent = ActSafe(env.observation_space, env.action_space, cfg)

    # Roll one full episode through the real policy path.
    obs, _ = env.reset()
    observations, next_observations, actions, rewards, costs = [], [], [], [], []
    truncated = False
    sim_steps = 0
    while not truncated:
        action = agent(obs[None], train=False)[0]
        next_obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        next_observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        costs.append(info["cost"])
        sim_steps += info["steps"]
        obs = next_obs
    assert sim_steps == 64  # horizon consumed exactly

    trajectory = TrajectoryData(
        np.asarray(observations)[None],
        np.asarray(next_observations)[None],
        np.asarray(actions)[None],
        np.asarray(rewards)[None],
        np.asarray(costs)[None],
    )
    agent.observe(trajectory)
    agent.observe_transition(sim_steps=sim_steps)
    assert not agent.replay_buffer.empty

    # Model + task actor-critic + OPAX actor-critic updates: exercises the
    # +1 state dim and the imagination-side time recurrence end to end.
    agent.update()
    metrics = {k: v.result.mean for k, v in agent.metrics_monitor.metrics.items()}
    assert "agent/model/loss" in metrics
    assert any("/opax/" in k for k in metrics), sorted(metrics)
    assert np.isfinite(metrics["agent/model/loss"])

    # report() must include the buffer dt-coverage diagnostics.
    from actsafe.rl.epoch_summary import EpochSummary

    report = agent.report(EpochSummary(), epoch=0, step=64)
    assert "train/ct/buffer/mean_dt" in report.metrics
    assert report.metrics["train/ct/buffer/frac_dt_1"] <= 1.0
