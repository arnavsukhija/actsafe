"""Fork-audit invariants for the safety-budget math (2026-07-06 audit).

These encode the audited claims as regression tests:
1. TASE budget is chunk-invariant and dt-schedule-independent.
2. Discrete budget cancels the action repeat: realized allowance is d at
   every R (budget-filling cannot explain a cost-vs-repeat slope).
3. SwitchCostWrapper's within-hold cost discounting composed with gamma_c^k
   across decisions telescopes to exact per-base-step discounting for ANY
   hold schedule (holding longer cannot buy allowance).
"""

import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box
from omegaconf import OmegaConf

from actsafe.actsafe.make_actor_critic import compute_episode_safety_budget
from actsafe.rl import ct_time
from actsafe.rl.wrappers import ConstantSwitchCost, SwitchCostWrapper


def _cfg(
    safety_budget=25.0,
    time_limit=1000,
    action_repeat=1,
    safety_discount=0.99,
    ct_enabled=False,
    safety_slack=0.0,
):
    return OmegaConf.create(
        {
            "training": {
                "safety_budget": safety_budget,
                "time_limit": time_limit,
                "action_repeat": action_repeat,
            },
            "agent": {
                "safety_discount": safety_discount,
                "safety_slack": safety_slack,
                "continuous_time": {"enabled": ct_enabled},
            },
        }
    )


def test_ct_budget_is_dt_independent():
    # B = d / (T * (1 - gamma_c)) = 2.5 for the standard PointGoal numbers,
    # regardless of action_repeat (and of any hold schedule, by test below).
    for action_repeat in (1, 2, 8):
        budget = compute_episode_safety_budget(
            _cfg(ct_enabled=True, action_repeat=action_repeat)
        )
        assert budget == pytest.approx(25.0 / (1000 * 0.01))


def test_discrete_budget_realized_allowance_is_d_at_every_repeat():
    # A constant per-base-step cost rate that exactly exhausts d over the
    # episode must sit exactly AT the budget at every repeat: the critic's
    # per-agent-step target sums R base costs, V ~= R*c_bar/(1-gamma_c),
    # and B(R) = (d*R/T)/(1-gamma_c) -> V/B == 1, R cancels.
    d, T, gamma_c = 25.0, 1000, 0.99
    c_bar = d / T  # per-base-step rate that exhausts the physical budget
    for repeat in (1, 2, 4, 8, 16):
        budget = compute_episode_safety_budget(_cfg(action_repeat=repeat))
        value = repeat * c_bar / (1.0 - gamma_c)
        assert value / budget == pytest.approx(1.0)


def test_undiscounted_budget_passthrough():
    budget = compute_episode_safety_budget(
        _cfg(safety_discount=1.0, safety_slack=0.5)
    )
    assert budget == pytest.approx(25.5)


class _ScriptedCostEnv(gymnasium.Env):
    """Emits a prescribed per-base-step cost sequence; never terminates."""

    def __init__(self, costs, max_steps):
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = Box(-1.0, 1.0, (2,), np.float32)
        self._max_episode_steps = max_steps
        self._costs = np.asarray(costs, dtype=np.float64)
        self.base_steps = 0
        self.dt = 0.02

    def reset(self, *args, **kwargs):
        self.base_steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        cost = float(self._costs[self.base_steps])
        self.base_steps += 1
        return np.zeros(3, np.float32), 0.0, False, False, {"cost": cost}


@pytest.mark.parametrize(
    "holds",
    [
        [1] * 37,
        [3, 16, 1, 7, 16, 2, 5, 16],  # mixed schedule, final hold clipped
        [16] * 5,
    ],
)
def test_wrapper_cost_accounting_is_chunk_invariant(holds):
    # sum_j gamma_c^{t_j} * info_j['cost'] == sum_t gamma_c^t * c_t for any
    # hold schedule, including the horizon-clipped last hold. This is the
    # telescoping identity that makes the TASE budget non-exploitable.
    gamma_c = 0.99
    horizon = 37
    rng = np.random.default_rng(0)
    costs = (rng.random(horizon) < 0.3).astype(np.float64)  # sparse 0/1 hazards
    env = SwitchCostWrapper(
        _ScriptedCostEnv(costs, horizon),
        min_repeat=1,
        max_repeat=16,
        switch_cost=ConstantSwitchCost(0.0),
        discounting=1.0,
        cost_discounting=gamma_c,
    )
    env.reset()
    discounted, raw, elapsed = 0.0, 0.0, 0
    truncated = False
    for k in holds:
        if truncated:
            break
        pseudo = ct_time.pseudo_from_dt_ratio(float(k), 1, 16)
        _, _, _, truncated, info = env.step(np.array([0.0, 0.0, pseudo]))
        discounted += gamma_c**elapsed * info["cost"]
        raw += info["cost_realized"]
        elapsed += info["steps"]
    assert elapsed == horizon  # schedule consumes the horizon exactly
    per_base_step = float(np.sum(gamma_c ** np.arange(horizon) * costs))
    assert discounted == pytest.approx(per_base_step, rel=1e-9)
    assert raw == pytest.approx(float(costs.sum()))
