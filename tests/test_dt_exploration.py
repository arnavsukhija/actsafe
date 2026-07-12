"""dt-exploration coverage guarantees (code_review.md Part A items 3-4).

The offline-coverage bug: UniformExploration sampled ALL action dims in
[0, 1), so the dt head's pseudo-time never went below 0 and the offline phase
only ever executed the upper half of [min_repeat, max_repeat]. These tests pin
the fixed behavior: with dt_pseudo_dim=True the dt dim spans the full [-1, 1]
(motor dims stay at upstream's [0, 1)), and a full-range pseudo-time reaches
EVERY integer hold length under the nearest-int mapping.
"""

import jax
import numpy as np

from actsafe.actsafe.exploration import UniformExploration
from actsafe.rl import ct_time


def _sample_actions(policy, n=512, action_dim=3):
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return np.asarray(jax.vmap(lambda key: policy(None, key))(keys))


def test_uniform_exploration_dt_dim_spans_full_pseudo_range():
    policy = UniformExploration(action_dim=3, dt_pseudo_dim=True).get_policy()
    actions = _sample_actions(policy)
    dt_pseudo = actions[:, -1]
    assert dt_pseudo.min() >= -1.0 and dt_pseudo.max() <= 1.0
    # Both halves of the range must actually be visited (the bug produced
    # [0, 1) only — no negative pseudo-times, so no holds below the midpoint).
    assert dt_pseudo.min() < -0.9
    assert dt_pseudo.max() > 0.9
    # Motor dims stay at upstream's [0, 1) for baseline comparability.
    motor = actions[:, :-1]
    assert motor.min() >= 0.0 and motor.max() < 1.0


def test_uniform_exploration_default_is_upstream_unchanged():
    policy = UniformExploration(action_dim=3).get_policy()
    actions = _sample_actions(policy)
    assert actions.min() >= 0.0 and actions.max() < 1.0


def test_full_pseudo_range_reaches_every_hold_length():
    k_min, k_max = 1, 16
    pseudo = np.linspace(-1.0, 1.0, 4001)
    holds = ct_time.dt_ratio_from_pseudo(pseudo, k_min, k_max)
    assert set(np.unique(holds)) == set(float(k) for k in range(k_min, k_max + 1))


def test_upstream_offline_range_covered_only_upper_half():
    # Documents the bug being fixed: [0, 1) pseudo-times reach only k >= 9
    # at max_repeat=16 (dt_raw in [8.5, 16) -> nearest int in [9, 16]).
    k_min, k_max = 1, 16
    pseudo = np.linspace(0.0, 0.999, 1000)
    holds = ct_time.dt_ratio_from_pseudo(pseudo, k_min, k_max)
    assert holds.min() >= 9.0
