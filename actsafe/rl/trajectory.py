from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy import typing as npt


class Transition(NamedTuple):
    observation: npt.NDArray[Any]
    next_observation: npt.NDArray[Any]
    action: npt.NDArray[Any]
    reward: npt.NDArray[Any]
    cost: npt.NDArray[Any]
    # Raw undiscounted physical cost of the hold (SwitchCostWrapper's info['cost_realized']).
    # `cost` is the chunk-invariant discounted-within-hold cost the critic/world-model learn;
    # `cost_realized` is the physical episode cost reported against the d budget. Optional so the
    # replay buffer (which reconstructs 5-field TrajectoryData) is unaffected — it stays None there.
    cost_realized: npt.NDArray[Any] | None = None
    # Raw undiscounted task reward of the hold WITHOUT the switch-cost penalty
    # (SwitchCostWrapper's info['reward_realized']). `reward` is the penalized,
    # discounted-within-hold reward the agent optimizes; this one measures true
    # task performance. None on the discrete path (falls back to `reward`).
    reward_realized: npt.NDArray[Any] | None = None
    # Executed base steps of the hold (SwitchCostWrapper's info['steps'];
    # action_repeat on the discrete path). This is the exposure of the
    # factored rate cost head: the world model learns a per-base-step cost
    # rate from (cost_realized, exposure) pairs. None falls back to 1.
    exposure: npt.NDArray[Any] | None = None
    # Raw per-base-step cost/reward sequences of the hold, zero-padded to k_max
    # (SwitchCostWrapper's info['cost_steps']/['reward_steps']; undiscounted, no
    # switch cost). Micro-step targets for the openloop world model; positions
    # beyond `exposure` are padding. None on the discrete path.
    cost_steps: npt.NDArray[Any] | None = None
    reward_steps: npt.NDArray[Any] | None = None


TrajectoryData = Transition


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)
    frames: list[npt.NDArray[np.float32 | np.int8]] = field(default_factory=list)

    def __len__(self):
        return len(self.transitions)

    def as_numpy(self) -> TrajectoryData:
        # Transpose list of tuples to a tuple of lists,
        # this magic is possible since transition is a named tuple.
        # This allows us make lists of observations, actions, rewards, etc.,
        # instead of list of transitions.
        o, next_o, a, r, c, cr, rr, k, cs, rs = zip(*self.transitions)
        # Stack on axis=1 to keep batch dimension first, and time axis second.
        stack = lambda x: np.stack(x, axis=1)
        # cost_realized / reward_realized may be absent (None) for transitions built
        # without them; fall back to the penalized/discounted values so the reported
        # metrics equal the discrete-path ones there.
        cr_stacked = stack(c) if cr[0] is None else stack(cr)
        rr_stacked = stack(r) if rr[0] is None else stack(rr)
        k_stacked = np.ones_like(stack(r)) if k[0] is None else stack(k)
        data = TrajectoryData(
            stack(o),
            stack(next_o),
            stack(a),
            stack(r),
            stack(c),
            cr_stacked,
            rr_stacked,
            k_stacked,
            # Per-step sequences exist only on the SwitchCostWrapper path.
            None if cs[0] is None else stack(cs),
            None if rs[0] is None else stack(rs),
        )
        return data
