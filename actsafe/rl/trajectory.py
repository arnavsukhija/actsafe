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
        o, next_o, a, r, c, cr = zip(*self.transitions)
        # Stack on axis=1 to keep batch dimension first, and time axis second.
        stack = lambda x: np.stack(x, axis=1)
        # cost_realized may be absent (None) for transitions built without it; fall back to
        # the discounted cost so the reported metric equals the discrete-path cost there.
        cr_stacked = stack(c) if cr[0] is None else stack(cr)
        data = TrajectoryData(
            stack(o), stack(next_o), stack(a), stack(r), stack(c), cr_stacked
        )
        return data
