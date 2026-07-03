from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
from numpy import typing as npt

from actsafe.rl.trajectory import Trajectory


@dataclass
class EpochSummary:
    _data: list[list[Trajectory]] = field(default_factory=list)
    cost_boundary: float = 25.0
    continuous_time: bool = False
    tmin: float = 0.0
    tmax: float = 0.0
    base_dt: float = 1.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> Tuple[float, float, float]:
        rewards, costs = [], []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                *_, r, c = trajectory.as_numpy()
                rewards.append(r)
                costs.append(c)
        # Stack data from all tasks on the first axis,
        # giving a [#tasks, #episodes, #time, ...] shape.
        # Continuous-time (variable-dt) episodes can have different numbers of
        # transitions, so pad the time axis with zeros before stacking (padding
        # with zero reward/cost past an episode's own length doesn't change its
        # sum, which is all _objective/_feasibility use the time axis for).
        stacked_rewards = _stack_padded(rewards)
        stacked_costs = _stack_padded(costs)
        return (
            _objective(stacked_rewards),
            _objective(stacked_costs),
            _feasibility(stacked_costs, self.cost_boundary),
        )

    @property
    def videos(self):
        all_vids = []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                if len(trajectory.frames) > 0:
                    all_vids.append(trajectory.frames)
        if len(all_vids) == 0:
            return None
        # all_vids[-1] is a list of frames, each frame is (N, H, W, 3)
        # Convert to (T, N, H, W, 3)
        vids = np.asarray(all_vids[-1]) 
        T, N, H, W, C = vids.shape
        
        # Tile N environment videos into a grid
        grid_size = int(np.ceil(np.sqrt(N)))
        grid_h = grid_size
        grid_w = (N + grid_size - 1) // grid_size
        
        # Pad with zeros if N < grid_h * grid_w
        if N < grid_h * grid_w:
            padding = np.zeros((T, grid_h * grid_w - N, H, W, C), dtype=vids.dtype)
            vids = np.concatenate([vids, padding], axis=1)
            
        # Reshape and transpose to (T, grid_h*H, grid_w*W, C)
        vids = vids.reshape(T, grid_h, grid_w, H, W, C)
        vids = vids.transpose(0, 1, 3, 2, 4, 5)
        vids = vids.reshape(T, grid_h * H, grid_w * W, C)
        
        # WandB expects (T, C, H, W)
        return vids.transpose(0, 3, 1, 2)

    def extend(self, samples: List[Trajectory]) -> None:
        self._data.append(samples)

    @property
    def continuous_time_metrics(self) -> dict[str, float]:
        """Compute diagnostic metrics for continuous-time control.

        Returns dt_ratio and force statistics from actual environment actions.
        These reveal whether the Opax time-normalization fix is working:
        - If dt_ratio is always ~max, the agent is still exploiting the curiosity hack.
        - If dt_ratio varies (especially values near 1-2), the fix is effective.
        - If mean |force| is ~0, the agent is still frozen.
        """
        if not self.continuous_time or self.empty:
            return {}

        pseudo_times, forces = [], []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                data = trajectory.as_numpy()
                # action shape: [batch, time, action_dim]. Episodes can have
                # different `time` lengths under variable dt, so flatten each
                # episode's own time axis and pool across episodes instead of
                # concatenating on a fixed time axis (which requires equal
                # lengths and is what previously crashed on TASE runs).
                action = data.action
                pseudo_times.append(action[..., -1].reshape(-1))
                forces.append(action[..., :-1].reshape(-1, action.shape[-1] - 1))

        pseudo_time = np.concatenate(pseudo_times)
        force = np.concatenate(forces, axis=0)

        # Convert pseudo_time to dt_ratio using the same formula as the environment
        time_for_action = ((self.tmax - self.tmin) / 2.0 * pseudo_time) + (self.tmax + self.tmin) / 2.0
        dt_ratio = np.maximum(np.round(time_for_action / self.base_dt), 1.0)

        return {
            "train/ct/mean_dt_ratio": float(np.mean(dt_ratio)),
            "train/ct/std_dt_ratio": float(np.std(dt_ratio)),
            "train/ct/frac_dt_1": float(np.mean(dt_ratio == 1.0)),
            "train/ct/frac_dt_max": float(np.mean(
                dt_ratio == np.round(self.tmax / self.base_dt)
            )),
            "train/ct/mean_abs_force": float(np.mean(np.abs(force))),
            "train/ct/std_force": float(np.std(force)),
        }


def _stack_padded(arrays: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    max_len = max(a.shape[-1] for a in arrays)
    if all(a.shape[-1] == max_len for a in arrays):
        return np.stack(arrays)
    padded = [
        np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, max_len - a.shape[-1])])
        for a in arrays
    ]
    return np.stack(padded)


def _objective(rewards: npt.NDArray[Any]) -> float:
    if rewards.ndim == 3:
        return float(rewards.sum(2).mean())
    elif rewards.ndim == 4:
        return rewards.sum(2).mean((0, 1))
    else:
        raise ValueError(f"Expected 3 or 4 dimensions, got {rewards.ndim} dimensions")


def _feasibility(costs: npt.NDArray[Any], boundary: float) -> float:
    return float((costs.sum(2).mean(1) <= boundary).mean())
