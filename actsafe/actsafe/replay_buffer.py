from typing import Iterator
import jax
import numpy as np

from actsafe.common.double_buffer import double_buffer
from actsafe.rl.trajectory import TrajectoryData


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        batch_size: int,
        sequence_length: int,
        num_rewards: int,
    ):
        self.episode_id = 0
        self.max_length = max_length
        self.sequence_length = sequence_length
        self.dtype = np.float32
        self.obs_dtype = np.uint8
        self.observation = np.zeros(
            (
                capacity,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.obs_dtype,
        )
        self.action = np.zeros(
            (
                capacity,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (capacity, max_length, num_rewards),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                capacity,
                max_length,
            ),
            dtype=self.dtype,
        )
        # Track the actual (non-padded) length of each stored episode.
        # Episodes shorter than max_length are zero-padded; this array tells
        # _sample_batch() where the real data ends.
        self.lengths = np.zeros(capacity, dtype=np.int32)
        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        self.batch_size = batch_size

    def add(self, trajectory: TrajectoryData):
        capacity, *_ = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.episode_id + batch_size, capacity)
        episode_slice = slice(self.episode_id, end)
        if trajectory.reward.ndim == 2:
            trajectory = TrajectoryData(
                trajectory.observation,
                trajectory.next_observation,
                trajectory.action,
                trajectory.reward[..., None],
                trajectory.cost,
            )
        # Actual episode length (may be shorter than max_length in continuous time).
        actual_len = min(trajectory.action.shape[1], self.max_length)
        # Zero-pad: clear the slot first, then write only the actual steps.
        # (The buffer is already zero-initialized, but slots may be reused.)
        for data in (self.action, self.reward, self.cost):
            data[episode_slice] = 0
        self.observation[episode_slice] = 0

        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[episode_slice, :actual_len] = val[:batch_size, :actual_len].astype(self.dtype)
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size, :actual_len],
                trajectory.next_observation[:batch_size, actual_len - 1: actual_len],
            ],
            axis=1,
        )
        self.observation[episode_slice, :actual_len + 1] = observation.astype(self.obs_dtype)
        # Record the true length for safe sampling later.
        self.lengths[episode_slice] = actual_len
        self.episode_id = (self.episode_id + batch_size) % capacity
        self._valid_episodes = min(self._valid_episodes + batch_size, capacity)

    def _sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        valid_episodes: int | None = None,
    ):
        if valid_episodes is not None:
            valid_episodes = valid_episodes
        else:
            valid_episodes = self._valid_episodes

        # Only sample from episodes long enough to yield a full sequence.
        valid_ids = np.where(self.lengths[:valid_episodes] > sequence_length)[0]
        if len(valid_ids) == 0:
            # No valid episodes yet (all shorter than sequence_length).
            # Yield a dummy batch of zeros and let the caller retry later.
            return

        while True:
            episode_ids = valid_ids[self.rs.choice(len(valid_ids), size=batch_size)]
            ep_lengths = self.lengths[episode_ids]  # shape: (batch_size,)
            # Per-episode valid start range: [0, ep_len - sequence_length)
            low = np.array([
                self.rs.randint(0, ep_lengths[i] - sequence_length)
                for i in range(batch_size)
            ])
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # bootstrapping)
            a, r, c = [
                x[episode_ids[:, None], timestep_ids[:, :-1]]
                for x in (
                    self.action,
                    self.reward,
                    self.cost,
                )
            ]
            o = self.observation[episode_ids[:, None], timestep_ids]
            o, next_o = o[:, :-1], o[:, 1:]
            yield o, next_o, a, r, c

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        if self.empty:
            return
        batch_gen = self._sample_batch(self.batch_size, self.sequence_length)
        if batch_gen is None:
            # All stored episodes are shorter than sequence_length — cannot train yet.
            return
        iterator = (
            TrajectoryData(
                *next(batch_gen)
            )  # type: ignore
            for _ in range(n_batches)
        )
        if jax.default_backend() == "gpu":
            iterator = double_buffer(iterator)  # type: ignore
        yield from iterator

    @property
    def empty(self):
        return self._valid_episodes == 0 or not np.any(
            self.lengths[:self._valid_episodes] > self.sequence_length
        )
