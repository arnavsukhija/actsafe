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
        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __getstate__(self):
        """Prepare the buffer for pickling by only saving used episodes."""
        state = self.__dict__.copy()
        # Save the full capacity information so we can reconstruct upon load
        state["full_capacity"] = self.observation.shape[0]
        state["max_episode_length"] = self.observation.shape[1] - 1
        
        # Only save valid data to drastically reduce pickle size early in training.
        # This prevents saving 16GB of zeros to disk.
        valid = self._valid_episodes
        state["observation"] = self.observation[:valid + 1]
        state["action"] = self.action[:valid]
        state["reward"] = self.reward[:valid]
        state["cost"] = self.cost[:valid]
        return state

    def __setstate__(self, state):
        """Restore the buffer, re-allocating zeros for the unused capacity."""
        if "full_capacity" not in state:
            self.__dict__.update(state)
            return

        # Extract full capacity and max length from the pickled state
        capacity = state.pop("full_capacity")
        max_length = state.pop("max_episode_length")
        
        # Load the dictionary first to initialize other fields
        self.__dict__.update(state)
        
        # Now re-allocate the full-size zero-filled arrays
        obs_shape = self.observation.shape[2:]
        act_shape = self.action.shape[2:]
        num_rewards = self.reward.shape[2]
        
        full_observation = np.zeros((capacity, max_length + 1) + obs_shape, dtype=self.obs_dtype)
        full_action = np.zeros((capacity, max_length) + act_shape, dtype=self.dtype)
        full_reward = np.zeros((capacity, max_length, num_rewards), dtype=self.dtype)
        full_cost = np.zeros((capacity, max_length), dtype=self.dtype)
        
        # Copy the pickled data back into the beginning...
        valid = self._valid_episodes
        full_observation[:valid + 1] = self.observation
        full_action[:valid] = self.action
        full_reward[:valid] = self.reward
        full_cost[:valid] = self.cost
        
        self.observation = full_observation
        self.action = full_action
        self.reward = full_reward
        self.cost = full_cost

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
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[episode_slice] = val[:batch_size].astype(self.dtype)
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size],
                trajectory.next_observation[:batch_size, -1:],
            ],
            axis=1,
        )
        self.observation[episode_slice] = observation.astype(self.obs_dtype)
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
        time_limit = self.observation.shape[1]
        assert time_limit > sequence_length
        while True:
            low = self.rs.choice(time_limit - sequence_length - 1, batch_size)
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
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
        iterator = (
            TrajectoryData(
                *next(self._sample_batch(self.batch_size, self.sequence_length))
            )  # type: ignore
            for _ in range(n_batches)
        )
        if jax.default_backend() == "gpu":
            iterator = double_buffer(iterator)  # type: ignore
        yield from iterator

    @property
    def empty(self):
        return self._valid_episodes == 0
