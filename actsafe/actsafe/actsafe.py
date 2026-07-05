from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig

from actsafe.common.learner import Learner
from actsafe.actsafe import rssm
from actsafe.actsafe.exploration import UniformExploration, make_exploration
from actsafe.actsafe.make_actor_critic import make_actor_critic
from actsafe.actsafe.multi_reward import MultiRewardBridge
from actsafe.actsafe.replay_buffer import ReplayBuffer
from actsafe.actsafe.sentiment import make_sentiment
from actsafe.actsafe.world_model import WorldModel, evaluate_model, variational_step
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.metrics import MetricsMonitor
from actsafe.rl.trajectory import TrajectoryData, Transition
from actsafe.rl.types import FloatArray, Report
from actsafe.rl.utils import PRNGSequence, Until, add_to_buffer


@eqx.filter_jit
def policy(policy_fn, model, prev_state, observation, key):
    def per_env_policy(prev_state, observation, key):
        model_key, policy_key = jax.random.split(key)
        current_rssm_state = model.infer_state(
            prev_state.rssm_state, observation, prev_state.prev_action, model_key
        )
        flat = current_rssm_state.flatten()
        if model.continuous_time:
            # Agent state = (latent, elapsed-time fraction). The clock channel is
            # spatially constant; preprocess maps its 255*frac value to frac - 0.5.
            time_fraction = observation[-1, 0, 0] + 0.5
            flat = jnp.concatenate([flat, time_fraction[None].astype(flat.dtype)])
        action = policy_fn(flat, policy_key)
        return action, AgentState(current_rssm_state, action)

    observation = preprocess(observation)
    return jax.vmap(per_env_policy)(
        prev_state, observation, jax.random.split(key, observation.shape[0])
    )


class AgentState(NamedTuple):
    rssm_state: rssm.State
    prev_action: jax.Array

    @classmethod
    def init(cls, batch_size: int, cell: rssm.RSSM, action_dim: int) -> "AgentState":
        rssm_state = cell.init
        rssm_state = jax.tree_map(
            lambda x: jnp.repeat(x[None], batch_size, 0), rssm_state
        )
        prev_action = jnp.zeros((batch_size, action_dim))
        self = cls(rssm_state, prev_action)
        return self


class ActSafe:
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        self.config = config
        num_rewards = 2 if self.config.agent.unsupervised else 1
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            sequence_length=config.agent.replay_buffer.sequence_length,
            batch_size=config.agent.replay_buffer.batch_size,
            capacity=config.agent.replay_buffer.capacity,
            num_rewards=num_rewards,
        )
        self.prng = PRNGSequence(config.training.seed)
        action_dim = int(np.prod(action_space.shape))
        assert len(observation_space.shape) == 3
        ct_cfg = config.agent.get("continuous_time", {})
        ct_enabled = ct_cfg.get("enabled", False)
        self.model = WorldModel(
            image_shape=observation_space.shape,
            action_dim=action_dim,
            key=next(self.prng),
            ensemble_size=config.agent.sentiment.ensemble_size,
            initialization_scale=config.agent.sentiment.model_initialization_scale,
            num_rewards=num_rewards,
            k_min=float(ct_cfg.get("min_repeat", 1)) if ct_enabled else 1.0,
            k_max=float(ct_cfg.get("max_repeat", 1)) if ct_enabled else 1.0,
            horizon_steps=float(config.training.time_limit) if ct_enabled else 1.0,
            **config.agent.model,
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        # CT: agent state = (latent, elapsed-time fraction) -> +1 input dim for
        # the actor and both critics (full-MDP time under episode truncation).
        state_dim = (
            config.agent.model.stochastic_size
            + config.agent.model.deterministic_size
            + (1 if ct_enabled else 0)
        )
        self.actor_critic = make_actor_critic(
            config,
            config.training.safe,
            state_dim,
            action_dim,
            next(self.prng),
            make_sentiment(self.config.agent.sentiment.objective_optimism),
            make_sentiment(self.config.agent.sentiment.constraint_pessimism),
        )
        self.exploration = make_exploration(
            config,
            action_dim,
            next(self.prng),
        )
        self.offline = UniformExploration(action_dim)
        self.state = AgentState.init(
            config.training.parallel_envs, self.model.cell, action_dim
        )
        # All training schedule counters are advanced in observe_transition()
        # by the actual base simulation steps consumed per env step, so that
        # every threshold (train_every, exploration_steps, offline_steps) means
        # identical amounts of simulated time in both discrete and continuous
        # control modes.
        self._train_sim_steps = 0
        self._train_every = config.agent.train_every
        self.should_explore = Until(config.agent.exploration_steps, steps=1)
        self.should_collect_offline = Until(config.agent.offline_steps, steps=1)
        # TASE dt-exploration: during the warm-up window, resample the dt head of the
        # executed action uniformly over [-1, 1] (pseudo-time), independent of the
        # policy, while keeping the policy's force dims. This decouples the world
        # model's (state, u, t) training coverage from the acting policy's dt
        # preferences — the oTaCoS-style data assumption — so that same-state
        # multi-dt contrasts exist in the buffer even if the policy's dt collapses.
        # Applies only to real-environment rollouts (this class only acts in the
        # real env); imagination and all loss formulations are untouched.
        self._dt_exploration_uniform = bool(
            ct_cfg.get("enabled", False)
            and ct_cfg.get("dt_exploration", "policy") == "uniform"
        )
        dt_exploration_steps = ct_cfg.get("dt_exploration_steps", None)
        if dt_exploration_steps is None:
            dt_exploration_steps = config.agent.exploration_steps
        self.should_explore_dt = Until(dt_exploration_steps, steps=1)
        learn_model_steps = (
            config.agent.learn_model_steps
            if config.agent.learn_model_steps is not None
            else float("inf")
        )
        self.learn_model = Until(learn_model_steps, steps=1)
        self.metrics_monitor = MetricsMonitor()
        self.zero_shot = False

    def __call__(
        self,
        observation: FloatArray,
        train: bool = False,
    ) -> FloatArray:
        # Fire a training update every _train_every accumulated sim steps.
        # The remainder is carried forward so no sim steps are "lost" across calls.
        should_train_now = self._train_sim_steps >= self._train_every
        if should_train_now:
            self._train_sim_steps -= self._train_every
        if train and should_train_now and not self.replay_buffer.empty:
            self.update()
        if self.should_collect_offline():
            policy_fn = self.offline.get_policy()
        else:
            policy_fn = (
                self.exploration.get_policy()
                if self.should_explore()
                else self.actor_critic.actor.act
            )
        # Counters are ticked in observe_transition() with actual sim steps.
        actions, self.state = policy(
            policy_fn,
            self.model,
            self.state,
            observation,
            next(self.prng),
        )
        # getattr: agents un-pickled from pre-dt-exploration checkpoints lack these.
        if getattr(self, "_dt_exploration_uniform", False) and self.should_explore_dt():
            # Full-range pseudo-time (the offline UniformExploration policy samples
            # [0, 1), which only covers the upper half of [min_repeat, max_repeat]).
            dt_pseudo = jax.random.uniform(
                next(self.prng), (actions.shape[0],), minval=-1.0, maxval=1.0
            )
            actions = actions.at[:, -1].set(dt_pseudo)
            # Keep prev_action consistent with the action actually executed, so the
            # RSSM filter conditions on the true (state, u, t) transition.
            self.state = AgentState(self.state.rssm_state, actions)
        return np.asarray(actions)

    def observe(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.replay_buffer,
            trajectory,
            self.config.training.scale_reward,
        )
        self.state = jax.tree_map(lambda x: jnp.zeros_like(x), self.state)

    def observe_transition(self, transition: Transition | None = None, *, sim_steps: int = 0) -> None:
        # Advance all training schedule counters by the actual number of base
        # simulation steps consumed for this transition.
        # In discrete mode sim_steps == action_repeat × num_active_envs.
        # In continuous-time mode it includes the num_repetitions from
        # SwitchCostWrapper, making all thresholds mean the same simulated time
        # regardless of control frequency.
        if sim_steps > 0:
            self._train_sim_steps += sim_steps
            self.should_explore.count += sim_steps
            self.should_collect_offline.count += sim_steps
            # getattr: agents un-pickled from pre-dt-exploration checkpoints lack this.
            if (dt_until := getattr(self, "should_explore_dt", None)) is not None:
                dt_until.count += sim_steps
            self.learn_model.count += sim_steps

    def update(self):
        total_steps = self.config.agent.update_steps
        if (
            not self.should_explore()
            and self.config.agent.unsupervised
            and not self.learn_model()
            and not self.zero_shot
        ):
            total_steps = self.config.agent.zero_shot_steps
        for batch in self.replay_buffer.sample(total_steps):
            batch = TrajectoryData(
                batch.observation,
                batch.next_observation,
                batch.action,
                batch.reward * self.config.agent.reward_scale,
                batch.cost,
            )
            inferred_rssm_states = self.update_model(batch)
            initial_states = inferred_rssm_states.reshape(
                -1, inferred_rssm_states.shape[-1]
            )
            sample_size = min(128, initial_states.shape[0])
            indices = jax.random.choice(
                next(self.prng), initial_states.shape[0], (sample_size,), replace=False
            )
            initial_states = initial_states[indices]
            if self.should_explore():
                if not self.config.agent.unsupervised:
                    outs = self.actor_critic.update(
                        MultiRewardBridge(self.model, 0),
                        initial_states,
                        next(self.prng),
                    )
                else:
                    outs = {}
                exploration_outs = self.exploration.update(
                    self.model, initial_states, next(self.prng)
                )
                outs.update(exploration_outs)
            else:
                if self.config.agent.unsupervised:
                    if self.learn_model():
                        index = 0
                    else:
                        self.zero_shot = True
                        index = self.config.agent.reward_index
                else:
                    index = -1
                outs = self.actor_critic.update(
                    MultiRewardBridge(self.model, index),
                    initial_states,
                    next(self.prng),
                )
            for k, v in outs.items():
                self.metrics_monitor[k] = v

    def update_model(self, batch: TrajectoryData) -> jax.Array:
        features, actions = _prepare_features(batch)
        inference_only = self.config.agent.unsupervised and not self.learn_model()
        (self.model, self.model_learner.state), (loss, rest) = variational_step(
            features,
            actions,
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
            self.config.agent.beta,
            self.config.agent.free_nats,
            self.config.agent.kl_mix,
            inference_only=inference_only,
        )
        self.metrics_monitor["agent/model/loss"] = float(loss.mean())
        self.metrics_monitor["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.metrics_monitor["agent/model/kl"] = float(rest["kl_loss"].mean())
        flat_states = rest["states"].flatten()
        if self.model.continuous_time:
            # Arrival-time fraction of each posterior state, read from the clock
            # channel of the same (batch, step) positions (preprocess maps the
            # 255*frac channel to frac - 0.5); imagination rollouts start from
            # (latent, time) so the critics see episode time consistently.
            time_fraction = features.observation[:, :, -1, 0, 0] + 0.5
            flat_states = jnp.concatenate(
                [flat_states, time_fraction[..., None].astype(flat_states.dtype)], -1
            )
        return flat_states

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        if self.config.agent.evaluate_model:
            batch = next(self.replay_buffer.sample(1))
            features, actions = _prepare_features(batch)
            video = evaluate_model(self.model, features, actions, next(self.prng))
            return Report(metrics=metrics, videos={"agent/model/prediction": video})
        else:
            return Report(metrics=metrics)


@jax.jit
def _prepare_features(batch: TrajectoryData) -> tuple[rssm.Features, jax.Array]:
    terminals = jnp.zeros_like(batch.reward)
    features = rssm.Features(
        jnp.asarray(preprocess(batch.next_observation)),
        jnp.asarray(batch.reward),
        jnp.asarray(batch.cost),
        jnp.asarray(terminals),
    )
    actions = jnp.asarray(batch.action)
    return features, actions


def preprocess(image):
    return image / 255.0 - 0.5
