import jax
from omegaconf import DictConfig

from actsafe.actsafe.opax_bridge import OpaxBridge
from actsafe.actsafe.make_actor_critic import make_actor_critic
from actsafe.actsafe.sentiment import identity, make_sentiment
from actsafe.rl.types import Model, Policy


class Exploration:
    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        return {}

    def get_policy(self) -> Policy:
        raise NotImplementedError("Must be implemented by subclass")


def make_exploration(
    config: DictConfig, action_dim: int, key: jax.Array
) -> Exploration:
    if config.agent.exploration_strategy == "opax":
        return OpaxExploration(config, action_dim, key)
    elif config.agent.exploration_strategy == "uniform":
        ct_cfg = config.agent.get("continuous_time", {})
        return UniformExploration(
            action_dim, dt_pseudo_dim=ct_cfg.get("enabled", False)
        )
    else:
        raise NotImplementedError("Unknown exploration strategy")


class OpaxExploration(Exploration):
    def __init__(
        self,
        config: DictConfig,
        action_dim: int,
        key: jax.Array,
    ):
        ct_cfg = config.agent.get("continuous_time", {})
        self.continuous_time = ct_cfg.get("enabled", False)
        # CT: agent state = (latent, elapsed-time fraction) -> +1 input dim
        # (must match the task actor-critic and WorldModel.sample()).
        state_dim = (
            config.agent.model.stochastic_size
            + config.agent.model.deterministic_size
            + (1 if self.continuous_time else 0)
        )
        self.actor_critic = make_actor_critic(
            config,
            config.training.safe,
            state_dim,
            action_dim,
            key,
            objective_sentiment=identity,
            constraint_sentiment=make_sentiment(
                config.agent.sentiment.constraint_pessimism
            ),
        )
        self.reward_scale = config.agent.exploration_reward_scale
        self.epistemic_scale = config.agent.exploration_epistemic_scale

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        model = OpaxBridge(
            model,
            self.reward_scale,
            self.epistemic_scale,
        )
        outs = self.actor_critic.update(model, initial_states, key)
        outs = {f"{_append_opax(k)}": v for k, v in outs.items()}
        return outs

    def get_policy(self) -> Policy:
        return self.actor_critic.actor.act


def _append_opax(string):
    parts = string.split("/")
    parts.insert(2, "opax")
    return "/".join(parts)


class UniformExploration(Exploration):
    def __init__(self, action_dim: int, dt_pseudo_dim: bool = False):
        self.action_dim = action_dim
        if dt_pseudo_dim:
            # CT: the last action dim is a pseudo-time in [-1, 1]. Sampling it
            # from upstream's [0, 1) covers only the upper half of
            # [min_repeat, max_repeat] (e.g. only k in [9, 16] at max_repeat=16)
            # — the offline-coverage bug diagnosed in code_review.md §3. Motor
            # dims deliberately stay at upstream's [0, 1) for comparability
            # with every existing baseline run.
            def policy(_, key: jax.Array) -> jax.Array:
                motor_key, dt_key = jax.random.split(key)
                action = jax.random.uniform(motor_key, (action_dim,))
                dt_pseudo = jax.random.uniform(dt_key, (), minval=-1.0, maxval=1.0)
                return action.at[-1].set(dt_pseudo)

            self.policy = policy
        else:
            self.policy = lambda _, key: jax.random.uniform(key, (self.action_dim,))

    def get_policy(self) -> Policy:
        return self.policy
