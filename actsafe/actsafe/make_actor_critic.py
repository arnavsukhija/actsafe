import logging
import numpy as np
from actsafe.actsafe.augmented_lagrangian import AugmentedLagrangianPenalizer
from actsafe.actsafe.dummy_penalizer import DummyPenalizer
from actsafe.actsafe.lbsgd import LBSGDPenalizer
from actsafe.actsafe.safe_actor_critic import SafeModelBasedActorCritic
from actsafe.actsafe.sentiment import bayes


_LOG = logging.getLogger(__name__)


def make_actor_critic(
    cfg,
    safe,
    state_dim,
    action_dim,
    key,
    objective_sentiment=bayes,
    constraint_sentiment=bayes,
):

    continuous_time_enabled = cfg.agent.get("continuous_time", {}).get("enabled", False)
    if cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps:
        # safety_budget is the total episode cost limit; convert to discounted value by
        # dividing by time_limit (getting per-step budget) and then the geometric series factor.
        episode_safety_budget = (
            cfg.training.safety_budget / cfg.training.time_limit
        ) / (1.0 - cfg.agent.safety_discount)
    else:
        episode_safety_budget = cfg.training.safety_budget
    episode_safety_budget += cfg.agent.safety_slack
    _LOG.info(f"Episode safety budget: {episode_safety_budget}")
    if safe:
        if cfg.agent.penalizer.name == "lbsgd":
            penalizer = LBSGDPenalizer(
                cfg.agent.penalizer.m_0,
                cfg.agent.penalizer.m_1,
                cfg.agent.penalizer.eta,
                cfg.agent.penalizer.eta_rate,
                cfg.agent.actor_optimizer.lr,
                cfg.agent.penalizer.backup_lr,
            )
        elif cfg.agent.penalizer.name == "lagrangian":
            penalizer = AugmentedLagrangianPenalizer(
                cfg.agent.penalizer.initial_lagrangian,
                cfg.agent.penalizer.initial_multiplier,
                cfg.agent.penalizer.multiplier_factor,
            )
        else:
            raise NotImplementedError
    else:
        penalizer = DummyPenalizer()
    return SafeModelBasedActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_config=cfg.agent.actor,
        critic_config=cfg.agent.critic,
        actor_optimizer_config=cfg.agent.actor_optimizer,
        critic_optimizer_config=cfg.agent.critic_optimizer,
        safety_critic_optimizer_config=cfg.agent.safety_critic_optimizer,
        horizon=cfg.agent.plan_horizon,
        discount=cfg.agent.discount,
        safety_discount=cfg.agent.safety_discount,
        lambda_=cfg.agent.lambda_,
        safety_budget=episode_safety_budget,
        continuous_time=cfg.agent.continuous_time.enabled,
        tmin=cfg.agent.continuous_time.t_min if cfg.agent.continuous_time.enabled else None,
        tmax=cfg.agent.continuous_time.t_max if cfg.agent.continuous_time.enabled else None,
        base_dt=cfg.agent.continuous_time.base_dt if cfg.agent.continuous_time.enabled else None,
        penalizer=penalizer,
        key=key,
        objective_sentiment=objective_sentiment,
        constraint_sentiment=constraint_sentiment,
    )
