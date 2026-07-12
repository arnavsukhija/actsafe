import logging
import numpy as np
from actsafe.actsafe.augmented_lagrangian import AugmentedLagrangianPenalizer
from actsafe.actsafe.dummy_penalizer import DummyPenalizer
from actsafe.actsafe.lbsgd import LBSGDPenalizer
from actsafe.actsafe.safe_actor_critic import SafeModelBasedActorCritic
from actsafe.actsafe.sentiment import bayes


_LOG = logging.getLogger(__name__)


def compute_episode_safety_budget(cfg) -> float:
    """Discounted per-agent-step safety budget handed to the critic constraint.

    Three regimes (audited 2026-07-06, both discounted branches exact):
    - CT/TASE: B = d / (T * (1 - gamma_c)) — chunk-invariant (telescoping),
      dt-schedule-independent; holding longer cannot buy allowance.
    - Discrete discounted: B(R) = (d / episode_steps) / (1 - gamma_c) with
      episode_steps = T / R — per-agent-step cost targets also scale with R,
      so the R cancels and realized allowance is d at every repeat.
    - Undiscounted: B = d.
    """
    continuous_time_enabled = cfg.agent.get("continuous_time", {}).get("enabled", False)
    if continuous_time_enabled and cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps:
        episode_safety_budget = cfg.training.safety_budget / (
            cfg.training.time_limit * (1.0 - cfg.agent.safety_discount)
        )
    elif cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps:
        # Frequency-fair discrete budget: divide by episode length in AGENT steps so the
        # same d enforces the same realized episode cost at every action_repeat.
        episode_steps = cfg.training.time_limit / cfg.training.action_repeat
        episode_safety_budget = (
            cfg.training.safety_budget / episode_steps
        ) / (1.0 - cfg.agent.safety_discount)
    else:
        episode_safety_budget = cfg.training.safety_budget
    episode_safety_budget += cfg.agent.safety_slack
    return episode_safety_budget


def make_actor_critic(
    cfg,
    safe,
    state_dim,
    action_dim,
    key,
    objective_sentiment=bayes,
    constraint_sentiment=bayes,
):

    episode_safety_budget = compute_episode_safety_budget(cfg)
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
        k_min=cfg.agent.continuous_time.get("min_repeat", 1)
        if cfg.agent.continuous_time.enabled
        else None,
        k_max=cfg.agent.continuous_time.get("max_repeat")
        if cfg.agent.continuous_time.enabled
        else None,
        penalizer=penalizer,
        key=key,
        objective_sentiment=objective_sentiment,
        constraint_sentiment=constraint_sentiment,
        actor_entropy_coef=cfg.agent.get("actor_entropy_coef", 0.0),
    )
