from gymnasium.wrappers.compatibility import EnvCompatibility
from omegaconf import DictConfig
import numpy as np
from actsafe.benchmark_suites.utils import get_domain_and_task

from actsafe.rl.types import EnvironmentFactory
from actsafe.rl.wrappers import ChannelFirst


def _control_dt(env, robot_name, default=0.02):
    """Real control timestep for a safe_adaptation_gym env.

    SafeAdaptationGym advances the MuJoCo sim `nstep = _ROBOT_TO_CONTROL_FREQUENCY[robot]`
    physics steps per env.step(), so the control dt = physics_timestep * nstep. Neither
    is exposed as a Gym attribute (the env predates Gymnasium), so we reach into the
    unwrapped env's `robot.sim` MuJoCo model. Falls back to `default` (point's 0.02) if
    the internal layout ever changes.
    """
    try:
        from safe_adaptation_gym.safe_adaptation_gym import (
            _ROBOT_TO_CONTROL_FREQUENCY,
        )

        base = env
        while not hasattr(base, "robot") and hasattr(base, "env"):
            base = base.env
        physics_dt = float(base.robot.sim.model.opt.timestep)
        return physics_dt * _ROBOT_TO_CONTROL_FREQUENCY[robot_name]
    except Exception:  # pragma: no cover - defensive: never crash env construction
        return default


def sample_task(seed):
    easy_tasks = (
        "go_to_goal",
        "push_box",
        "collect",
        "catch_goal",
        "press_buttons",
        "unsupervised",
    )
    task = np.random.RandomState(seed).choice(easy_tasks)
    return task


class SafeAdaptationEnvCompatibility(EnvCompatibility):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset(options=options), {}


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import safe_adaptation_gym

        _, task_cfg = get_domain_and_task(cfg)
        if task_cfg.task is not None:
            task = task_cfg.task
        else:
            task = sample_task(cfg.training.seed)
        env = safe_adaptation_gym.make(
            robot_name=task_cfg.robot_name,
            task_name=task,
            seed=cfg.training.seed,
            rgb_observation=task_cfg.image_observation.enabled,
            render_lidar_and_collision=False,
        )
        env = SafeAdaptationEnvCompatibility(env)
        if (
            task_cfg.image_observation.enabled
            and task_cfg.image_observation.image_format == "channels_first"
        ):
            env = ChannelFirst(env)
        # TASE / continuous-time: SwitchCostWrapper must be OUTERMOST (after ChannelFirst)
        # so the time-to-go channel is not axis-moved and the worker skips its ActionRepeat.
        if cfg.agent.get("continuous_time", {}).get("enabled", False):
            from actsafe.rl.wrappers import SwitchCostWrapper, ConstantSwitchCost

            ct_cfg = cfg.agent.continuous_time

            # Real control dt (point: 0.004 physics * 5 substeps = 0.02 s). This env exposes
            # no Gym dt attr, so probing fell back to 0.01 (2x wrong) before _control_dt.
            dt = _control_dt(env, task_cfg.robot_name)
            tmin = ct_cfg.get("min_time_factor", 1) * dt
            tmax = ct_cfg.get("max_time_factor", 16) * dt
            switch_cost_val = ct_cfg.get("switch_cost", 0.1)
            env = SwitchCostWrapper(
                env,
                t_min=tmin,
                t_max=tmax,
                switch_cost=ConstantSwitchCost(switch_cost_val),
                discounting=cfg.agent.discount,          # within-hold reward discount
                cost_discounting=cfg.agent.safety_discount,  # within-hold cost discount
                dt=dt,                                   # explicit real control dt
            )
        return env

    return make_env  # type: ignore
