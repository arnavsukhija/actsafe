import numpy as np
from PIL import Image
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.core import Wrapper
from gymnasium.spaces import Box
from typing import Tuple

from actsafe.rl import ct_time


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        intermediate_states = []
        while current_step < self.repeat and not done:
            obs, reward, terminal, truncated, info = self.env.step(action)
            intermediate_states.append(obs)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            current_step += 1
            done = truncated or terminal
        info["steps"] = current_step
        info["cost"] = total_cost
        info["intermediate_states"] = intermediate_states
        return obs, total_reward, terminal, truncated, info


class ImageObservation(ObservationWrapper):
    def __init__(
        self, env, image_size, image_format="channels_first", *, render_kwargs=None
    ):
        super(ImageObservation, self).__init__(env)
        assert image_format in ["channels_first", "channels_last"]
        size = image_size + (3,) if image_format == "chw" else (3,) + image_size
        self.observation_space = Box(0, 255, size, np.float32)
        if render_kwargs is None:
            render_kwargs = {}
        self._render_kwargs = render_kwargs
        self.image_size = image_size
        self.image_format = image_format

    def observation(self, observation):
        image = self.env.render(**self._render_kwargs)
        return self.preprocess(image)

    def preprocess(self, image):
        image = Image.fromarray(image)
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image, copy=False)
        if self.image_format == "channels_first":
            image = np.moveaxis(image, -1, 0)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image


class ChannelFirst(ObservationWrapper):
    def __init__(self, env):
        super(ChannelFirst, self).__init__(env)
        shape = self.unwrapped.observation_space.shape
        assert isinstance(shape, tuple) and len(shape) == 3
        self.observation_space = Box(0, 255, (shape[2], shape[0], shape[1]), np.float32)

    def observation(self, observation):
        image = np.moveaxis(observation, -1, 0)
        return image


## Wrapper for Time-Adaptive Actsafe

class SwitchCost:
    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        raise NotImplementedError

class ConstantSwitchCost(SwitchCost):
    def __init__(self, cost: float):
        self.cost = cost

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.cost


class SwitchCostWrapper(Wrapper):
    """Executes the agent's (action, pseudo-time) pair as a held action.

    The hold length is expressed in REPEAT UNITS (k base control steps,
    k in [min_repeat, max_repeat]); physical seconds appear only in the
    info['dt'] debug field. Episode time accounting is in base steps.
    """

    def __init__(self,
                 env: gymnasium.Env,
                 min_repeat: int,
                 max_repeat: int,
                 switch_cost: SwitchCost = ConstantSwitchCost(1.0),
                 discounting: float = 1.0,
                 cost_discounting: float | None = None,
                 dt: float | None = None):
        super().__init__(env)
        assert 1 <= min_repeat <= max_repeat, "Expects 1 <= min_repeat <= max_repeat."
        self.switch_cost = switch_cost
        self.k_min = int(min_repeat)
        self.k_max = int(max_repeat)
        self.discounting = discounting
        # Cost discounted within the hold by the safety discount (defaults to reward discount).
        self.cost_discounting = (
            discounting if cost_discounting is None else cost_discounting
        )
        def _get_attr(e, name, default=None):
            try:
                return e.get_wrapper_attr(name)
            except (AttributeError, KeyError):
                return getattr(e, name, default)

        # Physical control dt: used ONLY for the info['dt'] debug field.
        # Prefer the explicit control dt passed by the factory; fall back to attr probing.
        if dt is not None:
            self.dt = dt
        else:
            self.dt = _get_attr(self.env, 'dt')
            if self.dt is None:
                self.dt = _get_attr(self.env, 'control_timestep', lambda: 0.01)()

        max_steps = _get_attr(self.env, '_max_episode_steps', 1000)
        time_limit = _get_attr(self.env, 'time_limit')
        if time_limit is not None:
            max_steps = time_limit

        # Episode clock counts UP: elapsed base steps since reset (0 -> step_horizon).
        self.step_horizon = int(max_steps)
        self.steps_elapsed = 0
        # Augment spaces
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        if isinstance(obs_space, Box):
            # The appended clock channel is 255 * elapsed/horizon in [0, 255].
            if len(obs_space.shape) == 1:
                low = np.append(obs_space.low, 0.0)
                high = np.append(obs_space.high, 255.0)
                self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)
            elif len(obs_space.shape) == 3:
                low = np.concatenate([obs_space.low, np.zeros_like(obs_space.low[:1])], axis=0)
                high = np.concatenate([obs_space.high, np.full_like(obs_space.high[:1], 255.0)], axis=0)
                self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)
            
        if isinstance(act_space, Box):
            low = np.append(act_space.low, -1.0)
            high = np.append(act_space.high, 1.0)
            self.action_space = Box(low=low, high=high, dtype=act_space.dtype)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and appends the elapsed-time clock (0 at reset)
        to the initial observation.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self.steps_elapsed = 0
        return self._augment_observation(obs), info

    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        # The clock is stored as 255 * elapsed/horizon so it survives the replay
        # buffer's uint8 storage like a pixel channel (raw step counts overflow
        # uint8; raw fractions quantize to ~3 values). Consumers recover the
        # fraction as channel/255 (i.e. +0.5 after the standard preprocess).
        clock = 255.0 * self.steps_elapsed / self.step_horizon
        if len(obs.shape) == 1:
            return np.concatenate([obs, np.array([clock], dtype=obs.dtype)])
        time_channel = np.full_like(obs[:1], clock)
        return np.concatenate([obs, time_channel], axis=0)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Performs the action in the environment, held for num_repetitions base steps.
        """
        # separate action components for execution on env
        u, pseudo_time_for_action = action[:-1], action[-1]
        # The agent's requested hold: affine map of pseudo-time into [k_min, k_max],
        # quantized to the nearest integer number of base steps (ties up).
        num_repetitions = int(
            ct_time.dt_ratio_from_pseudo(pseudo_time_for_action, self.k_min, self.k_max)
        )
        # Never request past the episode horizon (the budget is defined over exactly
        # step_horizon base steps); always execute at least one step. The clock below
        # advances by the steps that ACTUALLY executed, so early termination mid-hold
        # cannot desynchronize the accounting.
        steps_remaining = self.step_horizon - self.steps_elapsed
        num_repetitions = min(num_repetitions, max(steps_remaining, 1))

        done = False
        truncated = False
        total_reward = 0.0
        total_reward_realized = 0.0
        total_cost = 0.0
        total_cost_realized = 0.0
        current_step = 0
        info = {"steps": 0}
        intermediate_states = []
        # Raw per-base-step sequences of the hold (undiscounted, no switch cost),
        # zero-padded to k_max: the openloop world model's micro-step targets.
        # Padding beyond info['steps'] is zeros; consumers mask by exposure.
        cost_steps = np.zeros(self.k_max, dtype=np.float32)
        reward_steps = np.zeros(self.k_max, dtype=np.float32)

        obs = None
        while current_step < num_repetitions and not (done or truncated):
            obs, reward, done, truncated, step_info = self.env.step(u)
            intermediate_states.append(obs)

            # Discount reward and cost within the hold (chunk-invariant macro-MDP); raw
            # cost summed separately for the d=25 metric. See handoff for the derivation.
            step_cost = step_info.get('cost', 0.0)
            total_reward += (self.discounting ** current_step) * reward
            total_reward_realized += reward
            total_cost += (self.cost_discounting ** current_step) * step_cost
            total_cost_realized += step_cost
            cost_steps[current_step] = step_cost
            reward_steps[current_step] = reward
            
            # Update info with latest from step_info (excluding specific accumulation keys)
            for k, v in step_info.items():
                if k not in ['cost', 'steps']:
                    info[k] = v
                    
            current_step += 1
            
        # penalize reward with switch cost
        total_reward = total_reward - self.switch_cost(None, u)

        # Advance the elapsed-steps clock by what ACTUALLY executed (the hold can
        # end early on inner-env termination/truncation).
        self.steps_elapsed += current_step

        # Check truncation due to running out of time
        if self.steps_elapsed >= self.step_horizon:
            truncated = True
            self.steps_elapsed = self.step_horizon

        augmented_obs = self._augment_observation(obs)

        info['steps'] = current_step
        info['cost'] = total_cost                    # discounted-within-hold -> critic
        info['cost_realized'] = total_cost_realized  # raw physical sum -> d=25 metric
        # Raw undiscounted task reward, WITHOUT the switch-cost penalty: the true task
        # performance metric. `reward` (returned below) is what the agent optimizes.
        info['reward_realized'] = total_reward_realized
        info['dt'] = current_step * self.dt          # physical seconds, debug only
        info['intermediate_states'] = intermediate_states
        info['cost_steps'] = cost_steps              # raw per-base-step costs, k_max-padded
        info['reward_steps'] = reward_steps          # raw per-base-step rewards, k_max-padded
        
        return augmented_obs, float(total_reward), done, truncated, info

