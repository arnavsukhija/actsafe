import numpy as np
from PIL import Image
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.core import Wrapper
from gymnasium.spaces import Box
from typing import Tuple


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

    def __init__(self,
                 env: gymnasium.Env, 
                 t_min: float, 
                 t_max: float, 
                 switch_cost: SwitchCost = ConstantSwitchCost(1.0),
                 discounting: float = 1.0):
        super().__init__(env)
        self.switch_cost = switch_cost
        self.tmin = t_min
        self.tmax = t_max
        self.discounting = discounting
        self.dt = getattr(self.env.unwrapped, 'dt', 0.01)
        if not hasattr(self.env.unwrapped, 'dt'):
            self.dt = getattr(self.env.unwrapped, 'control_timestep', lambda: 0.01)()
        max_steps = getattr(self.env, '_max_episode_steps', 1000)
        if hasattr(self.env.unwrapped, 'time_limit'):
            max_steps = self.env.unwrapped.time_limit
        self.time_horizon = max_steps * self.dt
        self.time_to_go = self.time_horizon        
        # Augment spaces
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        if isinstance(obs_space, Box):
            low = np.append(obs_space.low, 0.0)
            high = np.append(obs_space.high, np.inf)
            self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)
        if isinstance(act_space, Box):
            low = np.append(act_space.low, -1.0)
            high = np.append(act_space.high, 1.0)
            self.action_space = Box(low=low, high=high, dtype=act_space.dtype)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and appends the time to the initial observation.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self.time_to_go = self.time_horizon
        
        state = np.concatenate([obs, np.array([self.time_to_go], dtype=obs.dtype)])
        return state, info

    def compute_time(self,
                      pseudo_time: float,
                      t_lower: float,
                      t_upper: float) -> float:
        
        time_for_action = ((t_upper - t_lower) / 2.0 * pseudo_time) + (t_upper + t_lower) / 2.0
        return np.floor(time_for_action / self.dt) * self.dt
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Performs the action in the environment and repeats for time_for_action times
        """
        # separate action components for execution on env
        u, pseudo_time_for_action = action[:-1], action[-1]
        time_for_action = self.compute_time(pseudo_time_for_action, self.tmin, self.tmax)
        time_for_action = min(time_for_action, self.time_to_go)
        
        num_repetitions = int(round(time_for_action / self.dt))
        if num_repetitions < 1:
            num_repetitions = 1
            
        # Ensure exact float accounting matching steps
        time_for_action = num_repetitions * self.dt
            
        done = False
        truncated = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        intermediate_states = []

        obs = None
        while current_step < num_repetitions and not (done or truncated):
            obs, reward, done, truncated, step_info = self.env.step(u)
            intermediate_states.append(obs)
            
            # apply simple discounting within the step
            total_reward += (self.discounting ** current_step) * reward
            total_cost += (self.discounting ** current_step) * step_info.get('cost', 0.0)
            
            # Update info with latest from step_info (excluding specific accumulation keys)
            for k, v in step_info.items():
                if k not in ['cost', 'steps']:
                    info[k] = v
                    
            current_step += 1
            
        # penalize reward with switch cost
        total_reward = total_reward - self.switch_cost(None, u)
        
        # append time to observation
        self.time_to_go -= time_for_action
        
        # Check truncation due to running out of time
        if self.time_to_go <= 1e-4:
            truncated = True
            self.time_to_go = 0.0
            
        augmented_obs = np.concatenate([obs, np.array([self.time_to_go], dtype=obs.dtype)])
        
        info['steps'] = current_step
        info['cost'] = total_cost
        info['dt'] = time_for_action 
        info['intermediate_states'] = intermediate_states
        
        return augmented_obs, float(total_reward), done, truncated, info

