import numpy as np
from tqdm import tqdm

from actsafe.rl.episodic_async_env import EpisodicAsync
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.trajectory import Trajectory, TrajectoryData, Transition
from actsafe.rl.types import Agent


def _summarize_episodes(
    trajectory: TrajectoryData,
) -> tuple[float, float]:
    reward = float(trajectory.reward.sum(1).mean())
    cost = float(trajectory.cost.sum(1).mean())
    return reward, cost


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    num_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[list[Trajectory], int]:
    episode_count = 0
    episodes: list[Trajectory] = []
    
    with tqdm(
        total=num_episodes,
        unit=f"Batch (✕ {environment.num_envs} envs)",
    ) as pbar:
        while episode_count < num_episodes:
            observations = environment.reset()
            active_mask = np.ones(environment.num_envs, dtype=bool)
            per_env_trajectories = [Trajectory() for _ in range(environment.num_envs)]
            
            batch_sim_steps = 0
            while active_mask.any(): # with a continuous setup, it can happen that some envs are done before others -> TAKE THIS INTO ACCOUNT
                render = render_episodes > 0
                if render:
                    # Rendering is only supported for the first few envs anyway in EpisodicAsync
                    frame = environment.render()
                    for i in range(min(len(frame), environment.num_envs)):
                        if active_mask[i]:
                            per_env_trajectories[i].frames.append(frame[i])
                
                actions = agent(observations, train)
                next_observations, rewards, done, infos = environment.step(actions)
                
                # Count the actual base simulation steps consumed this agent call.
                # info['steps'] is written by SwitchCostWrapper (= num_repetitions).
                # For plain discrete envs it falls back to action_repeat.
                sim_steps_this_call = sum(
                    infos[i].get("steps", environment.action_repeat)
                    for i in range(environment.num_envs)
                    if active_mask[i]
                )
                batch_sim_steps += sim_steps_this_call
                
                for i in range(environment.num_envs):
                    if not active_mask[i]:
                        continue
                    
                    # Create single-batch transition
                    cost = infos[i].get("cost", 0)
                    transition = Transition(
                        observations[i:i+1], 
                        next_observations[i:i+1], 
                        actions[i:i+1], 
                        np.array([rewards[i]]), 
                        np.array([cost])
                    )
                    per_env_trajectories[i].transitions.append(transition)
                
                # Tick training schedule counters with actual sim steps.
                agent.observe_transition(transition, sim_steps=sim_steps_this_call)
                
                observations = next_observations
                active_mask &= ~done

            # Batch complete. Update global step counter with total sim steps from this batch.
            step += batch_sim_steps

            # Process all trajectories.
            for i in range(environment.num_envs):
                trajectory = per_env_trajectories[i]
                np_trajectory = trajectory.as_numpy()
                
                if train:
                    agent.observe(np_trajectory)
                
                reward, cost = _summarize_episodes(np_trajectory)
                pbar.set_postfix({"reward": reward, "cost": cost})
                
                if render_episodes > 0:
                    render_episodes -= 1
                    
                episodes.append(trajectory)
                pbar.update(1)
                episode_count += 1
                
                if episode_count >= num_episodes:
                    break
                    
    return episodes, step


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    num_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[EpochSummary, int]:
    summary = EpochSummary()
    samples, step = interact(
        agent,
        env,
        num_episodes,
        train,
        step,
        render_episodes,
    )
    summary.extend(samples)
    return summary, step
