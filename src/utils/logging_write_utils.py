"""Utilities for logging"""
import os
import time
import warnings
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.utils import tensor_norm


def logging_plots_to_gif(
    log_path: str, num_frames: int = 10, starts_with: str = "plot_"
):
    """Create GIF from plots."""

    # collect path to pictures
    paths = list()
    for file in os.listdir(log_path):
        if file.endswith(".png") and file.startswith(starts_with):
            paths.append(f"{log_path}/{file}")

    if len(paths) == 0:
        return  # No figures found

    # sort and subselect
    paths = sorted(paths, key=lambda x: int(x[x.rfind("_") + 1 : -4]))
    paths = paths[0 :: max(1, int(len(paths) / num_frames))]

    # create GIF
    images = []
    for path in paths:
        images.append(imageio.imread(path))
    imageio.mimsave(f"{log_path}/{starts_with}movie.gif", images, duration=0.5)


def log_data_dict_to_learner_loggers(
    learners, data_dict: Dict[int, float], data_name: str
):
    for agent_id, learner in learners.items():
        learner.logger.record(data_name, data_dict[agent_id])


def log_figure_to_writer(
    writer: SummaryWriter, fig: plt.Figure, iteration: int, name: str
):
    if fig is not None:
        writer.add_figure(name, fig, iteration)


def evaluate_policies(
    learners,
    env: MATorchVecEnv,
    device: Union[str, int] = None,
    n_eval_episodes: int = 20,
    deterministic: bool = True,
):
    """Runs policy for ``n_eval_episodes`` episodes and logs average reward.

    :param learners: The RL learners to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    """
    env = copy(env)

    episode_iter = 0
    n_envs = env.num_envs
    episode_rewards = {agent_id: [] for agent_id in range(env.model.num_agents)}
    episode_lengths = []

    current_rewards = {
        agent_id: torch.zeros(n_envs, device=device)
        for agent_id in range(env.model.num_agents)
    }
    current_lengths = torch.zeros((n_envs,), dtype=int, device=device)
    observations = env.reset()
    episode_rollout_ends = torch.zeros((env.num_envs), dtype=bool, device=env.device)
    while episode_iter < n_eval_episodes:
        actions = th_ut.get_ma_actions(
            learners, observations, deterministic=deterministic
        )
        observations, rewards, dones, infos = env.step(actions)
        for agent_id in range(env.model.num_agents):
            current_rewards[agent_id] += rewards[agent_id]
        current_lengths += 1
        episode_starts = dones
        if dones.any().detach().item():
            episode_lengths.append(current_lengths[dones])
            current_lengths[dones] = 0
            for agent_id in range(env.model.num_agents):
                episode_rewards[agent_id].append(current_rewards[agent_id][dones])
                current_rewards[agent_id][dones] = 0.0

            episode_rollout_ends = torch.logical_or(episode_rollout_ends, dones)
            if episode_rollout_ends.all().cpu().item():
                episode_iter += 1
                episode_rollout_ends = torch.zeros(
                    (env.num_envs), dtype=bool, device=env.device
                )

    mean_episode_lengths = (
        torch.mean(torch.concat(episode_lengths).float()).detach().item()
    )

    for agent_id, learner in learners.items():
        learner.logger.record(
            "eval/ep_rew_mean",
            torch.mean(torch.concat(episode_rewards[agent_id])).detach().item(),
        )
        learner.logger.record(
            "eval/ep_rew_std",
            torch.std(torch.concat(episode_rewards[agent_id])).detach().item(),
        )
        learner.logger.record("eval/ep_len_mean", mean_episode_lengths)


def log_training_progress(learners, iteration, log_interval, break_for_policy_sharing):
    if log_interval is not None and iteration % log_interval == 0:
        for agent_id, learner in learners.items():
            if break_for_policy_sharing(agent_id):
                break
            fps = int(
                (learner.num_timesteps - learner._num_timesteps_at_start)
                / (time.time() - learner.start_time)
            )
            if len(learner.ep_info_buffer) > 0 and len(learner.ep_info_buffer[0]) > 0:
                learner.logger.record(
                    "rollout/ep_rew_mean",
                    torch.mean(
                        torch.concat(
                            [
                                ep_info[agent_id]["sa_episode_returns"]
                                for ep_info in learner.ep_info_buffer
                            ]
                        )
                    )
                    .detach()
                    .item(),
                )
                learner.logger.record(
                    "rollout/ep_len_mean",
                    torch.mean(
                        torch.concat(
                            [
                                ep_info[agent_id]["sa_episode_lengths"]
                                for ep_info in learner.ep_info_buffer
                            ]
                        )
                    )
                    .detach()
                    .item(),
                )
            learner.logger.record("time/fps", fps)
            learner.logger.record(
                "time/time_elapsed", int(time.time() - learner.start_time)
            )
            learner.logger.record("time/total_timesteps", learner.num_timesteps)
            learner.logger.dump(step=learner.num_timesteps)


def change_in_parameter_space(learners, current_parameters, running_length):
    prev_parameters = current_parameters
    current_parameters = get_policy_parameters(learners)
    for i, learner in learners.items():
        running_length[i] += tensor_norm(current_parameters[i], prev_parameters[i])
        learner.logger.record("train/running_length", running_length[i])

    return current_parameters, running_length


def get_policy_parameters(learners):
    """Collect all current neural network parameters of the policy."""
    param_dict = {}
    for agent_id, learner in learners.items():
        if learner.policy is not None:
            param_dict[agent_id] = parameters_to_vector(
                [_ for _ in learner.policy.parameters()]
            )
        else:
            param_dict[agent_id] = torch.zeros(1)

    return param_dict
