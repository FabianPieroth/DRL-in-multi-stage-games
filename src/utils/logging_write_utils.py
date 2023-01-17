"""Utilities for logging"""
import os
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
from torch.utils.tensorboard import SummaryWriter

import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import MATorchVecEnv


def logging_plots_to_gif(log_path: str, num_frames: int = 10):
    """Create GIF from plots."""

    # collect path to pictures
    paths = list()
    for file in os.listdir(log_path):
        if file.endswith(".png"):
            paths.append(f"{log_path}/{file}")

    # sort and subselect
    paths = sorted(paths, key=lambda x: int(x[x.rfind("_") + 1 : -4]))
    paths = paths[0 :: max(1, int(len(paths) / num_frames))]

    # create GIF
    images = []
    for path in paths:
        images.append(imageio.imread(path))
    imageio.mimsave(f"{log_path}/movie.gif", images, duration=0.5)


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
