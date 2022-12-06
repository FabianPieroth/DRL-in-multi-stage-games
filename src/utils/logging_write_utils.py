"""Utilities for logging"""
import os
import warnings
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


def evaluate_policies(
    learners,
    env: MATorchVecEnv,
    device: Union[str, int] = None,
    n_eval_episodes: int = 20,
    deterministic: bool = True,
    render: bool = False,
    callbacks: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

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
        if render:
            env.render()

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
    return episode_rewards, episode_lengths


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
