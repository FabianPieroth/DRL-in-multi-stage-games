from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor

from src.envs.equilibria import EquilibriumStrategy
from src.envs.torch_vec_env import BaseEnvForVec, MATorchVecEnv
from src.learners.base_learner import MABaseAlgorithm, SABaseAlgorithm
from src.learners.utils import tensor_norm


def log_l2_distance_to_equilibrium(
    env: BaseEnvForVec,
    learners: Dict[int, Union[MABaseAlgorithm, SABaseAlgorithm]],
    equ_strategies: Dict[int, EquilibriumStrategy],
    num_envs: int,
    num_stages: int,
) -> Dict[int, List[float]]:
    """Run algorithms for num_steps steps on num_envs environments without training.
    Args:
        env: The environment to run the algorithms on.
        algorithms: A dictionary mapping agent ids to algorithms.
        num_envs: The number of environments to run in parallel.
        num_steps: The number of steps to run the algorithms for.
    Returns:
        A tuple of lists of states, observations, actions and rewards. There is one more state and observation than action and reward. This is because the sequence begins and ends with a state. The ith action and reward correspond to the transition from the ith state to the (i+1)th state.
    """
    excluded_agents = []
    if len(set(learners.values())) == 1:
        excluded_agents = [agent_id for agent_id in learners.keys() if agent_id != 0]

    agent_ids = list(
        set(learners.keys()) & set(equ_strategies.keys()) - set(excluded_agents)
    )

    l2_distances = {i: [None] * num_stages for i in agent_ids}

    with torch.no_grad():
        states = env.sample_new_states(num_envs)
        observations = env.get_observations(states)
        for stage in range(num_stages):
            learner_actions = env.get_ma_actions_for_env(
                learners,
                observations=observations,
                deterministic=True,
                no_grad=True,
                states=states,
            )
            equ_actions = env.get_ma_actions_for_env(
                equ_strategies,
                observations=observations,
                deterministic=True,
                no_grad=True,
                states=states,
            )

            observations, rewards, _, states = env.compute_step(states, equ_actions)

            for agent_id in agent_ids:
                l2_distances[agent_id][stage] = tensor_norm(
                    learner_actions[agent_id], equ_actions[agent_id]
                )
    log_l2_distances(learners, l2_distances)
    return l2_distances


def log_l2_distances(learners, distances_l2):
    for agent_id, distances in distances_l2.items():
        for stage, distance in enumerate(distances):
            learners[agent_id].logger.record(
                "eval/L2_distance_stage_" + str(stage), distance
            )


def run_algorithms(
    env: BaseEnvForVec,
    algorithms: Dict[int, Union[MABaseAlgorithm, SABaseAlgorithm]],
    num_envs: int,
    num_steps: int,
) -> Tuple[
    List[Union[Tensor, TensorDict]],
    List[Dict[int, Tensor]],
    List[Dict[int, Tensor]],
    List[Dict[int, Tensor]],
]:
    """Run algorithms for num_steps steps on num_envs environments without
    training.

    Args:
        env: The environment to run the algorithms on.
        algorithms: A dictionary mapping agent ids to algorithms.
        num_envs: The number of environments to run in parallel.
        num_steps: The number of steps to run the algorithms for.
    Returns:
        A tuple of lists of states, observations, actions and rewards. There is
        one more state and observation than action and reward. This is because
        the sequence begins and ends with a state. The ith action and reward
        correspond to the transition from the ith state to the (i+1)th state.
    """
    device = env.device
    states_list = []
    observations_list = []
    actions_list = []
    rewards_list = []
    with torch.no_grad():
        states = env.sample_new_states(num_envs)
        states_list.append(states)
        observations = env.get_observations(states)
        observations_list.append(observations)
        for i in range(num_steps):
            actions = {}
            for agent_id, algorithm in algorithms.items():
                action, _ = algorithm.predict(observations[agent_id])
                actions[agent_id] = action
            if isinstance(actions, np.ndarray) or isinstance(actions, list):
                actions = torch.tensor(actions, device=device)

            observations, rewards, dones, states = env.compute_step(states, actions)
            assert isinstance(observations, dict) and all(
                isinstance(obs, torch.Tensor) and obs.shape[0] == num_envs
                for obs in observations.values()
            ), "Observations must be a dict of torch tensors"
            assert isinstance(rewards, dict) and all(
                isinstance(reward, torch.Tensor) and reward.shape == (num_envs,)
                for reward in rewards.values()
            ), "Rewards must be a dict of torch tensors"
            assert isinstance(dones, torch.Tensor) and dones.shape == (
                num_envs,
            ), "Dones must be a torch tensor"
            assert (
                isinstance(states, torch.Tensor) and states.shape[0] == num_envs
            ) or (
                isinstance(states, TensorDict)
                and all(
                    isinstance(s, torch.Tensor) and s.shape[0] == num_envs
                    for s in states.values()
                )
            ), "Next states must be a torch tensor or dict of torch tensors of batch size num_envs"

            states_list.append(states)
            observations_list.append(observations)
            actions_list.append(actions)
            rewards_list.append(rewards)

            n_dones = dones.sum().cpu().item()
            if n_dones > 0:
                states = states.clone()
                states[dones] = env.sample_new_states(n_dones)
                observations = env.get_observations(states)
    return states_list, observations_list, actions_list, rewards_list
