import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor

from src.envs.equilibria import EquilibriumStrategy
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.base_learner import MABaseAlgorithm, OnPolicyBaseAlgorithm

_CUDA_OOM_ERR_MSG_START = "CUDA out of memory. Tried to allocate"
_CPU_OOM_ERR_MSG_START = "[enforce fail at alloc_cpu.cpp:73]"
ERR_MSG_OOM_SINGLE_BATCH = "Failed for good. Even a batch size of 1 leads to OOM!"


def catch_failed_batch_allocation(device: int, batch_size: int, e):
    if device == "cpu" and str(e).startswith(_CPU_OOM_ERR_MSG_START):
        raise RuntimeError("Can't determine variable batch size to fit in CPU.")
    if not str(e).startswith(_CUDA_OOM_ERR_MSG_START):
        raise e
    if batch_size <= 1:
        traceback.print_exc()
        raise RuntimeError(ERR_MSG_OOM_SINGLE_BATCH)


def log_l2_distance_to_equilibrium(
    env: BaseEnvForVec,
    learners: Dict[int, Union[MABaseAlgorithm, OnPolicyBaseAlgorithm]],
    equ_strategies: Dict[int, EquilibriumStrategy],
    num_envs: int,
    num_stages: int,
) -> Dict[int, List[float]]:
    """Estimate the l2_distance between the learned and the equilibrium strategies.
    We estimate the following:
        L_2^{it}(\beta, \beta^*)
        = \left(\frac{1}{n_{\text{num_envs}}} \sum_{s_{it} \sim P_{it}\left(\cdot \,|\,\beta^* \right)}\left(
            \beta_{it}(s_{it}) - \beta_{it}^*(s_{it}) \right)^2 \right)^{\frac{1}{2}}
    We do this batch-wise:
    1. Calculate the squared sum over l2-distances for one batch.
    2. Sum up the squared distances over the batches.
    3. Divide by the total number of simulations (num_envs) and take the square-root.
    """
    excluded_agents = []
    if len(set(learners.values())) == 1:
        excluded_agents = [agent_id for agent_id in learners.keys() if agent_id != 0]

    agent_ids = list(
        set(learners.keys()) & set(equ_strategies.keys()) - set(excluded_agents)
    )
    batch_size = num_envs
    num_done_sims = 0
    l2_distances = {i: [0.0] * num_stages for i in agent_ids}

    while num_done_sims <= num_envs:
        try:
            print("\nCalculate L2-loss with batch_size: " + str(batch_size))
            learner_actions_list = []
            states_list, observations_list, equ_actions_list, _ = run_algorithms(
                env, equ_strategies, batch_size, num_stages, deterministic=True
            )
            get_ma_actions_for_observation_list(
                env, learners, learner_actions_list, states_list, observations_list
            )
            equ_actions_list, learner_actions_list = env.l2_loss_adaption_callback(
                states_list=states_list,
                observations_list=observations_list,
                equ_actions_list=equ_actions_list,
                learner_actions_list=learner_actions_list,
            )
            store_squared_l2_distance_in_dict(
                l2_distances, learner_actions_list, equ_actions_list
            )
            num_done_sims += batch_size
        except RuntimeError as e:
            catch_failed_batch_allocation(env.device, batch_size, e)
            batch_size = int(batch_size / 2)

    normalize_batch_wise_calculated_l2_distance(num_done_sims, l2_distances)
    log_l2_distances(learners, l2_distances)
    return l2_distances


def normalize_batch_wise_calculated_l2_distance(
    num_done_sims: int, l2_distances: Dict[int, List[float]]
):
    for agent_id, sqaured_l2_distance_list in l2_distances.items():
        for stage in range(len(sqaured_l2_distance_list)):
            l2_distances[agent_id][stage] = (
                l2_distances[agent_id][stage] / num_done_sims
            ) ** (1 / 2)


def get_ma_actions_for_observation_list(
    env: BaseEnvForVec,
    learners: Dict[int, Union[MABaseAlgorithm, OnPolicyBaseAlgorithm]],
    learner_actions_list: List[Dict[int, torch.Tensor]],
    states_list: List[Optional[Dict[int, torch.Tensor]]],
    observations_list: List[Dict[int, torch.Tensor]],
):
    for stage, observations in enumerate(observations_list):
        learner_actions_list.append(
            env.get_ma_actions_for_env(
                learners,
                observations=observations,
                deterministic=True,
                no_grad=True,
                states=states_list[stage],
            )
        )


def store_squared_l2_distance_in_dict(
    l2_distances_dict: Dict[int, List[float]],
    learner_actions_list: List[Dict[int, torch.Tensor]],
    equ_actions_list: List[Dict[int, torch.Tensor]],
):
    """We add the squared L2-distance to the current values of the l2_distances_dict.
    This way, we can add it batch-wise and normalize it in the end (dividing by num_samples and tacking the square-root).
    """
    for agent_id, sqaured_l2_distance_list in l2_distances_dict.items():
        for stage in range(len(sqaured_l2_distance_list)):
            l2_distances_dict[agent_id][stage] += (
                (
                    torch.dist(
                        learner_actions_list[stage][agent_id],
                        equ_actions_list[stage][agent_id],
                    )
                    ** 2
                )
                .detach()
                .item()
            )


def log_l2_distances(
    learners: Dict[int, Union[MABaseAlgorithm, OnPolicyBaseAlgorithm]],
    distances_l2: Dict[int, List[float]],
):
    for agent_id, distances in distances_l2.items():
        for stage, distance in enumerate(distances):
            learners[agent_id].logger.record(
                "eval/L2_distance_stage_" + str(stage), distance
            )


def run_algorithms(
    env: BaseEnvForVec,
    algorithms: Dict[int, Union[MABaseAlgorithm, OnPolicyBaseAlgorithm]],
    num_envs: int,
    num_steps: int,
    deterministic: bool = False,
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
                action, _ = algorithm.predict(
                    observations[agent_id], deterministic=deterministic
                )
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
