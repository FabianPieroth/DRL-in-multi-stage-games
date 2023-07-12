from typing import Dict, List, Optional, Union

import torch

from src.envs.equilibria import EquilibriumStrategy
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.base_learner import MABaseAlgorithm, SABaseAlgorithm
from src.learners.utils import tensor_norm


def log_l2_distance_to_equilibrium(
    env: BaseEnvForVec,
    learners: Dict[int, Union[MABaseAlgorithm, SABaseAlgorithm]],
    equ_strategies: Dict[int, EquilibriumStrategy],
    num_envs: int,
    num_stages: int,
):
    """Run algorithms for num_steps steps on num_envs environments without training.
    Args:
        env: The environment to run the algorithms on.
        algorithms: A dictionary mapping agent ids to algorithms.
        num_envs: The number of environments to run in parallel.
        num_steps: The number of steps to run the algorithms for.
    Returns:
        A tuple of lists of states, observations, actions and rewards. There is one more state and observation than action and reward. This is because the sequence begins and ends with a state. The ith action and reward correspond to the transition from the ith state to the (i+1)th state.
    """
    excluded_agents = None
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


def log_l2_distances(learners, distances_l2):
    for agent_id, distances in distances_l2.items():
        for stage, distance in enumerate(distances):
            learners[agent_id].logger.record(
                "eval/L2_distance_stage_" + str(stage), distance
            )
