"""This module does testing."""
import copy
from itertools import product

import hydra
import pytest

import src.utils_folder.env_utils as env_ut
import src.utils_folder.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def run_limited_learning(config):
    """Runs multi agent learning for `adapted_config`."""
    config = copy.deepcopy(config)

    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    ma_learner.learn(
        total_timesteps=config["total_training_steps"],
        n_rollout_steps=config["ma_n_rollout_steps"],
        log_interval=None,
        eval_freq=config["eval_freq"],
        n_eval_episodes=1,
        tb_log_name="MultiAgent",
    )
    hydra.core.global_hydra.GlobalHydra().clear()


def test_rockpaperscissors():
    config = io_ut.get_and_store_config()
    config["total_training_steps"] = 1
    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs
    run_limited_learning(config)


ids, testdata = zip(
    *[
        ["first-price", ("first", 2, True, False)],
        ["second-price", ("second", 2, True, False)],
        ["large", ("second", 4, True, False)],
        ["non-reduced-observation-space", ("second", 2, False, False)],
        ["collapse-symmetric-opponents", ("second", 3, False, True)],
    ]
)


@pytest.mark.parametrize(
    "mechanism_type,num_agents,reduced_observation_space,collapse_symmetric_opponents",
    testdata,
    ids=ids,
)
def test_sequential_auctions(
    mechanism_type, num_agents, reduced_observation_space, collapse_symmetric_opponents
):
    config = io_ut.get_and_store_config()
    config["total_training_steps"] = 1

    rl_envs = hydra.compose("../configs/rl_envs/sequential_auction.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs

    config["rl_envs"]["mechanism_type"] = mechanism_type
    config["rl_envs"]["num_agents"] = num_agents
    config["rl_envs"]["num_rounds_to_play"] = num_agents - 1
    config["rl_envs"]["reduced_observation_space"] = reduced_observation_space
    config["rl_envs"]["collapse_symmetric_opponents"] = collapse_symmetric_opponents

    run_limited_learning(config)
