"""This module tests a single step for each environment."""
import hydra
import pytest

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut


def test_learning_rockpaperscissors():
    """Runs multi agent learning in RPS."""
    config = io_ut.get_and_store_config()
    config["total_training_steps"] = 1
    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs
    tst_ut.run_limited_learning(config)


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
def test_learning_sequential_auctions(
    mechanism_type, num_agents, reduced_observation_space, collapse_symmetric_opponents
):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    config = io_ut.get_and_store_config()
    config["total_training_steps"] = 1
    config["policy_sharing"] = True

    rl_envs = hydra.compose("../configs/rl_envs/sequential_auction.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs

    config["rl_envs"]["mechanism_type"] = mechanism_type
    config["rl_envs"]["num_agents"] = num_agents
    config["rl_envs"]["num_rounds_to_play"] = num_agents - 1
    config["rl_envs"]["reduced_observation_space"] = reduced_observation_space
    config["rl_envs"]["collapse_symmetric_opponents"] = collapse_symmetric_opponents

    tst_ut.run_limited_learning(config)
