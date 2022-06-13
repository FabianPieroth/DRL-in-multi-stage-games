"""This module tests a single step for each environment."""
import hydra
import pytest
import torch

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


def test_learning_rockpaperscissors():
    """Runs multi agent learning in RPS."""
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["total_training_steps"] = 1
    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    rl_envs["num_agents"] = 3
    config["algorithms"] = ["ppo", "ppo", "ppo"]
    config["rl_envs"] = rl_envs
    tst_ut.run_limited_learning(config)


ids_sequ_auction, testdata_sequ_auction = zip(
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
    testdata_sequ_auction,
    ids=ids_sequ_auction,
)
def test_learning_sequential_auctions(
    mechanism_type, num_agents, reduced_observation_space, collapse_symmetric_opponents
):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
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

    io_ut.enrich_config(config)
    config["total_training_steps"] = 1

    tst_ut.run_limited_learning(config)


ids_sign_contest, testdata_sign_contest = zip(
    *[
        ["no-signaling-small", ("true_valuations", 4)],
        ["no-signaling-large", ("true_valuations", 4)],
        ["signaling-small", ("winning_bids", 8)],
        ["signaling-large", ("winning_bids", 8)],
    ]
)


@pytest.mark.parametrize(
    "information_case, num_agents", testdata_sign_contest, ids=ids_sign_contest
)
def test_learning_signaling_contest(information_case, num_agents):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["policy_sharing"] = True
    config["algorithms"] = ["ppo" for _ in range(num_agents)]
    rl_envs = hydra.compose("../configs/rl_envs/signaling_contest.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    rl_envs["information_case"] = information_case
    config["rl_envs"] = rl_envs
    config["rl_envs"]["num_agents"] = num_agents

    io_ut.enrich_config(config)
    config["total_training_steps"] = 1

    tst_ut.run_limited_learning(config)
