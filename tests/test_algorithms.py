"""This module tests a single iteration for each combination of algorithms."""
import hydra
import pytest
import torch

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"

ids, testdata = zip(
    *[
        ["all_ppo", ["ppo", "ppo", "ppo"]],
        ["all_reinforce", ["reinforce", "reinforce", "reinforce"]],
        ["ppo_ppo_dummy", ["ppo", "ppo", "rps_single_action"]],
        ["dummy_ppo_dummy", ["rps_single_action", "ppo", "rps_single_action"]],
        ["all_dummy", ["rps_single_action", "rps_single_action", "rps_single_action"]],
    ]
)


@pytest.mark.parametrize("algorithms", testdata, ids=ids)
def test_algos_in_rockpaperscissors(algorithms):
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["iteration_num"] = 1
    config["num_envs"] = 32
    config["policy_sharing"] = False
    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs
    config["algorithms"] = algorithms
    io_ut.enrich_config(config)
    tst_ut.run_limited_learning(config)
