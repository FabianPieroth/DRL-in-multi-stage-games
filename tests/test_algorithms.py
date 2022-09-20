"""This module tests a single iteration for each combination of algorithms."""
import hydra
import omegaconf
import pytest
import torch

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"

ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_rps",
    "reinforce": "reinforce_for_rps",
    "dqn": "dqn_for_rps",
    "rps_single_action": "rps_rock",
}

ids, testdata = zip(
    *[
        ["all_ppo", ["ppo", "ppo", "ppo"]],
        ["all_reinforce", ["reinforce", "reinforce", "reinforce"]],
        ["ppo_ppo_dummy", ["ppo", "ppo", "rps_single_action"]],
        ["dummy_ppo_dummy", ["rps_single_action", "ppo", "rps_single_action"]],
        ["all_dummy", ["rps_single_action", "rps_single_action", "rps_single_action"]],
        ["all_dqn", ["dqn", "dqn", "dqn"]],
        ["dummy_dqn_dummy", ["rps_single_action", "dqn", "rps_single_action"]],
    ]
)


@pytest.mark.parametrize("algorithms", testdata, ids=ids)
def test_algos_in_rockpaperscissors(algorithms):
    hydra.core.global_hydra.GlobalHydra().clear()
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["iteration_num"] = 1
    config["num_envs"] = 32
    config["policy_sharing"] = False
    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    for algo_name in algorithms:
        algo_instance = ALGO_INSTANCE_DICT[algo_name]
        algorithm_config = omegaconf.OmegaConf.load(
            "./configs/algorithm_configs/" + algo_name + "/" + algo_instance + ".yaml"
        )
        config["algorithm_configs"][algo_name] = algorithm_config
    config["rl_envs"] = rl_envs
    config["algorithms"] = algorithms
    io_ut.enrich_config(config)
    tst_ut.run_limited_learning(config)
