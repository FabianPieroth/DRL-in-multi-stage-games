"""This module tests a single iteration for each combination of algorithms."""
import hydra
import pytest
import torch

import src.utils.io_utils as io_ut
import src.utils.test_utils as tst_ut

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
    io_ut.set_global_seed(0)

    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"policy_sharing={False}",
        f"n_steps_per_iteration={1}",
        f"num_envs={32}",
        f"iteration_num={1}",
    ]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose("rl_envs/rockpaperscissors.yaml").rl_envs
    tst_ut.set_specific_algo_configs(config, algorithms, ALGO_INSTANCE_DICT)
    io_ut.enrich_config(config)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)
