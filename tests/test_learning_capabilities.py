"""Test whether the learners are getting closer to the expected results."""
import hydra
import pytest
import torch

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"

ids, testdata = zip(
    *[
        ["first-price-ppo", ("first", "ppo", False, False, 30, 0.08)],
        ["first-price-reinforce", ("first", "reinforce", False, False, 30, 0.08)],
        ["second-price", ("second", "ppo", False, False, 50, 0.13)],
        ["first-price_policy_sharing", ("first", "ppo", True, False, 30, 0.08)],
        ["first-price_collapse_opponents", ("first", "ppo", True, True, 30, 0.08)],
    ]
)


@pytest.mark.parametrize(
    "mechanism_type, learner, policy_sharing, collapse_symmetric_opponents, iteration_num, error_bound",
    testdata,
    ids=ids,
)
def test_learning_in_sequential_auction(
    mechanism_type: str,
    learner: str,
    policy_sharing: bool,
    collapse_symmetric_opponents: bool,
    iteration_num: int,
    error_bound: float,
):
    hydra.core.global_hydra.GlobalHydra().clear()
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["iteration_num"] = iteration_num
    config["policy_sharing"] = policy_sharing
    config["algorithms"] = [learner] * 3
    config["num_envs"] = 1024
    config["algorithm_configs"][learner]["n_rollout_steps"] = 4
    config["eval_freq"] = iteration_num + 2

    rl_envs = hydra.compose("../configs/rl_envs/sequential_auction.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs
    num_agents = 3
    config["rl_envs"]["mechanism_type"] = mechanism_type
    config["rl_envs"]["num_agents"] = num_agents
    config["rl_envs"]["num_rounds_to_play"] = num_agents - 1
    config["rl_envs"]["reduced_observation_space"] = True
    config["rl_envs"]["collapse_symmetric_opponents"] = collapse_symmetric_opponents
    io_ut.enrich_config(config)
    ma_learner = tst_ut.run_limited_learning(config)
    _, _, l2_distances = ma_learner.env.model.do_equilibrium_and_actual_rollout(
        ma_learner.learners, 2048
    )
    average_l2_distance = (
        torch.mean(torch.tensor(list(l2_distances.values()))).detach().item()
    )
    assert (
        average_l2_distance < error_bound
    ), "The strategies are unexpectedly far away from equilibrium!"


ids_sc, testdata_sc = zip(
    *[
        ["symmetric_true_valuations", ("true_valuations", True, 300, 0.13)],
        ["non_symmetric_true_valuations", ("true_valuations", False, 300, 0.13)],
        # ["symmetric_winning_bids", ("winning_bids", True, 300, 0.08)],
        # ["non_symmetric_winning_bids", ("winning_bids", False, 300, 0.13)],
    ]
)


@pytest.mark.parametrize(
    "information_case, policy_sharing, iteration_num, error_bound",
    testdata_sc,
    ids=ids_sc,
)
def test_learning_in_signaling_contest(
    information_case, policy_sharing, iteration_num, error_bound
):
    hydra.core.global_hydra.GlobalHydra().clear()
    config = io_ut.get_and_store_config()
    config["device"] = DEVICE
    config["iteration_num"] = iteration_num
    config["policy_sharing"] = policy_sharing
    config["algorithms"] = "ppo"
    config["num_envs"] = 1024
    config["algorithm_configs"]["ppo"]["n_rollout_steps"] = 2
    config["eval_freq"] = iteration_num + 2

    rl_envs = hydra.compose("../configs/rl_envs/signaling_contest.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    config["rl_envs"] = rl_envs
    config["rl_envs"]["num_agents"] = 4
    config["rl_envs"]["information_case"] = information_case
    io_ut.enrich_config(config)
    ma_learner = tst_ut.run_limited_learning(config)
    _, _, l2_distances = ma_learner.env.model.do_equilibrium_and_actual_rollout(
        ma_learner.learners, 2048
    )
    average_l2_distance = (
        torch.mean(torch.tensor(list(l2_distances.values()))).detach().item()
    )
    print(average_l2_distance)
    assert (
        average_l2_distance < error_bound
    ), "The strategies are unexpectedly far away from equilibrium!"
