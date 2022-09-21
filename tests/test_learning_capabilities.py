"""Test whether the learners are getting closer to the expected results."""
import hydra
import pytest
import torch

import src.utils_folder.io_utils as io_ut
import src.utils_folder.logging_utils as log_ut
import src.utils_folder.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_sequ_auction",
    "reinforce": "reinforce_for_sequ_auction",
}


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
    config = io_ut.get_config()
    config["device"] = DEVICE
    config["iteration_num"] = iteration_num
    config["policy_sharing"] = policy_sharing
    algorithms = [learner] * 3
    config["algorithms"] = algorithms
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
    tst_ut.set_specific_algo_configs(
        config, algorithms, SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT
    )
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
    io_ut.clean_logs_after_test(config)


ids_sc, testdata_sc = zip(
    *[
        ["symmetric_true_valuations", ("true_valuations", True, 400, 0.13)],
        ["non_symmetric_true_valuations", ("true_valuations", False, 300, 0.16)],
        # ["symmetric_winning_bids", ("winning_bids", True, 300, 0.08)],
        # ["non_symmetric_winning_bids", ("winning_bids", False, 300, 0.13)],
    ]
)


SIGNALING_CONTEST_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_signaling_contest",
    "reinforce": "reinforce_for_signaling_contest",
}


@pytest.mark.parametrize(
    "information_case, policy_sharing, iteration_num, error_bound",
    testdata_sc,
    ids=ids_sc,
)
def test_learning_in_signaling_contest(
    information_case, policy_sharing, iteration_num, error_bound
):
    hydra.core.global_hydra.GlobalHydra().clear()
    config = io_ut.get_config()
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
    tst_ut.set_specific_algo_configs(
        config, ["ppo"], SIGNALING_CONTEST_ALGO_INSTANCE_DICT
    )
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
    io_ut.clean_logs_after_test(config)


ids_rps, testdata_rps = zip(
    *[["dqn_vs_rock_rock", ("dqn", 60, 0.99)], ["ppo_vs_rock_rock", ("ppo", 60, 0.95)]]
)


RPS_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_rps",
    "reinforce": "reinforce_for_rps",
    "dqn": "dqn_for_rps",
    "rps_single_action": "rps_rock",
}


@pytest.mark.parametrize(
    "algo_name, iteration_num, error_bound", testdata_rps, ids=ids_rps
)
def test_learning_in_rps(algo_name, iteration_num, error_bound):
    hydra.core.global_hydra.GlobalHydra().clear()
    config = io_ut.get_config()
    config["device"] = DEVICE
    config["iteration_num"] = iteration_num
    config["policy_sharing"] = False
    algorithms = [algo_name, "rps_single_action", "rps_single_action"]
    config["algorithms"] = algorithms
    config["num_envs"] = 1024
    config["eval_freq"] = iteration_num + 2

    rl_envs = hydra.compose("../configs/rl_envs/rockpaperscissors.yaml")[""][""][""][
        "configs"
    ]["rl_envs"]
    tst_ut.set_specific_algo_configs(config, algorithms, RPS_ALGO_INSTANCE_DICT)
    config["rl_envs"] = rl_envs
    config["rl_envs"]["num_agents"] = 3
    io_ut.enrich_config(config)
    ma_learner = tst_ut.run_limited_learning(config)
    states = ma_learner.env.model.sample_new_states(n=2 ** 12)
    observations = ma_learner.env.model.get_observations(states)
    ma_actions = log_ut.get_eval_ma_actions(
        ma_learner.learners,
        observations,
        {agent_id: None for agent_id in range(3)},
        None,
        False,
    )
    learner_percentage_paper = (
        (torch.sum(ma_actions[0] == 1) / ma_actions[0].shape[0]).detach().cpu().item()
    )
    fixed_rock_player_percentage = (
        (
            (torch.sum(ma_actions[1] == 0) + torch.sum(ma_actions[2] == 0))
            / (2 * ma_actions[0].shape[0])
        )
        .detach()
        .cpu()
        .item()
    )
    assert (
        learner_percentage_paper >= error_bound
    ), "The algorithm did not learn the best response until now!"
    assert (
        fixed_rock_player_percentage == 1.0
    ), "The opponents are not playing rock all the time!"
    io_ut.clean_logs_after_test(config)
