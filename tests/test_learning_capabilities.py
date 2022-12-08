"""Test whether the learners are getting closer to the expected results."""
import hydra
import pytest
import torch

import src.utils.io_utils as io_ut
import src.utils.test_utils as tst_ut
import src.utils.torch_utils as th_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_sequ_auction",
    "reinforce": "reinforce_for_sequ_auction",
}


ids, testdata = zip(
    *[
        ["first-price-ppo", ("first", "ppo", False, False, 300, 0.10)],
        ["first-price-reinforce", ("first", "reinforce", False, False, 300, 0.10)],
        ["second-price", ("second", "ppo", False, False, 300, 0.25)],
        ["first-price_policy_sharing", ("first", "ppo", True, False, 300, 0.10)],
        ["first-price_collapse_opponents", ("first", "ppo", True, True, 300, 0.10)],
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
    io_ut.set_global_seed(0)

    # change some of the default configurations
    num_agents = 3
    overrides = [
        f"device={DEVICE}",
        f"iteration_num={iteration_num}",
        f"policy_sharing={policy_sharing}",
        f"num_envs={1024}",
        f"algorithms=[{learner}]",
        f"algorithm_configs.{learner}.n_rollout_steps={4}",
        f"eval_freq={iteration_num + 2}",
    ]
    env_overrides = [
        f"rl_envs.mechanism_type={mechanism_type}",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.num_rounds_to_play={num_agents - 1}",
        f"rl_envs.reduced_observation_space={True}",
        f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
    ]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/sequential_auction.yaml", env_overrides
    ).rl_envs

    tst_ut.set_specific_algo_configs(
        config, [learner] * 3, SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT
    )
    io_ut.enrich_config(config)

    # Run learning
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
        ["symmetric_true_valuations", ("true_valuations", True, 100, 0.1)],
        ["non_symmetric_true_valuations", ("true_valuations", False, 100, 0.1)],
        # ["symmetric_winning_bids",        ("winning_bids",    True,  300, 0.08)],
        # ["non_symmetric_winning_bids",    ("winning_bids",    False, 300, 0.13)],
    ]
)


SIGNALING_CONTEST_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_signaling_contest",
    "reinforce": "reinforce_for_signaling_contest",
}


@pytest.mark.skip(reason="instable: unclear which hyperparameters are required")
@pytest.mark.parametrize(
    "information_case, policy_sharing, iteration_num, error_bound",
    testdata_sc,
    ids=ids_sc,
)
def test_learning_in_signaling_contest(
    information_case, policy_sharing, iteration_num, error_bound
):
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)

    # change some of the default configurations
    overrides = [
        f"device={DEVICE}",
        f"iteration_num={iteration_num}",
        f"policy_sharing={policy_sharing}",
        f"algorithms=[ppo]",
        f"eval_freq={iteration_num + 2}",
    ]
    env_overrides = [f"rl_envs.information_case={information_case}"]
    config = io_ut.get_config(overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/signaling_contest.yaml", env_overrides
    ).rl_envs

    tst_ut.set_specific_algo_configs(
        config, ["ppo"], SIGNALING_CONTEST_ALGO_INSTANCE_DICT
    )
    io_ut.enrich_config(config)

    # Run learning
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
    io_ut.set_global_seed(0)

    # change some of the default configurations
    algorithms = [algo_name, "rps_single_action", "rps_single_action"]
    overrides = [
        f"device={DEVICE}",
        f"iteration_num={iteration_num}",
        f"policy_sharing={False}",
        f"algorithms={algorithms}",
        f"num_envs={1024}",
        f"eval_freq={iteration_num + 2}",
    ]
    env_overrides = [f"rl_envs.num_agents={3}"]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/rockpaperscissors.yaml", env_overrides
    ).rl_envs

    tst_ut.set_specific_algo_configs(config, algorithms, RPS_ALGO_INSTANCE_DICT)
    io_ut.enrich_config(config)

    # Run learning
    ma_learner = tst_ut.run_limited_learning(config)
    states = ma_learner.env.model.sample_new_states(n=2 ** 12)
    observations = ma_learner.env.model.get_observations(states)
    ma_actions = th_ut.get_ma_actions(
        ma_learner.learners, observations, deterministic=False
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
