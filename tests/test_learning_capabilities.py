"""Test whether the learners are getting closer to the expected results."""
import hydra
import pytest
import torch

import src.utils.io_utils as io_ut
import src.utils.test_utils as tst_ut
import src.utils.torch_utils as th_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


ids, testdata = zip(
    *[
        [
            "first-price-ppo",
            ("first", "ppo", "symmetric_uniform", 3, 2, False, False, 300, 0.10),
        ],
        [
            "first-price-reinforce",
            ("first", "reinforce", "symmetric_uniform", 3, 2, False, False, 300, 0.15),
        ],
        [
            "second-price",
            ("second", "ppo", "symmetric_uniform", 3, 2, False, False, 300, 0.25),
        ],
        [
            "first-price_policy_sharing",
            ("first", "ppo", "symmetric_uniform", 3, 2, True, False, 300, 0.10),
        ],
        [
            "first-price_collapse_opponents",
            ("first", "ppo", "symmetric_uniform", 3, 2, True, True, 300, 0.10),
        ],
        [
            "first-price-affiliated-values-ppo",
            ("first", "ppo", "affiliated_uniform", 2, 1, False, False, 300, 0.10),
        ],
        [
            "second-price-mineral-rights-ppo",
            (
                "second",
                "ppo",
                "mineral_rights_common_value",
                3,
                1,
                False,
                False,
                300,
                0.15,
            ),
        ],
    ]
)


@pytest.mark.parametrize(
    "mechanism_type, learner, sampler_type, num_agents, num_rounds, policy_sharing, collapse_symmetric_opponents, iteration_num, error_bound",
    testdata,
    ids=ids,
)
def test_learning_in_sequential_auction(
    mechanism_type: str,
    learner: str,
    sampler_type: str,
    num_agents: int,
    num_rounds: int,
    policy_sharing: bool,
    collapse_symmetric_opponents: bool,
    iteration_num: int,
    error_bound: float,
):
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)

    # change some of the default configurations
    overrides = [
        f"device={DEVICE}",
        f"iteration_num={iteration_num}",
        f"policy_sharing={policy_sharing}",
        f"num_envs={1024}",
        f"algorithms=[{learner}]",
        f"algorithm_configs.{learner}.n_rollout_steps={4}",
        f"algorithm_configs.{learner}.learning_rate_schedule=constant",
        f"eval_freq={iteration_num + 2}",
        f"rl_envs=sequential_auction",
        f"rl_envs/sampler={sampler_type}",
        f"rl_envs.mechanism_type={mechanism_type}",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.num_rounds_to_play={num_rounds}",
        f"rl_envs.reduced_observation_space={True}",
        f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
    ]
    config = io_ut.get_config(overrides=overrides)

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
        # ["symmetric_true_valuations", ("true_valuations", True, 1200, 0.2)],
        ["non_symmetric_true_valuations", ("true_valuations", False, 800, 0.25)],
        # ["symmetric_winning_bids",        ("winning_bids", True, 1200, 0.1)],
        # ["non_symmetric_winning_bids",    ("winning_bids", False, 1200, 0.1)],
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
    io_ut.set_global_seed(0)

    # change some of the default configurations
    overrides = [
        f"device={DEVICE}",
        f"iteration_num={iteration_num}",
        f"policy_sharing={policy_sharing}",
        f"algorithms=[ppo]",
        f"eval_freq={iteration_num + 2}",
        f"rl_envs=signaling_contest",
        f"rl_envs.information_case={information_case}",
        f"algorithm_configs.ppo.learning_rate_schedule=exponential",
        f"algorithm_configs.ppo.learning_rate=1e-2",
    ]
    config = io_ut.get_config(overrides=overrides)

    # Run learning
    ma_learner = tst_ut.run_limited_learning(config)
    _, _, l2_distances = ma_learner.env.model.eval_vs_equilibrium_strategies(
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
    *[["dqn_vs_rock_rock", ("dqn", 60, 0.99)], ["ppo_vs_rock_rock", ("ppo", 200, 0.95)]]
)


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
        f"rl_envs=rockpaperscissors",
        f"rl_envs.num_agents={3}",
        f"algorithm_configs.ppo.learning_rate_schedule=constant",
    ]
    config = io_ut.get_config(overrides=overrides)

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
