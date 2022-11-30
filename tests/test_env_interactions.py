"""This module tests a single step for each environment."""
import hydra
import pytest
import torch

import src.utils.io_utils as io_ut
import src.utils.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"

RPS_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_rps",
    "reinforce": "reinforce_for_rps",
    "dqn": "dqn_for_rps",
    "rps_single_action": "rps_rock",
}


def test_learning_rockpaperscissors():
    """Runs multi agent learning in RPS."""
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)

    algorithms = ["ppo", "ppo", "ppo"]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"n_steps_per_iteration={1}",
        f"num_envs={1}",
        f"iteration_num={1}",
    ]
    env_overrides = [f"rl_envs.num_agents={3}"]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/rockpaperscissors.yaml", env_overrides
    ).rl_envs
    tst_ut.set_specific_algo_configs(config, algorithms, RPS_ALGO_INSTANCE_DICT)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


ids_sequ_auction, testdata_sequ_auction = zip(
    *[
        ["first-price", ("first", 2, True, False)],
        ["second-price", ("second", 2, True, False)],
        ["large", ("second", 4, True, False)],
        ["non-reduced-observation-space", ("second", 2, False, False)],
        ["collapse-symmetric-opponents", ("second", 3, False, True)],
    ]
)

SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_sequ_auction",
    "reinforce": "reinforce_for_sequ_auction",
}


@pytest.mark.parametrize(
    "mechanism_type,num_agents,reduced_observation_space,collapse_symmetric_opponents",
    testdata_sequ_auction,
    ids=ids_sequ_auction,
)
def test_learning_sequential_auctions(
    mechanism_type, num_agents, reduced_observation_space, collapse_symmetric_opponents
):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)

    algorithms = ["ppo"]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"policy_sharing={True}",
        f"n_steps_per_iteration={1}",
        f"num_envs={1}",
        f"iteration_num={1}",
    ]
    env_overrides = [
        f"rl_envs.mechanism_type={mechanism_type}",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.num_rounds_to_play={num_agents - 1}",
        f"rl_envs.reduced_observation_space={reduced_observation_space}",
        f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
    ]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/sequential_auction.yaml", env_overrides
    ).rl_envs
    tst_ut.set_specific_algo_configs(
        config, algorithms, SEQUENTIAL_AUCTION_ALGO_INSTANCE_DICT
    )

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


SIGNALING_CONTEST_ALGO_INSTANCE_DICT = {
    "ppo": "ppo_for_signaling_contest",
    "reinforce": "reinforce_for_signaling_contest",
}


ids_sign_contest, testdata_sign_contest = zip(
    *[
        ["no-signaling-small", ("true_valuations", 4)],
        ["no-signaling-large", ("true_valuations", 8)],
        ["signaling-small", ("winning_bids", 4)],
        ["signaling-large", ("winning_bids", 8)],
    ]
)


@pytest.mark.parametrize(
    "information_case, num_agents", testdata_sign_contest, ids=ids_sign_contest
)
def test_learning_signaling_contest(information_case, num_agents):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    io_ut.set_global_seed(0)
    hydra.core.global_hydra.GlobalHydra().clear()

    algorithms = ["ppo" for _ in range(num_agents)]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"policy_sharing={True}",
        f"n_steps_per_iteration={1}",
        # f"num_envs={1}",
        f"iteration_num={1}",
    ]
    env_overrides = [
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.information_case={information_case}",
    ]
    config = io_ut.get_config(overrides=overrides)
    config.rl_envs = hydra.compose(
        "rl_envs/signaling_contest.yaml", env_overrides
    ).rl_envs
    tst_ut.set_specific_algo_configs(
        config, algorithms, SIGNALING_CONTEST_ALGO_INSTANCE_DICT
    )

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)
