"""This module tests a single step for each environment."""
import hydra
import pytest
import torch

import src.utils.io_utils as io_ut
import src.utils.test_utils as tst_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


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
        f"rl_envs=rockpaperscissors",
        f"rl_envs.num_agents={3}",
    ]
    config = io_ut.get_config(overrides=overrides)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


ids_sequ_auction, testdata_sequ_auction = zip(
    *[
        ["first-price-uniform", ("first", 2, True, False, "symmetric_uniform")],
        ["second-price-uniform", ("second", 2, True, False, "symmetric_uniform")],
        ["large", ("second", 4, True, False, "symmetric_uniform")],
        [
            "non-reduced-observation-space",
            ("second", 2, False, False, "symmetric_uniform"),
        ],
        [
            "collapse-symmetric-opponents",
            ("second", 3, False, True, "symmetric_uniform"),
        ],
        ["large-first-price-gaussian", ("first", 4, True, False, "symmetric_gaussian")],
        [
            "large-first-price-mineral-rights",
            ("first", 4, True, False, "mineral_rights_common_value"),
        ],
        [
            "large-first-price-affiliated_uniform",
            ("first", 4, True, False, "affiliated_uniform"),
        ],
        [
            "gaussian-non-reduced-observation-space",
            ("second", 4, False, False, "symmetric_gaussian"),
        ],
        [
            "mineral-rights-non-reduced-observation-space",
            ("second", 4, False, False, "mineral_rights_common_value"),
        ],
        [
            "affiliated-non-reduced-observation-space",
            ("second", 4, False, False, "affiliated_uniform"),
        ],
    ]
)


@pytest.mark.parametrize(
    "mechanism_type,num_agents,reduced_observation_space,collapse_symmetric_opponents, sampler_type",
    testdata_sequ_auction,
    ids=ids_sequ_auction,
)
def test_learning_sequential_auctions(
    mechanism_type,
    num_agents,
    reduced_observation_space,
    collapse_symmetric_opponents,
    sampler_type,
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
        f"rl_envs=sequential_auction",
        f"rl_envs.mechanism_type={mechanism_type}",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.num_stages={num_agents - 1}",
        f"rl_envs.reduced_observation_space={reduced_observation_space}",
        f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
        f"rl_envs/sampler={sampler_type}",
    ]
    config = io_ut.get_config(overrides=overrides)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


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
    """Runs multi agent learning in signaling contest for specified parameters."""
    io_ut.set_global_seed(0)
    hydra.core.global_hydra.GlobalHydra().clear()

    algorithms = ["ppo" for _ in range(num_agents)]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"policy_sharing={True}",
        f"n_steps_per_iteration={1}",
        f"num_envs={2}",
        f"iteration_num={1}",
        f"rl_envs=signaling_contest",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.information_case={information_case}",
    ]
    config = io_ut.get_config(overrides=overrides)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


def test_learning_bertrand_competition():
    """Runs multi agent learning in Bertrand Competition for specified parameters."""
    io_ut.set_global_seed(0)
    hydra.core.global_hydra.GlobalHydra().clear()

    algorithms = ["ppo" for _ in range(2)]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"n_steps_per_iteration={1}",
        f"num_envs={20}",
        f"iteration_num={1}",
        f"rl_envs=bertrand_competition",
        f"rl_envs.num_agents={2}",
    ]
    config = io_ut.get_config(overrides=overrides)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)


ids_coin_game, testdata_coin_game = zip(
    # (num_agents, only_pick_up_own_coin, agents_can_stack, coins_can_stack, move_order_type, boundary_treatment_type
    #  reward_structure, penalty_structure, make_obs_invariant_to_agent_order)
    *[
        [
            "single_agent-standard",
            (1, False, False, False, "random", "loop", "unitary", "no_penalty", True),
        ],
        [
            "two_agents-standard",
            (2, False, False, False, "random", "loop", "unitary", "no_penalty", True),
        ],
        [
            "standard-only-pick-up-own-coin",
            (2, True, False, False, "random", "loop", "unitary", "no_penalty", True),
        ],
        [
            "standard-agents_can_stack",
            (2, False, True, False, "random", "loop", "unitary", "no_penalty", True),
        ],
        [
            "standard-coins_can_stack",
            (2, False, False, True, "random", "loop", "unitary", "no_penalty", True),
        ],
        [
            "standard-static-move-order",
            (2, False, False, False, "static", "loop", "unitary", "no_penalty", True),
        ],
        [
            "standard-closed-boundaries",
            (2, False, False, False, "random", "closed", "unitary", "no_penalty", True),
        ],
        [
            "standard-all-others-penalty",
            (2, False, False, False, "random", "loop", "unitary", "all_others", True),
        ],
        [
            "standard-obs-NOT-invariant",
            (2, False, False, False, "random", "loop", "unitary", "no_penalty", False),
        ],
    ]
)


@pytest.mark.parametrize(
    "num_agents, only_pick_up_own_coin, agents_can_stack, coins_can_stack, move_order_type, boundary_treatment_type, reward_structure, penalty_structure, make_obs_invariant_to_agent_order",
    testdata_coin_game,
    ids=ids_coin_game,
)
def test_learning_coin_game(
    num_agents,
    only_pick_up_own_coin,
    agents_can_stack,
    coins_can_stack,
    move_order_type,
    boundary_treatment_type,
    reward_structure,
    penalty_structure,
    make_obs_invariant_to_agent_order,
):
    """Runs multi agent learning in sequential auctions for specified parameters."""
    io_ut.set_global_seed(0)
    hydra.core.global_hydra.GlobalHydra().clear()

    algorithms = ["ppo" for _ in range(num_agents)]
    overrides = [
        f"device={DEVICE}",
        f"algorithms={algorithms}",
        f"policy_sharing={False}",
        f"n_steps_per_iteration={25}",
        f"num_envs={2}",
        f"iteration_num={1}",
        f"rl_envs=coin_game",
        f"rl_envs.num_agents={num_agents}",
        f"rl_envs.only_pick_up_own_coin={only_pick_up_own_coin}",
        f"rl_envs.agents_can_stack={agents_can_stack}",
        f"rl_envs.coins_can_stack={coins_can_stack}",
        f"rl_envs.move_order_type={move_order_type}",
        f"rl_envs.boundary_treatment_type={boundary_treatment_type}",
        f"rl_envs.reward_structure={reward_structure}",
        f"rl_envs.penalty_structure={penalty_structure}",
        f"rl_envs.make_obs_invariant_to_agent_order={make_obs_invariant_to_agent_order}",
    ]
    config = io_ut.get_config(overrides=overrides)

    tst_ut.run_limited_learning(config)
    io_ut.clean_logs_after_test(config)
