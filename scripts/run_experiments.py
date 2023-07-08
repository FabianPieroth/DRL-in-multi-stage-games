"""Run experiments as reported in the paper."""
import os
import sys
from itertools import product

sys.path.append(os.path.realpath("."))

import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut

LOG_PATH = "./logs/experiments_to_report"


def run_sequential_sales_experiment():
    environment = "sequential_auction"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 1
    runs = 10
    iteration_num = 10_000
    policy_sharing = True

    num_rounds_to_play_options = [1, 2, 4]
    mechanism_type_options = ["first", "second"]
    algorithm_options = ["ppo", "reinforce"]
    verifier_discretization = 64
    options = product(
        num_rounds_to_play_options, mechanism_type_options, algorithm_options
    )

    for option in options:
        num_rounds_to_play, mechanism_type, algorithm = option

        if num_rounds_to_play > 3:
            verifier_discretization = 16

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"eval_freq={iteration_num}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={policy_sharing}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"verifier.action_discretization={verifier_discretization}",
                    f"verifier.obs_discretization={verifier_discretization}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_asymmetric_second_price_sequential_sales_experiment():
    environment = "asymmetric_second_price_sequential_sales"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 1
    runs = 10
    iteration_num = 10_000

    num_rounds_to_play = 2
    mechanism_type = "second"
    algorithm_options = ["reinforce", "ppo"]

    for algorithm in algorithm_options:
        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"eval_freq={iteration_num}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={False}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_symmetric_budget_constraint_with_affiliation_experiments():
    environment = "symmetric_budget_constraint_plus_affiliations"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 1
    runs = 10
    iteration_num = 10_000

    num_rounds_to_play = 2
    mechanism_type_options = ["first", "second"]
    algorithm_options = ["reinforce", "ppo"]
    budgets_options = [0.6, 0.8]
    sampler = "affiliated_uniform"
    options = product(mechanism_type_options, algorithm_options, budgets_options)

    for option in options:
        mechanism_type, algorithm, budgets = option
        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"eval_freq={iteration_num}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={True}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                    f"rl_envs.budgets={budgets}",
                    f"rl_envs/sampler={sampler}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_sequential_sales_interdependent_plus_risk_experiment():
    environment = "sequential_auction"
    log_path = f"{LOG_PATH}/{environment}_interdependent_plus_risk_experiment/"

    device = 1
    runs = 10
    iteration_num = 10_000
    policy_sharing = True

    num_rounds_to_play_options = [2]
    algorithm_options = ["reinforce", "ppo"]
    risk_aversion_options = [0.25, 0.5, 0.75]

    settings = [
        dict(
            sampler_name="mineral_rights_common_value",
            mechanism_type="second",
            num_agents=3,
        ),
        dict(sampler_name="affiliated_uniform", mechanism_type="first", num_agents=2),
    ]

    options = product(
        num_rounds_to_play_options, algorithm_options, settings, risk_aversion_options
    )

    for option in options:
        num_rounds_to_play, algorithm, setting, risk_aversion = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"eval_freq={iteration_num}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={policy_sharing}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs.mechanism_type={setting['mechanism_type']}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={setting['num_agents'] + (num_rounds_to_play - 1)}",
                    f"rl_envs/sampler={setting['sampler_name']}",
                    f"rl_envs.risk_aversion={risk_aversion}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_signaling_contest_experiment():
    environment = "signaling_contest"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 1
    runs = 10
    iteration_num = 10_000
    policy_sharing = True

    information_cases = ["true_valuations", "winning_bids"]
    algorithms = ["ppo", "reinforce"]
    options = product(algorithms, information_cases)
    log_std_init = -3.0

    for option in options:
        algorithm, information_case = option
        if algorithm == "reinforce" and information_case == "winning_bids":
            log_std_init = -2.0

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"eval_freq={iteration_num}",
                    f"policy_sharing={policy_sharing}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs=signaling_contest",
                    f"rl_envs.information_case={information_case}",
                    f"policy.log_std_init={log_std_init}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


if __name__ == "__main__":
    run_sequential_sales_experiment()
    run_asymmetric_second_price_sequential_sales_experiment()
    run_symmetric_budget_constraint_with_affiliation_experiments()
    run_sequential_sales_interdependent_plus_risk_experiment()
    run_signaling_contest_experiment()
