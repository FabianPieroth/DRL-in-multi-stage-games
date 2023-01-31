"""Run experiments as reported in the paper."""
import os
import sys
from itertools import product

sys.path.append(os.path.realpath("."))

import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator

LOG_PATH = "./logs/test"


def run_sequential_sales_experiment():
    environment = "sequential_auction"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 2
    runs = 3
    iteration_num = 5000
    policy_sharing = True
    # learning_rate_schedule = "exponential"
    collapse_symmetric_opponents = False

    num_rounds_to_play_options = [1, 2, 4]
    mechanism_type_options = ["first", "second"]
    algorithm_options = ["ppo", "reinforce"]
    options = product(
        num_rounds_to_play_options, mechanism_type_options, algorithm_options
    )

    for option in options:
        num_rounds_to_play, mechanism_type, algorithm = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    # f"algorithm_configs.{algorithm}.learning_rate_schedule={learning_rate_schedule}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={policy_sharing}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_sequential_sales_risk_experiment():
    environment = "sequential_auction"
    log_path = f"{LOG_PATH}/{environment}_risk_experiment/"

    device = 1
    runs = 5
    iteration_num = 5_000
    policy_sharing = True
    collapse_symmetric_opponents = False

    num_rounds_to_play_options = [2]
    mechanism_type_options = ["first", "second"]
    algorithm_options = ["ppo", "reinforce"]
    risk_aversion_options = [0.5, 0.75, 1.0]
    options = product(
        num_rounds_to_play_options,
        mechanism_type_options,
        algorithm_options,
        risk_aversion_options,
    )

    for option in options:
        num_rounds_to_play, mechanism_type, algorithm, risk_aversion = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={num_rounds_to_play}",
                    f"policy_sharing={policy_sharing}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                    f"rl_envs.risk_aversion={risk_aversion}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_signaling_contest_experiment():
    environment = "signaling_contest"
    log_path = f"{LOG_PATH}/{environment}_experiment/"

    device = 3
    runs = 3
    iteration_num = 5_000
    policy_sharing = False

    information_cases = ["true_valuations", "winning_bids"]
    algorithms = ["ppo", "reinforce"]
    action_dependent_stds = [False, True]
    options = product(algorithms, information_cases, action_dependent_stds)

    for option in options:
        algorithm, information_case, action_dependent_std = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"policy_sharing={policy_sharing}",
                    f"policy.action_dependent_std={action_dependent_std}",
                    f"log_path={log_path}",
                    f"verify_br={True}",
                    f"rl_envs=signaling_contest",
                    f"rl_envs.information_case={information_case}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


if __name__ == "__main__":
    run_sequential_sales_experiment()
    run_sequential_sales_risk_experiment()
    run_signaling_contest_experiment()
