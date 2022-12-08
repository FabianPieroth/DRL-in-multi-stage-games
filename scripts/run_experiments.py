"""Run experiments as reported in the paper."""
import os
import sys
from itertools import product

sys.path.append(os.path.realpath("."))

import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator

LOG_PATH = "./logs/test/"


def run_sequential_sales_experiment():

    device = 1
    verifier_device = "null"
    runs = 2
    iteration_num = 2_000
    n_steps_per_iteration = 128
    policy_sharing = True

    collapse_symmetric_opponents_options = [True, False]
    num_rounds_to_play_options = [2, 4]
    mechanism_type_options = ["first", "second"]
    algorithms = ["ppo", "reinforce"]
    options = product(
        collapse_symmetric_opponents_options,
        num_rounds_to_play_options,
        mechanism_type_options,
        algorithms,
    )

    for option in options:
        collapse_symmetric_opponents, num_rounds_to_play, mechanism_type, algorithm = (
            option
        )

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                [
                    f"device={device}",
                    f"seed={i}",
                    f"policy_sharing={policy_sharing}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                    f"log_path={LOG_PATH}",
                    f"verify_br={True}",
                    f"verifier.device={verifier_device}",
                    # f"verifier.batch_size={4}",
                    f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
                    f"rl_envs.mechanism_type={mechanism_type}",
                    f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                    f"rl_envs.num_agents={num_rounds_to_play + 1}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


def run_signaling_contest_experiment():

    device = 1
    verifier_device = "null"
    runs = 2
    iteration_num = 2_000
    n_steps_per_iteration = 128
    policy_sharing = True

    algorithms = ["ppo", "reinforce"]
    information_cases = ["true_valuations", "winning_bids"]
    options = product(algorithms, information_cases)

    for option in options:
        algorithm, information_case = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                [
                    f"device={device}",
                    f"seed={i}",
                    f"policy_sharing={policy_sharing}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                    f"log_path={LOG_PATH}",
                    f"verify_br={True}",
                    f"verifier.device={verifier_device}",
                    # f"verifier.batch_size={4}",
                    f"rl_envs=signaling_contest",
                    f"rl_envs.information_case={information_case}",
                ]
            )

            # Set up env and learning
            coord_ut.start_ma_learning(config)

    print("Done!")


if __name__ == "__main__":
    run_sequential_sales_experiment()
    run_signaling_contest_experiment()
