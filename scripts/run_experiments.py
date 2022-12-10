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
    policy_sharing = True

    n_steps_per_iteration_options = [8, 32, 128]
    n_rollout_steps_options = [8, 32, 128]
    collapse_symmetric_opponents_options = [True, False]
    num_rounds_to_play_options = [2, 4]
    mechanism_type_options = ["first", "second"]
    algorithm_options = ["ppo", "reinforce"]
    action_dependent_std_options = [False, True]
    options = product(
        n_steps_per_iteration_options,
        n_rollout_steps_options,
        collapse_symmetric_opponents_options,
        num_rounds_to_play_options,
        mechanism_type_options,
        algorithm_options,
        action_dependent_std_options,
    )

    for option in options:
        (
            n_steps_per_iteration,
            n_rollout_steps,
            collapse_symmetric_opponents,
            num_rounds_to_play,
            mechanism_type,
            algorithm,
            action_dependent_std,
        ) = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                overrides=[
                    f"device={device}",
                    f"seed={i}",
                    f"algorithms=[{algorithm}]",
                    f"iteration_num={iteration_num}",
                    f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                    f"algorithm_configs.{algorithm}.n_rollout_steps={n_rollout_steps}",
                    f"policy_sharing={policy_sharing}",
                    f"policy.action_dependent_std=[{action_dependent_std}]",
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
                    f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                    f"policy_sharing={policy_sharing}",
                    f"policy.action_dependent_std=[{action_dependent_std}]",
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
