"""Run experiments as reported in the paper."""
import os
import sys
from itertools import product

import hydra
import numpy as np
import pandas as pd

sys.path.append(os.path.realpath("."))

import src.utils.env_utils as env_ut
import src.utils.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator

LOG_PATH = "./logs/test/"


def run_sequential_sales_experiment():

    runs = 2
    iteration_num = 2
    n_steps_per_iteration = 1
    policy_sharing = False

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
                f"device={6}",
                f"seed={i}",
                f"policy_sharing={policy_sharing}",
                f"algorithms=[{algorithm}]",
                f"iteration_num={iteration_num}",
                f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                f"verify_br={True}",
                f"log_path={LOG_PATH}",
                f"rl_envs.collapse_symmetric_opponents={collapse_symmetric_opponents}",
                f"rl_envs.mechanism_type={mechanism_type}",
                f"rl_envs.num_rounds_to_play={num_rounds_to_play}",
                f"rl_envs.num_agents={num_rounds_to_play + 1}",
            )

            # Set up env and learning
            env = env_ut.get_env(config)
            ma_learner = MultiAgentCoordinator(config, env)
            ma_learner.learn(
                total_timesteps=config.total_training_steps,
                n_steps_per_iteration=config.n_steps_per_iteration,
                log_interval=1,
                eval_freq=config.eval_freq,
                n_eval_episodes=5,
            )

            # Wrap up
            io_ut.wrap_up_experiment_logging(config)

    print("Done!")


def run_signaling_contest_experiment():

    runs = 2
    iteration_num = 2
    n_steps_per_iteration = 1
    policy_sharing = False

    algorithms = ["ppo", "reinforce"]
    information_cases = ["true_valuations", "winning_bids"]
    options = product(algorithms, information_cases)

    for option in options:
        algorithm, information_case = option

        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure and set hyperparameters
            config = io_ut.get_config(
                f"seed={i}",
                f"policy_sharing={policy_sharing}",
                f"algorithms=[{algorithm}]",
                f"iteration_num={iteration_num}",
                f"n_steps_per_iteration=[{n_steps_per_iteration}]",
                f"verify_br={True}",
                f"log_path={LOG_PATH}",
                f"rl_envs=signaling_contest",
                f"rl_envs.information_case={information_case}",
            )

            # Set up env and learning
            env = env_ut.get_env(config)
            ma_learner = MultiAgentCoordinator(config, env)
            ma_learner.learn(
                total_timesteps=config.total_training_steps,
                n_steps_per_iteration=config.n_steps_per_iteration,
                log_interval=1,
                eval_freq=config.eval_freq,
                n_eval_episodes=5,
            )

            # Wrap up
            io_ut.wrap_up_experiment_logging(config)

    print("Done!")


if __name__ == "__main__":

    run_sequential_sales_experiment()
    run_signaling_contest_experiment()
