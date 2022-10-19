import copy
import os
import sys
from itertools import product

sys.path.append(os.path.realpath("."))

import src.utils_folder.env_utils as env_ut
import src.utils_folder.io_utils as io_ut
from experiments.evaluation_utils import *
from src.learners.multi_agent_learner import MultiAgentCoordinator


def run_sequential_sales_experiment():

    runs = 3
    total_training_steps = 50_000_000
    n_steps_per_iteration = 200
    policy_sharing = True

    collapse_symmetric_opponents_options = [True, False]
    num_rounds_to_play_options = [2, 4]
    mechanism_type_options = ["first", "second"]
    options = product(
        collapse_symmetric_opponents_options,
        num_rounds_to_play_options,
        mechanism_type_options,
    )

    for option in options:
        collapse_symmetric_opponents, num_rounds_to_play, mechanism_type = option
        config = io_ut.get_config()
        for i in range(runs):
            print("=============\nStart new run\n-------------")

            # Configure
            adapted_config = copy.deepcopy(config)
            adapted_config["seed"] = i
            adapted_config["experiment_log_path"] = (
                adapted_config["experiment_log_path"][:7]
                + "final/"
                + adapted_config["experiment_log_path"][7:]
                + f"{i}/"
            )
            # Hyperparameters
            adapted_config["total_training_steps"] = total_training_steps
            adapted_config["n_steps_per_iteration"] = n_steps_per_iteration
            adapted_config["rl_envs"][
                "collapse_symmetric_opponents"
            ] = collapse_symmetric_opponents
            adapted_config["rl_envs"]["mechanism_type"] = mechanism_type
            adapted_config["rl_envs"]["num_rounds_to_play"] = num_rounds_to_play
            adapted_config["rl_envs"]["num_agents"] = num_rounds_to_play + 1
            adapted_config["policy_sharing"] = policy_sharing
            io_ut.store_config_and_set_seed(adapted_config)

            # Set up env and learning
            env = env_ut.get_env(adapted_config)
            ma_learner = MultiAgentCoordinator(adapted_config, env)
            ma_learner.learn(
                total_timesteps=adapted_config["total_training_steps"],
                n_steps_per_iteration=adapted_config["n_steps_per_iteration"],
                log_interval=1,
                eval_freq=adapted_config["eval_freq"],
                n_eval_episodes=5,
            )

            # Wrap up
            io_ut.wrap_up_experiment_logging(adapted_config)

    print("Done!")


def evaluate_sequential_sales_experiment():
    path = "/home/kohring/sequential-auction-on-gpu/logs/final/sequential_auction"
    df = get_log_df(path)

    metrics = [
        "eval/action_equ_L2_distance_stage_0",
        "eval/action_equ_L2_distance_stage_1",
        "eval/action_equ_L2_distance_stage_2",
        "eval/action_equ_L2_distance_stage_3",
    ]

    # 1. Scalability in terms of rounds to play
    df_select = df[df.metric.isin(metrics)]
    df_select = df_select[df_select.time_step == max(df_select.time_step)]
    df_select.metric = df_select.metric.apply(metric_python2latex)

    index_without_seeds = list(df_select.index.names)
    index_without_seeds.remove("seed")
    index_without_seeds.append("metric")
    aggregate_df = df_select.groupby(index_without_seeds).agg(
        {"value": ["mean", "std"]}
    )
    aggregate_df = aggregate_df.round(decimals=4)
    aggregate_df.columns = ["mean", "std"]

    df_to_tex(
        df=aggregate_df,
        name="table_sequential_sales.tex",
        label="tab:results",
        caption="",
        path=path,
    )

    # 2. Collapse opponents
    # TODO

    pass


if __name__ == "__main__":

    run_sequential_sales_experiment()
    evaluate_sequential_sales_experiment()
