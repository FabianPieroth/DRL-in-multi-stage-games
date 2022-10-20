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
    total_training_steps = 200_000_000
    n_steps_per_iteration = 200
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
            adapted_config["algorithms"] = algorithm
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
    df_select = df_select[
        df_select.index.get_level_values("collapse_symmetric_opponents") == False
    ]
    df_select = df_select.droplevel(level="collapse_symmetric_opponents")

    reduced_index = list(df_select.index.names)
    reduced_index.remove("seed")
    reduced_index.remove("algorithms")
    reduced_index.append("metric")
    df_select = df_select.reset_index()
    df_select = df_select.set_index(reduced_index)
    df_select.algorithms = df_select.algorithms.apply(metric_python2latex)

    aggregate_df = pd.pivot_table(
        df_select,
        values="value",
        index=reduced_index,
        columns=["algorithms"],
        aggfunc=[np.mean, np.std],
    )

    # Formatting
    aggregate_df = aggregate_df.swaplevel(axis=1)
    aggregate_df = aggregate_df[aggregate_df.columns[[0, 2, 1, 3]]]

    aggregate_df = aggregate_df.round(decimals=4)
    final_df = pd.DataFrame()
    for algorithm in set(i[0] for i in aggregate_df.columns):
        final_df[algorithm] = (
            aggregate_df[(algorithm, "mean")].astype(str)
            + " ("
            + aggregate_df[(algorithm, "std")].astype(str)
            + ")"
        )

    # Write to disk
    df_to_tex(
        df=final_df,
        name="table_sequential_sales.tex",
        label="tab:table_sequential_sales",
        caption="",
        path=path,
    )

    # 2. Collapse opponents
    # TODO

    pass


if __name__ == "__main__":

    run_sequential_sales_experiment()
    evaluate_sequential_sales_experiment()
