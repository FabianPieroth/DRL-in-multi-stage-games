"""Evaluating the experiments run in `run_experiments.py`."""
import os
import sys

sys.path.append(os.path.realpath("."))

import src.utils.logging_read_utils as ex_ut
from scripts.run_experiments import LOG_PATH


def evaluate_sequential_sales_experiment():
    environment = "sequential_auction"
    path = f"{LOG_PATH}/{environment}_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = [
        "rl_envs.collapse_symmetric_opponents",
        "policy_sharing",
        "rl_envs.mechanism_type",
        "rl_envs.num_rounds_to_play",
        "policy.action_dependent_std",
        "n_steps_per_iteration",
    ]
    metrics = [
        "eval/action_equ_L2_distance_stage_0",
        "eval/action_equ_L2_distance_stage_1",
        "eval/action_equ_L2_distance_stage_2",
        "eval/action_equ_L2_distance_stage_3",
        "eval/utility_loss",
    ]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment, path)


def evaluate_sequential_sales_risk_experiment():
    environment = "sequential_auction"
    path = f"{LOG_PATH}/{environment}_risk_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = ["rl_envs.risk_aversion"]
    metrics = ["eval/utility_loss"]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # NOTE: possibly want to rearrange the columns
    # pivot = pivot[pivot.columns[[0, 2, 1, 3]]]

    # Write to disk
    ex_ut.save_df(pivot, environment + "_risk", path)


def evaluate_signaling_contest_experiment():
    environment = "signaling_contest"
    path = f"{LOG_PATH}/{environment}_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = ["rl_envs.information_case"]
    metrics = [
        "eval/action_equ_L2_distance_round_1",
        "eval/action_equ_L2_distance_round_2",
        "eval/utility_loss",
    ]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment, path)


if __name__ == "__main__":
    evaluate_sequential_sales_experiment()
    evaluate_sequential_sales_risk_experiment()
    evaluate_signaling_contest_experiment()
