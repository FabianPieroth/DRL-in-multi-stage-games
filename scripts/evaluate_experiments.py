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
        # "rl_envs.collapse_symmetric_opponents",
        # "policy_sharing",
        "rl_envs.mechanism_type",
        "rl_envs.num_stages",
        # "policy.action_dependent_std",
    ]
    metrics = [
        "eval/L2_distance_stage_0",
        "eval/L2_distance_stage_1",
        "eval/L2_distance_stage_2",
        "eval/L2_distance_stage_3",
        "eval/estimated_utility_loss",
        "eval/utility_loss",
    ]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment, path)


def evaluate_asymmetric_second_price_sequential_sales_experiment():
    environment = "sequential_auction"
    path = (
        f"{LOG_PATH}/asymmetric_second_price_sequential_sales_experiment/{environment}"
    )
    df = ex_ut.get_log_df(path)

    hyperparameters = ["agent_id"]
    metrics = [
        "eval/estimated_utility_loss",
        "eval/bid_mean_stage_0",
        "eval/bid_mean_stage_1",
        "eval_bid_stddev_stage_0",
        "eval_bid_stddev_stage_0",
        "eval/estimated_actual_utilities",
    ]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment + "_asymmetric_second_price", path)


def evaluate_sequential_sales_interdependent_plus_risk_experiment():
    environment = "sequential_auction"
    path = f"{LOG_PATH}/{environment}_interdependent_plus_risk_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = [
        "rl_envs.mechanism_type",
        "rl_envs.sampler.name",
        "rl_envs.risk_aversion",
    ]
    metrics = ["eval/estimated_utility_loss"]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment + "_interdependence_plus_risk", path)


def evaluate_sequential_sales_symmetric_budget_constraints_with_affiliation_experiment():
    environment = "sequential_auction"
    path = f"{LOG_PATH}/symmetric_budget_constraint_plus_affiliations_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = ["rl_envs.mechanism_type", "rl_envs.budgets"]
    metrics = ["eval/estimated_utility_loss"]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(
        pivot, environment + "_symmetric_budget_constraints_plus_affiliations", path
    )


def evaluate_signaling_contest_experiment():
    environment = "signaling_contest"
    path = f"{LOG_PATH}/{environment}_experiment/{environment}"
    df = ex_ut.get_log_df(path)

    hyperparameters = ["rl_envs.information_case"]
    metrics = [
        "eval/L2_distance_stage_0",
        "eval/L2_distance_stage_1",
        "eval/estimated_utility_loss",
        "eval/utility_loss",
    ]
    df = ex_ut.get_last_iter(df, hyperparameters, metrics, L2_average=False)

    # Create pivot table
    assert df.size > 0, "No experiments were run for these parameters."
    pivot = ex_ut.get_pivot_table(df, hyperparameters)

    # Write to disk
    ex_ut.save_df(pivot, environment, path)


if __name__ == "__main__":
    evaluate_sequential_sales_experiment()
    evaluate_sequential_sales_interdependent_plus_risk_experiment()
    evaluate_sequential_sales_symmetric_budget_constraints_with_affiliation_experiment()
    evaluate_asymmetric_second_price_sequential_sales_experiment()
    evaluate_signaling_contest_experiment()
