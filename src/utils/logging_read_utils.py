"""Utilities for reading in and aggregating logs."""
import os
from typing import List

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def python2latex(python_name: str) -> str:
    """Transform code naming of metrics and parameters into proper text and
    LaTeX format.
    """

    if python_name.startswith("eval/action_equ_L2_distance_stage_"):
        stage = int(python_name[python_name.rfind("_") + 1 :]) + 1
        return "$L_2^{S" + str(stage) + "}$"

    p2l = {
        # algorithms
        "ppo": "PPO",
        "reinforce": r"\textsc{Reinforce}",
        # metrics
        "eval/utility_loss": "$\ell$",
        "eval/estimated_utility_loss": r"$\ell^\text{est}$",
        # game
        "rl_envs.risk_aversion": r"risk $\rho$",
        "rl_envs.num_rounds_to_play": "rounds $k$",
        "rl_envs.mechanism_type": "mechanism",
        "rl_envs.information_case": "information",
    }
    if python_name in p2l.keys():
        return p2l[python_name]

    # Fall back: No LaTeX formulation found
    return python_name


def get_last_iter(
    df: pd.DataFrame, hyperparameters: List[str], metrics: List[str]
) -> pd.DataFrame:
    """Limit DataFrame to metrics of interest and last iteration."""

    # Limit data to metrics of interest
    df_select = df[df.metric.isin(metrics)]

    # Limit data to relevant columns
    columns = hyperparameters.copy()
    columns += ["seed", "algorithms", "step", "metric", "value"]
    df_select = df_select[columns]

    # Beautify
    df_select.metric = df_select.metric.apply(python2latex)
    df_select.algorithms = df_select.algorithms.apply(python2latex)

    # Limit data to last iteration
    columns.remove("value")
    columns.remove("step")
    df_select = df_select.groupby(columns).max("step")
    df_select.drop("step", axis=1, inplace=True)

    return df_select


def get_pivot_table(df: pd.DataFrame, hyperparameters: List[str]):
    """Create pivot table"""

    idx = hyperparameters.copy()
    idx += ["metric"]
    pivot = pd.pivot_table(
        df, values="value", index=idx, columns=["algorithms"], aggfunc=[np.mean, np.std]
    )

    # Formatting
    pivot = pivot.swaplevel(axis=1)
    pivot.index.names = [python2latex(p) for p in pivot.index.names]

    pivot = pivot.round(decimals=4)
    final_df = pd.DataFrame()
    for algorithm in set(i[0] for i in pivot.columns):
        final_df[algorithm] = (
            pivot[(algorithm, "mean")].astype(str)
            + " ("
            + pivot[(algorithm, "std")].astype(str)
            + ")"
        )

    return final_df


def save_df(df: pd.DataFrame, environment: str, path: str):
    """Write to disk"""
    df_to_tex(
        df=df,
        name=f"table_{environment}.tex",
        label=f"tab:table_{environment}",
        caption="",
        path=path,
    )


def df_to_tex(
    df: pd.DataFrame,
    name: str = "table.tex",
    label: str = "tab:results",
    caption: str = "",
    path: str = None,
):
    """Creates a tex file with the csv at `path` as a LaTeX table."""

    def bold(x):
        return r"\textbf{" + x + "}"

    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))

    df.to_latex(
        path + "/" + name,
        na_rep="--",
        escape=False,
        caption=caption,
        column_format="".join(["l"] * len(df.index.names) + ["r"] * len(df.columns)),
        label=label,
    )


def _dict_to_columns(df: pd.DataFrame, key, value):
    """Converts a column with dict entries into multiple columns.

    NOTE: Only supports two levels currently.
    """
    if isinstance(value, dict):
        for inner_key, inner_value in value.items():
            if isinstance(inner_value, dict):
                for inner_inner_key, inner_inner_value in inner_value.items():
                    df[key + "." + inner_key + "." + inner_inner_key] = [
                        inner_inner_value
                    ] * df.shape[0]
            else:
                df[key + "." + inner_key] = [inner_value] * df.shape[0]
    else:
        df[key] = [value] * df.shape[0]

    return df


def get_log_df(path: str):
    """Scrapes all TensorBoard logs from `path` and matches them with their
    configurations/hyperparameters. Returns wide format for easy aggregation of
    multiple experiments.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")

    summary_df = pd.DataFrame()
    for subdir, _, files in tqdm(list(os.walk(path))):

        if not any(f.endswith("run_config.yaml") for f in files):
            continue  # Directory {path} does not contain results in proper format.")

        for f in files:
            if f.startswith("events.out.tfevents"):
                tb_file = os.path.join(subdir, f)
            elif f == "run_config.yaml":
                config_file = os.path.join(subdir, f)

        # Read config
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Read data
        # General logs: Usually empty
        # event_acc = EventAccumulator(tb_file)
        # event_acc.Reload()

        # Learner specific logs
        agents = [
            l for l in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, l))
        ]
        for agent in agents:
            agent_df = tb2df(os.path.join(subdir, agent))

            # Format from long to wide
            w = agent_df.unstack()
            w = w.reset_index()
            w = w.rename(
                columns={"level_0": "metric", "level_1": "time_step", 0: "value"}
            )

            # Delete empty rows
            w = w[w.value != -1]

            # Add hyperparameters from config
            for hp_key, hp_value in config.items():
                w = _dict_to_columns(w, hp_key, hp_value)

            summary_df = pd.concat([summary_df, w], axis=0)

    def _reduce_algorithm_list_to_single_algorithm(row):
        if isinstance(row, list):
            if len(row) != 1:
                raise ValueError("Only supports single algorithm per setting.")
            return row[0]
        else:
            return row

    summary_df.algorithms = summary_df.algorithms.apply(
        _reduce_algorithm_list_to_single_algorithm
    )

    return summary_df


def tb2df(path: str):
    """TensorBoard log to `DataFrame`."""
    if len(os.listdir(path)) != 1:
        return pd.DataFrame()

    ea = EventAccumulator(os.path.join(path, os.listdir(path)[0])).Reload()
    tags = ea.Tags()["scalars"]

    steps, values = dict(), dict()
    for tag in tags:
        steps[tag], values[tag] = [], []
        for event in ea.Scalars(tag):
            steps[tag].append(event.step)
            values[tag].append(event.value)

    summary_df = pd.DataFrame()
    for tag in tags:
        df = pd.DataFrame({"step": steps[tag], tag: values[tag]}).set_index("step")
        summary_df = summary_df.join(df, how="outer")

    # drop empty 1st row
    if len(summary_df) > 0:
        summary_df.drop([0], inplace=True)

    # NOTE: We fill metrics that are calculated less frequently with last value
    summary_df.fillna(method="pad", inplace=True)

    return summary_df
