"""Utilities for reading in and aggregating logs."""
import os
from typing import List

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def metric_python2latex(metric_python_name: str):

    metric_latex_name = metric_python_name

    if metric_python_name.startswith("eval/action_equ_L2_distance_stage_"):
        stage = int(metric_python_name[metric_python_name.rfind("_") + 1 :]) + 1
        metric_latex_name = "$L_2^{S" + str(stage) + "}$"

    algorithm_dict = {"ppo": "PPO", "reinforce": "REINFORCE"}
    if metric_python_name in algorithm_dict.keys():
        metric_latex_name = algorithm_dict[metric_python_name]

    return metric_latex_name


def get_last_iter(
    df: pd.DataFrame, hyperparamters: List[str], metrics: List[str]
) -> pd.DataFrame:
    """Limit DataFrame to last iteration."""
    hyperparamters += ["seed", "algorithms"]
    df.set_index(hyperparamters, inplace=True)

    # Limit data to last iteration
    df_select = df[df.metric.isin(metrics)]
    df_select = df_select[df_select.time_step == max(df_select.time_step)]
    df_select.metric = df_select.metric.apply(metric_python2latex)

    # Clean up data and index
    reduced_index = hyperparamters
    reduced_index.remove("seed")
    reduced_index.remove("algorithms")
    reduced_index.append("metric")
    df_select = df_select.reset_index()
    df_select = df_select.set_index(reduced_index)
    df_select.algorithms = df_select.algorithms.apply(metric_python2latex)

    return df_select


def get_pivot_table(df: pd.DataFrame, hyperparamters: List[str]):
    """Create pivot table"""
    hyperparamters.append("metric")
    pivot = pd.pivot_table(
        df,
        values="value",
        index=hyperparamters,
        columns=["algorithms"],
        aggfunc=[np.mean, np.std],
    )

    # Formatting
    pivot = pivot.swaplevel(axis=1)
    pivot = pivot[pivot.columns[[0, 2, 1, 3]]]

    pivot = pivot.round(decimals=4)
    final_df = pd.DataFrame()
    for algorithm in set(i[0] for i in pivot.columns):
        final_df[algorithm] = (
            pivot[(algorithm, "mean")].astype(str)
            + " ("
            + pivot[(algorithm, "std")].astype(str)
            + ")"
        )
    return pivot


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


def get_log_df(path: str):
    """Scrapes all TensorBoard logs from `path` and matches them with their
    configurations/hyperparameters. Returns wide format for easy aggregation of
    multiple experiments.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")

    summary_df = pd.DataFrame()
    for subdir, _, files in os.walk(path):

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
                if isinstance(hp_value, dict):
                    # NOTE: only 2 levels are handled
                    for hp_inner_key, hp_inner_value in hp_value.items():
                        w[hp_key + "." + hp_inner_key] = [hp_inner_value] * w.shape[0]
                else:
                    w[hp_key] = [hp_value] * w.shape[0]

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
    """TensorBoard log to `DataFrame`.

    TODO: I think this looping is slow AF!
    """
    dirs = os.listdir(path)
    summary_iterators = [
        EventAccumulator(os.path.join(path, name)).Reload() for name in os.listdir(path)
    ]
    tags = summary_iterators[0].Tags()["scalars"]

    if len(tags) == 0:
        return pd.DataFrame()

    out = dict()
    steps = [e.step for e in summary_iterators[0].Scalars(tags[0])]
    for tag in tags:
        out[tag] = []
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            out[tag].append(events[0].value)

    tags, values = zip(*out.items())
    # TODO: there's a length mismatch somewhere!
    np_values = -np.ones([len(values), len(max(values, key=lambda x: len(x)))])
    for i, j in enumerate(values):
        np_values[i][0 : len(j)] = j

    summary_df = pd.DataFrame()
    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=[tag])
        summary_df = pd.concat([summary_df, df], axis=1)

    return summary_df
