"""Utilities for reading in and aggregating logs."""
import os

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
        column_format="l" + "r" * (len(df.columns) - 1),
        label=label,
    )


def get_log_df(path: str):
    """Scrapes all TensorBoard logs from `path` and matches them with a
    selection of their configurations/hyperparameters. Returns wide format for
    easy aggregation of multiple experiments.
    
    NOTE: We don't match it to all configs, because these include many
    redundancies currently.
    """
    summary_df = pd.DataFrame()
    for subdir, _, files in os.walk(path):
        if any(f.endswith("run_config.yaml") for f in files):

            for f in files:
                if f.startswith("events.out.tfevents"):
                    tb_file = os.path.join(subdir, f)
                elif f == "run_config.yaml":
                    config_file = os.path.join(subdir, f)

            # 1. Read config
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Hyperparameters
            hyperparamters = {
                "collapse_symmetric_opponents": config["rl_envs"][
                    "collapse_symmetric_opponents"
                ],
                "mechanism_type": config["rl_envs"]["mechanism_type"],
                "num_rounds_to_play": config["rl_envs"]["num_rounds_to_play"],
                "algorithms": config["algorithms"],
                "seed": config["seed"],
                # "experiment_log_path": config["experiment_log_path"],
            }

            # 2. Read data
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

                for hp_key, hp_value in hyperparamters.items():
                    w[hp_key] = [hp_value] * w.shape[0]

            summary_df = pd.concat([summary_df, w], axis=0)

    summary_df.set_index(list(hyperparamters.keys()), inplace=True)

    return summary_df


def tb2df(path: str):
    """TensorBoard log to `DataFrame`."""
    dirs = os.listdir(path)
    summary_iterators = [
        EventAccumulator(os.path.join(path, name)).Reload() for name in os.listdir(path)
    ]
    tags = summary_iterators[0].Tags()["scalars"]
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
