"""Evaluating the experiments run in `run_experiments.py`."""
import experiments.evaluation_utils as ex_ut
import numpy as np
import pandas as pd


def evaluate_sequential_sales_experiment():
    path = "/home/kohring/sequential-auction-on-gpu/logs/final/sequential_auction"
    df = ex_ut.get_log_df(path)

    metrics = [
        "eval/action_equ_L2_distance_stage_0",
        "eval/action_equ_L2_distance_stage_1",
        "eval/action_equ_L2_distance_stage_2",
        "eval/action_equ_L2_distance_stage_3",
    ]

    # 1. Scalability in terms of rounds to play
    df_select = df[df.metric.isin(metrics)]
    df_select = df_select[df_select.time_step == max(df_select.time_step)]
    df_select.metric = df_select.metric.apply(ex_ut.metric_python2latex)
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
    df_select.algorithms = df_select.algorithms.apply(ex_ut.metric_python2latex)

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
    ex_ut.df_to_tex(
        df=final_df,
        name="table_sequential_sales.tex",
        label="tab:table_sequential_sales",
        caption="",
        path=path,
    )

    # 2. Collapse opponents
    # TODO

    pass


def evaluate_signaling_contest_experiment():
    # TODO
    pass


if __name__ == "__main__":

    run_signaling_contest_experiment()
    evaluate_signaling_contest_experiment()
