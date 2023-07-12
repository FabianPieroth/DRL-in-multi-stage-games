"""Run verifier experiments as reported in the paper."""
import os
import sys
from itertools import product

sys.path.append(os.path.realpath("."))
root_path = os.path.join(os.path.expanduser("~"), "sequential-auction-on-gpu")
if root_path not in sys.path:
    sys.path.append(root_path)

from time import perf_counter as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

colors = colormaps["tab20b"].colors

import src.utils.io_utils as io_ut
from scripts.run_experiments import LOG_PATH
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.utils.coordinator_utils import get_env

directory = f"{LOG_PATH}verifier/"

# USER PARAMETERS
num_stages_options = [2]
num_simulations_options = [2 ** i for i in range(5, 25, 2)]
action_discretization_options = [128, 64, 32, 16]


def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def run_verifier_analysis():
    device = 2
    runs = 3
    policy_sharing = True

    utility_losses = np.zeros(
        (
            runs,
            len(num_stages_options),
            len(num_simulations_options),
            len(action_discretization_options),
        )
    )
    elapsed_times = np.zeros_like(utility_losses)

    options = enumerated_product(
        num_stages_options, num_simulations_options, action_discretization_options
    )
    for idx, option in options:
        num_stages, num_simulations, action_discretization = option

        for i in range(runs):
            config = io_ut.get_config(
                overrides=[
                    f"seed={i}",
                    f"device={device}",
                    f"policy_sharing={policy_sharing}",
                    f"verify_br=true",
                    f"verifier.num_simulations={num_simulations}",
                    f"verifier.action_discretization={action_discretization}",
                    f"rl_envs.num_stages={num_stages}",
                    f"rl_envs.num_agents={num_stages + 1}",
                    f"delete_logs_after_training=true",
                ]
            )
            env = get_env(config)
            ma_learner = MultiAgentCoordinator(config, env)

            tic = timer()
            utility_losses_for_all_agents = ma_learner.verify_br_against_BNE().values()
            elapsed_time = timer() - tic

            utility_losses[(i,) + idx] = np.mean(list(utility_losses_for_all_agents))
            elapsed_times[(i,) + idx] = elapsed_time

            # Wrap up
            io_ut.wrap_up_learning_logging(config)

    # Save to disk
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f"{directory}utility_losses.npy", utility_losses)
    np.save(f"{directory}elapsed_times.npy", elapsed_times)


def _plot(metric, time):
    markers = ["o", "^", "s", "p", ".", "+"] * 3
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(
        [num_simulations_options[0], num_simulations_options[-1]],
        [0, 0],
        "--",
        color="black",
    )
    for ax, y in zip(axs, [metric, time]):
        for j, num_simulations in enumerate(action_discretization_options):
            mean, std = y[:, :, j].mean(axis=0), y[:, :, j].std(axis=0)
            ax.plot(
                num_simulations_options,
                mean,
                label=f"{num_simulations}",
                ms=7,
                color=colors[j],
                marker=markers[j],
            )
            ax.fill_between(
                num_simulations_options,
                mean - std,
                mean + std,
                alpha=0.4,
                color=colors[j],
            )
        ax.set_xlabel("number of initial states $M_{IS}$")
        ax.grid(linestyle="--")
        ax.set_xscale("log", base=2)
    # axs[0].set_yscale('log', base=10)
    axs[0].set_ylim([-0.003, 0.02])
    axs[1].set_yscale("log", base=2)
    axs[0].legend(title="discretization")

    return axs


def evaluate_verifier_analysis():
    # Load
    utility_losses = np.load(f"{directory}utility_losses.npy")
    elapsed_times = np.load(f"{directory}elapsed_times.npy")
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    # Plot
    for j, num_stages in enumerate(num_stages_options):
        axs = _plot(utility_losses[:, j, :, :], elapsed_times[:, j, :, :])
        # plt.suptitle(f"Sequential sales: {num_stages} stages")
        axs[0].set_ylabel("approximate utility loss $\ell^{ver}$", fontsize=12)
        axs[1].set_ylabel("run time (seconds)", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"{directory}/utility_loss_analysis_{num_stages}_stages.pdf", dpi=600
        )


if __name__ == "__main__":
    run_verifier_analysis()
    evaluate_verifier_analysis()
