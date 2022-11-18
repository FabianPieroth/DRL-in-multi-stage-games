import os
import sys
import time

sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "sequential-auction-on-gpu"))

import hydra
import numpy as np
import torch

import src.utils.spaces_utils as sp_ut
from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.torch_vec_env import MATorchVecEnv


def make_vec_env(config, num_envs, device, render_n_envs=16):
    env = RockPaperScissors(config, device=device)
    neural_venv = MATorchVecEnv(
        env, num_envs=num_envs, device=device  # render_n_envs=render_n_envs
    )
    return neural_venv


def profile_env(env, n_parallel_envs, n_seq_steps, device):
    tic = time.perf_counter()
    for _ in range(n_seq_steps):
        actions_by_agent = torch.randint(
            low=0, high=3, size=(n_parallel_envs, env.model.num_agents), device=device
        )

        actions = sp_ut.ravel_multi_index(
            actions_by_agent, env.model.action_space_sizes
        )
        env.step(actions)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    sps = n_seq_steps * n_parallel_envs / elapsed_time

    print(f"n_parallel_envs: {n_parallel_envs}")
    print(f"elapsed_time: {elapsed_time:.3f} s")
    print(f"total SPS: {sps:.2f}")
    return sps


def plot(parallel_env_vals, sps_vals):
    import matplotlib.pyplot as plt
    import seaborn as sns

    perfect_scaling_slope = sps_vals[0] / parallel_env_vals[0]
    perfect_scaling_last_sps = perfect_scaling_slope * parallel_env_vals[-1]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.plot(
        [parallel_env_vals[0], parallel_env_vals[-1]],
        [sps_vals[0], perfect_scaling_last_sps],
        ls=":",
        color="k",
    )
    ax.plot(parallel_env_vals, sps_vals, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.despine(ax=ax, left=True)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel("# parallel environments")
    ax.set_ylabel("Steps per second")
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["10", "100", "1k", "10k", "100k"])
    ax.set_yticks([1e4, 1e5, 1e6, 1e7, 2e7])
    ax.set_yticklabels(["10k", "100k", "1M", "10M", "20M"])
    ax.set_title("Batch scaling Rock-Paper-Scissors")
    fig.tight_layout()
    plt.savefig("./logs/figures/benchmark_rps.png")
    plt.show()


def get_config():
    hydra.initialize(config_path="../configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def main():
    device = 1 if torch.cuda.is_available() else "cpu"
    config = get_config()
    print(f"device: {device}")

    n_seq_steps = 1001
    parallel_env_vals = (2 ** np.arange(4, 19)).tolist()
    sps_vals = []
    for n_parallel_envs in parallel_env_vals:
        env = make_vec_env(
            config, device=device, num_envs=n_parallel_envs, render_n_envs=1
        )
        sps = profile_env(
            env, n_parallel_envs=n_parallel_envs, n_seq_steps=n_seq_steps, device=device
        )
        sps_vals.append(sps)

    print(f"parallel_env_vals: {parallel_env_vals}")
    print(f"sps_vals: {sps_vals}")
    return parallel_env_vals, sps_vals


if __name__ == "__main__":
    parallel_env_vals, sps_vals = main()
    plot(parallel_env_vals, sps_vals)

    # Result 2022-01-09
    # parallel_env_vals: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    # sps_vals: [3919.086753243641, 7785.818450703951, 15295.833750795742, 30136.765511115667, 58909.47621110918,
    #            114458.24391407956, 224787.47122528145, 447411.51918771846, 892496.6896150738, 1790037.1099494188,
    #            3524822.2965223007, 7022339.540762063, 11220604.820258778, 14956807.41622171, 17488524.1235744]
