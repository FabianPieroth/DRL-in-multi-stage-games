"""
Resources:
* https://github.com/HumanCompatibleAI/adversarial-policies/blob/baa359420641b721aa8132d289c3318edc45ec74/src/aprl/envs/multi_agent.py


"""
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from src.envs.rock_paper_scissors import RockPaperScissors
from src.learners import VecPPO
from src.torch_vec_env import TorchVecEnv


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def benchmark_learning():
    config = get_config()
    device = "cuda:1"
    num_envs_list = [1, 2, 4, 8, 16]

    base_env = RockPaperScissors(config, device=device)

    learn_times = [None] * len(num_envs_list)
    eval_times = [None] * len(num_envs_list)
    for i, num_envs in enumerate(num_envs_list):

        env = TorchVecEnv(base_env, num_envs=num_envs, device=device)
        model = VecPPO(
            policy="MlpPolicy",
            env=env,
            device=device,
            tensorboard_log="logs",
            verbose=0,
        )

        # train the agent
        learning_tic = time.time()
        model.learn(1)
        learn_times[i] = time.time() - learning_tic

        # evaluate the trained agent
        obs = env.reset()
        eval_tic = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done.all():
                break
        eval_times[i] = time.time() - eval_tic

    return plot(num_envs_list, learn_times, eval_times)


def plot(parallel_env_vals, learn_times, eval_times):

    fig, ax1 = plt.subplots(figsize=(4, 4), dpi=120)

    color = "tab:red"
    ax1.plot(parallel_env_vals, learn_times, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylabel("learning times", color=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.plot(parallel_env_vals, eval_times, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylabel("evaluation times", color=color)

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # sns.despine(ax=ax, left=True)
    # ax.grid(axis="y", alpha=0.2)
    ax1.set_title("Batch Scaling: Learning in RPS")
    ax1.set_xlabel("# parallel environments")

    fig.tight_layout()
    plt.savefig("./benchmark_learning.png")


if __name__ == "__main__":
    benchmark_learning()
