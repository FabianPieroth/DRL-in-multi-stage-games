"""
# Resources:
* https://github.com/HumanCompatibleAI/adversarial-policies/blob/baa359420641b721aa8132d289c3318edc45ec74/src/aprl/envs/multi_agent.py
* Stable-baselines PPO self-play:
    * symmetric only (pytorch): https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
    * symmetric only (tensorflow): https://github.com/HumanCompatibleAI/adversarial-policies/blob/99700aab22f99f8353dc74b0ddaf8e5861ff34a5/src/aprl/agents/ppo_self_play.py
    * vanilla PG self-play (no stable baselines): https://github.com/mtrencseni/pytorch-playground/blob/master/11-gym-self-play/OpenAI%20Gym%20classic%20control.ipynb
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
    device = "cuda:2"
    num_envs_list = [2 ** i for i in range(0, 12, 3)]

    base_env = RockPaperScissors(config, device=device)

    learn_times = [None] * len(num_envs_list)
    eval_times = [None] * len(num_envs_list)
    for i, num_envs in enumerate(num_envs_list):

        env = TorchVecEnv(base_env, num_envs=num_envs, device=device)
        model = VecPPO(
            policy="MlpPolicy",
            env=env,
            device=device,
            batch_size=2048 * num_envs,
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
            # print('obs', obs)
            # print('action', action)
            # print('reward', reward)
        eval_times[i] = time.time() - eval_tic

    return plot(num_envs_list, learn_times, eval_times)


def plot(parallel_env_vals, learn_times, eval_times):

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=120)

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
