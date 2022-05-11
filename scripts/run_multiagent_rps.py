"""Run script for multi-agent PPO in RPS."""
import os
import sys
import time

sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "sequential-auction-on-gpu"))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.learners.ppo import VecPPO


def get_config():
    """Config"""
    hydra.initialize(config_path="../configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def multi_agent_rps_main():
    """Benchmark multi-agent learning in custom RPS env."""
    config = get_config()
    config["rl_envs"]["num_rounds_to_play"] = 1
    device = "cuda:1"
    num_envs = 2 ** 12
    n_steps = 512

    base_env = RockPaperScissors(config, device=device)
    env = MATorchVecEnv(base_env, num_envs=num_envs, device=device)

    learners = MultiAgentCoordinator(
        [
            VecPPO(
                policy="MlpPolicy",
                env=env,
                device=device,
                n_steps=n_steps,
                batch_size=n_steps * num_envs,
                tensorboard_log=f"logs/multi_agent_{j}",
                verbose=0,
            )
            for j in range(2)
        ]
    )

    # train the agents
    learners.learn(total_timesteps=100_000)

    return None


if __name__ == "__main__":
    multi_agent_rps_main()
