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

from src.envs.sequential_auction import SequentialFPSBAuction
from src.envs.torch_vec_env import TorchVecEnv
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.learners.ppo import VecPPO
from src.learners.utils import new_log_path


def get_config(path):
    """Config"""
    # hydra.initialize(config_path="../configs", job_name="run")
    # cfg = hydra.compose(config_name="config")
    # TODO: no time for this:-D
    import yaml

    with open(path, "r") as stream:
        cfg = yaml.safe_load(stream)
    return cfg


def multi_agent_auction_main():
    """Benchmark multi-agent learning in custom RPS env.
    
    TODO:
    * Custom net: ReLU on output
    """
    config = get_config("configs/rl_envs/sequential_fpsb_auction.yaml")
    device = "cuda:2"
    num_envs = 2 ** 12
    n_steps = 128

    # env
    base_env = SequentialFPSBAuction(config, device=device)
    env = TorchVecEnv(base_env, num_envs=num_envs, device=device)

    # policy
    policy_kwargs = dict(
        activation_fn=torch.nn.SELU, net_arch=[dict(pi=[20, 20], vf=[20, 20])]
    )

    log_path = new_log_path("logs/sequential-auction/run")
    print("============")
    print("Starting run")
    print("------------")
    learners = MultiAgentCoordinator(
        env=env,
        learner_class=VecPPO,
        learner_kwargs={
            "policy": "MlpPolicy",
            "device": device,
            "n_steps": n_steps,
            "batch_size": n_steps * num_envs,
            "tensorboard_log": log_path,
            "verbose": 0,
            "policy_kwargs": policy_kwargs,
        },
        policy_sharing=True,
    )

    # train the agent
    print(f"Logging to {log_path}.")
    learners.learn(total_timesteps=10_000_000_000)

    return None


if __name__ == "__main__":
    multi_agent_auction_main()
