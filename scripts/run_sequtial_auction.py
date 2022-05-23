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

from src.envs.sequential_auction import SequentialAuction
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.learners.ppo import VecPPO
from src.learners.utils import new_log_path
from src.utils_folder.logging_utils import logging_plots_to_gif


def get_config():
    """Config"""
    hydra.initialize(config_path="./../configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def multi_agent_auction_main():
    """Benchmark multi-agent learning in custom RPS env.

    TODO: @Nils: does this solve the negative biddings at other TODOs as well?
    * Custom net: ReLU on output
    """
    cfg = get_config("configs/rl_envs/sequential_fpsb_auction.yaml")
    config = cfg.rl_envs

    for num_rounds_to_play in [3]:
        for payment in ["first"]:

            config["num_rounds_to_play"] = num_rounds_to_play
            config["num_agents"] = config["num_rounds_to_play"] + 1

            collapse_symmetric_opponents = True

            device = "cuda:2"
            num_envs = 2 ** 12
            n_steps = 128
            payments = payment

            torch.set_printoptions(precision=4)

            # env
            base_env = SequentialAuction(
                config,
                payments=payments,
                collapse_symmetric_opponents=collapse_symmetric_opponents,
                device=device,
            )
            env = MATorchVecEnv(base_env, num_envs=num_envs, device=device)

            # policy
            policy_kwargs = dict(
                activation_fn=torch.nn.SELU, net_arch=[dict(pi=[10, 10], vf=[10, 10])]
            )

            log_path = new_log_path(
                f"logs/sequential-auction-debug/{payments}/{config['num_rounds_to_play']}/run"
            )
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
            learners.learn(total_timesteps=500_000_000)
            logging_plots_to_gif(learners.writer.log_dir)

    return None


if __name__ == "__main__":
    multi_agent_auction_main()
