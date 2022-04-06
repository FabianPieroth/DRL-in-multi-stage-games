from typing import Dict

import hydra
import torch

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.sequential_auction import SequentialAuction
from src.envs.torch_vec_env import BaseEnvForVec


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def get_env(config: Dict) -> BaseEnvForVec:
    env_name = config["rl_envs"]["name"]
    if env_name == "rockpaperscissors":
        return RockPaperScissors(config["rl_envs"], device=config["device"])
    elif env_name == "sequential_auction":
        return SequentialAuction(config["rl_envs"], device=config["device"])
    else:
        raise ValueError("No known env chosen, check again: " + str(env_name))


def main():
    config = get_config()
    env = get_env(config)
    starting_states = env.sample_new_states(123)
    actions = torch.randint(low=0, high=3, size=(123, 3))
    info = env.compute_step(starting_states, actions)


if __name__ == "__main__":
    main()
    print("Done!")
