from typing import Dict

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.sequential_auction import SequentialAuction
from src.envs.signaling_contest import SignalingContest
from src.envs.torch_vec_env import MATorchVecEnv


def get_env(config: Dict) -> MATorchVecEnv:
    env_name = config["rl_envs"]["name"]
    if env_name == "rockpaperscissors":
        env = RockPaperScissors(config["rl_envs"], device=config["device"])
    elif env_name == "sequential_auction":
        env = SequentialAuction(config["rl_envs"], device=config["device"])
    elif env_name == "signaling_contest":
        env = SignalingContest(config["rl_envs"], device=config["device"])
    else:
        raise ValueError("No known env chosen, check again: " + str(env_name))
    return MATorchVecEnv(env, num_envs=config["num_envs"], device=config["device"])
