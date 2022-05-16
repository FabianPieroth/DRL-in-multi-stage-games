import datetime
import warnings
from typing import Dict

import hydra

import src.utils_folder.env_utils as env_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def read_hydra_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def get_config():
    config = read_hydra_config()
    enrich_config(config)
    return config


def enrich_config(config: Dict):
    # TODO: This is tedious as the key needs to exist in the yaml file before one can assign a new value
    env_specific_log_path_extension = get_env_log_path_extension(config)
    config["experiment_log_path"] = (
        config["log_path"]
        + config["rl_envs"]["name"]
        + get_env_log_path_extension(config)
        + "/"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        + "/"
    )


def get_env_log_path_extension(config: Dict) -> str:
    if config["rl_envs"]["name"] == "rockpaperscissors":
        return ""
    elif config["rl_envs"]["name"] == "sequential_auction":
        return "/" + config["rl_envs"]["mechanism_type"]
    else:
        warnings.warn(
            "No env log path extension specified for: " + config["rl_envs"]["name"]
        )
        return ""


def main():
    config = get_config()
    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    # train the agents
    ma_learner.learn(
        total_timesteps=config["total_training_steps"],
        log_interval=1,
        eval_freq=20,
        n_eval_episodes=5,
        tb_log_name="MultiAgent",
    )


if __name__ == "__main__":
    main()
    print("Done!")
