import datetime
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
    config["experiment_log_path"] = (
        config["log_path"]
        + config["rl_envs"]["name"]
        + "/"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        + "/"
    )


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
