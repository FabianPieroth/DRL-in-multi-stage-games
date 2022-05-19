import datetime
import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf


def save_omegaconf_to_yaml(file: DictConfig, filename: str, path: str = "./"):
    full_path = path + filename + ".yaml"
    check_path_and_create(full_path)
    with open(full_path, "w") as f:
        OmegaConf.save(file, f)


def check_path_and_create(path: str):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:  # Guard against race condition
            print(path + " already exists")


def read_hydra_config():
    hydra.initialize(config_path="../../configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def get_and_store_config() -> DictConfig:
    config = read_hydra_config()
    enrich_config(config)
    store_config(config)
    return config


def store_config(config: DictConfig):
    save_omegaconf_to_yaml(config, "run_config", config["experiment_log_path"])


def enrich_config(config: DictConfig):
    # TODO: This is tedious as the key needs to exist in the yaml file before one can assign a new value
    config["experiment_log_path"] = (
        config["log_path"]
        + config["rl_envs"]["name"]
        + get_env_log_path_extension(config)
        + "/"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        + "/"
    )


def get_env_log_path_extension(config: DictConfig) -> str:
    if config["rl_envs"]["name"] == "rockpaperscissors":
        return ""
    elif config["rl_envs"]["name"] == "sequential_auction":
        return "/" + config["rl_envs"]["mechanism_type"]
    else:
        warnings.warn(
            "No env log path extension specified for: " + config["rl_envs"]["name"]
        )
        return ""
