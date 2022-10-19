import datetime
import os
import random
import shutil
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import src.utils_folder.policy_utils as pl_ut


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


def get_config() -> DictConfig:
    config = read_hydra_config()
    enrich_config(config)
    return config


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def store_config_and_set_seed(config: DictConfig):
    store_config(config)
    set_global_seed(config["seed"])


def store_config(config: DictConfig):
    save_omegaconf_to_yaml(config, "run_config", config["experiment_log_path"])


def enrich_config(config: DictConfig):
    config["experiment_log_path"] = get_experiment_log_path(config)
    config["n_steps_per_iteration"] = get_n_steps_per_iteration(config)
    config["total_training_steps"] = get_total_training_steps(config)


def get_total_training_steps(config: DictConfig) -> int:
    return (
        config["n_steps_per_iteration"] * config["num_envs"] * config["iteration_num"]
    )


def get_n_steps_per_iteration(config: DictConfig) -> int:
    n_rollout_steps = None
    for agent_id in range(config["rl_envs"]["num_agents"]):
        algo_name = pl_ut.get_algo_name(agent_id, config)
        algo_rollout_steps = config["algorithm_configs"][algo_name]["n_rollout_steps"]
        if algo_rollout_steps is not None and n_rollout_steps is None:
            n_rollout_steps = algo_rollout_steps
        elif algo_rollout_steps is not None and n_rollout_steps is not None:
            if algo_rollout_steps != n_rollout_steps:
                raise ValueError(
                    "Cannot handle algorithms with different rollout lengths! Check for agent: "
                    + str(agent_id)
                )
    if n_rollout_steps is None:
        return config["rl_envs"]["num_agents"]
    return n_rollout_steps


def get_experiment_log_path(config: DictConfig) -> str:
    return (
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
    elif config["rl_envs"]["name"] == "signaling_contest":
        return "/" + config["rl_envs"]["information_case"]
    elif config["rl_envs"]["name"] == "simple_soccer":
        return ""
    else:
        warnings.warn(
            "No env log path extension specified for: " + config["rl_envs"]["name"]
        )
        return ""


def wrap_up_experiment_logging(config: DictConfig):

    if config["delete_logs_after_training"]:
        delete_folder(config["experiment_log_path"])

    # Clear hydra config
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def delete_folder(path_to_folder: str):
    shutil.rmtree(path_to_folder, ignore_errors=True)


def clean_logs_after_test(config: DictConfig):
    config["delete_logs_after_training"] = True
    wrap_up_experiment_logging(config)
