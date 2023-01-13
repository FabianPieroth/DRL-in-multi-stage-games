import datetime
import os
import random
import shutil
import sys
import warnings
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import src.utils.policy_utils as pl_ut


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


def read_hydra_config(config_path: str, overrides: List[str] = []):
    hydra.initialize(config_path=config_path, job_name="run")
    cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def get_config(
    config_path: str = "../../configs", overrides: List[str] = []
) -> DictConfig:
    """Fetch project wide config from hierarchical config folder structure.

    Args:
        config_path (str, optional): Path to configs folder. Defaults to "../../configs".
        overrides (List[str], optional): List of defaults to overwrite when fetching config.
            Defaults to [].
            Example:
                overrides = [
                f"seed={i}",
                f"device='cuda:1'",
                f"rl_envs.num_agents={9}",
                f"rl_envs=signaling_contest"
                f"rl_envs/sampler=uniform_symmetric"  # When overriding deeper nested defaults
            ]
            NOTE: The order in overrides matters! Deeper changes first
    Returns:
        DictConfig:
    """
    config = read_hydra_config(config_path, overrides)
    enrich_config(config)

    # store config and set seed
    store_config(config)
    set_global_seed(config["seed"])

    return config


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def store_config(config: DictConfig):
    save_omegaconf_to_yaml(config, "run_config", config.experiment_log_path)


def enrich_config(config: DictConfig):
    config.experiment_log_path = get_experiment_log_path(config)
    config.n_steps_per_iteration = get_n_steps_per_iteration(config)
    config.total_training_steps = get_total_training_steps(config)
    remove_unused_algorithm_settings(config)


def get_total_training_steps(config: DictConfig) -> int:
    return config.n_steps_per_iteration * config.num_envs * config.iteration_num


def remove_unused_algorithm_settings(config: DictConfig):
    algos = config.algorithms
    config.algorithm_configs = {algo: config.algorithm_configs[algo] for algo in algos}


def get_n_steps_per_iteration(config: DictConfig) -> int:
    n_rollout_steps = None
    for agent_id in range(config.rl_envs.num_agents):
        algo_name = pl_ut.get_algo_name(agent_id, config)
        algo_rollout_steps = config.algorithm_configs[algo_name].n_rollout_steps
        if algo_rollout_steps is not None and n_rollout_steps is None:
            n_rollout_steps = algo_rollout_steps
        elif algo_rollout_steps is not None and n_rollout_steps is not None:
            if algo_rollout_steps != n_rollout_steps:
                raise ValueError(
                    "Cannot handle algorithms with different rollout lengths! Check for agent: "
                    + str(agent_id)
                )
    if n_rollout_steps is None:
        return config.rl_envs.num_agents
    return n_rollout_steps


def get_experiment_log_path(config: DictConfig) -> str:
    return (
        config.log_path
        + config.rl_envs.name
        + get_env_log_path_extension(config)
        + "/"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        + "/"
    )


def get_env_log_path_extension(config: DictConfig) -> str:
    if config.rl_envs.name == "rockpaperscissors":
        return ""
    elif config.rl_envs.name == "sequential_auction":
        return "/" + config.rl_envs.mechanism_type + "/" + config.rl_envs.sampler.name
    elif config.rl_envs.name == "signaling_contest":
        return "/" + config.rl_envs.information_case
    elif config.rl_envs.name == "simple_soccer":
        return ""
    else:
        warnings.warn("No env log path extension specified for: " + config.rl_envs.name)
        return ""


def wrap_up_learning_logging(config: DictConfig):

    if config.delete_logs_after_training:
        delete_folder(config.experiment_log_path)

    # Clear hydra config
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def delete_folder(path_to_folder: str):
    shutil.rmtree(path_to_folder, ignore_errors=True)


def clean_logs_after_test(config: DictConfig):
    config.delete_logs_after_training = True
    wrap_up_learning_logging(config)


def progress_bar(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
