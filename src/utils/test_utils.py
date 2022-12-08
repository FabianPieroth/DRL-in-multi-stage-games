import copy
from typing import Dict, List

import hydra
import omegaconf

import src.utils.env_utils as env_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def run_limited_learning(config):
    """Runs multi agent learning for `config`."""
    config = copy.deepcopy(config)

    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    ma_learner.learn()
    hydra.core.global_hydra.GlobalHydra().clear()
    return ma_learner


def set_specific_algo_configs(
    config: omegaconf.OmegaConf,
    algorithms: List[str],
    algo_instance_dict: Dict[str, str],
):
    for algo_name in algorithms:
        algo_instance = algo_instance_dict[algo_name]
        algorithm_config = omegaconf.OmegaConf.load(
            "./configs/algorithm_configs/" + algo_name + "/" + algo_instance + ".yaml"
        )
        config.algorithm_configs[algo_name] = algorithm_config
