import copy

import hydra

import src.utils_folder.env_utils as env_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def run_limited_learning(config):
    """Runs multi agent learning for `config`."""
    config = copy.deepcopy(config)

    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    ma_learner.learn(
        total_timesteps=config["total_training_steps"],
        n_rollout_steps=config["ma_n_rollout_steps"],
        log_interval=None,
        eval_freq=config["eval_freq"],
        n_eval_episodes=1,
        tb_log_name="MultiAgent",
    )
    hydra.core.global_hydra.GlobalHydra().clear()
