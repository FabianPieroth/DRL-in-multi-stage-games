import copy

from stable_baselines3.common.policies import register_policy

import src.utils_folder.env_utils as env_ut
import src.utils_folder.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.learners.policies.MlpPolicy import *


def main():

    register_policy("CustomActorCriticPolicy", CustomActorCriticPolicy)

    config = io_ut.get_config()
    for i in range(1):  # You can change adapted config inside the loop
        adapted_config = copy.deepcopy(config)
        adapted_config["seed"] = i
        adapted_config["experiment_log_path"] += (
            "/" + str(i) + "/"
        )  # Adapt depending on changes in loop
        io_ut.store_config_and_set_seed(adapted_config)
        env = env_ut.get_env(adapted_config)
        ma_learner = MultiAgentCoordinator(adapted_config, env)
        ma_learner.learn(
            total_timesteps=adapted_config["total_training_steps"],
            n_steps_per_iteration=adapted_config["n_steps_per_iteration"],
            log_interval=1,
            eval_freq=adapted_config["eval_freq"],
            n_eval_episodes=5,
        )
        io_ut.wrap_up_experiment_logging(adapted_config)


if __name__ == "__main__":
    main()
    print("Done!")
