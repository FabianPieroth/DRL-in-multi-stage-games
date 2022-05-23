import copy

import src.utils_folder.env_utils as env_ut
import src.utils_folder.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def main():
    config = io_ut.get_and_store_config()
    for i in range(1):  # You can change adapted config inside the loop
        adapted_config = copy.deepcopy(config)
        env = env_ut.get_env(adapted_config)
        ma_learner = MultiAgentCoordinator(adapted_config, env)
        ma_learner.learn(
            total_timesteps=adapted_config["total_training_steps"],
            n_rollout_steps=adapted_config["ma_n_rollout_steps"],
            log_interval=1,
            eval_freq=adapted_config["eval_freq"],
            n_eval_episodes=5,
            tb_log_name="MultiAgent",
        )


if __name__ == "__main__":
    main()
    print("Done!")
