import src.utils_folder.env_utils as env_ut
import src.utils_folder.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def main():
    config = io_ut.get_and_store_config()
    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    # train the agents
    ma_learner.learn(
        total_timesteps=config["total_training_steps"],
        log_interval=1,
        eval_freq=config["eval_freq"],
        n_eval_episodes=5,
        tb_log_name="MultiAgent",
    )


if __name__ == "__main__":
    main()
    print("Done!")
