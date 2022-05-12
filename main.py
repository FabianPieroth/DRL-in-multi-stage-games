import hydra

import src.utils_folder.env_utils as env_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def main():
    config = get_config()
    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)
    # train the agents
    ma_learner.learn(total_timesteps=config["total_training_steps"])


if __name__ == "__main__":
    main()
    print("Done!")
