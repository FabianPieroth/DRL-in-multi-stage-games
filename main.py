import copy

import hydra

import src.utils.env_utils as env_ut
import src.utils.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def main():

    for i in range(1):  # You can change adapted config inside the loop

        config = io_ut.get_config()

        # run learning
        env = env_ut.get_env(config)
        ma_learner = MultiAgentCoordinator(config, env)
        ma_learner.learn(
            total_timesteps=config.total_training_steps,
            n_steps_per_iteration=config.n_steps_per_iteration,
            log_interval=1,
            eval_freq=config.eval_freq,
            n_eval_episodes=5,
        )

        io_ut.wrap_up_experiment_logging(config)


if __name__ == "__main__":
    main()
    print("Done!")
