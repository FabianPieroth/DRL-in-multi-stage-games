import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut
from src.learners.multi_agent_learner import MultiAgentCoordinator


def main():

    for i in range(1):  # You can change adapted config with overrides inside the loop
        config = io_ut.get_config()
        # run learning
        ma_learner = coord_ut.get_ma_coordinator(config)
        ma_learner.learn()


if __name__ == "__main__":
    main()
    print("Done!")
