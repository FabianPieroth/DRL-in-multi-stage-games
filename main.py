import hydra
import torch

from src.envs.rock_paper_scissors import RockPaperScissors


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def main():
    config = get_config()
    env = RockPaperScissors(config, device=1)
    starting_states = env.sample_new_states(123)
    actions = torch.randint(low=0, high=27, size=(123, 1))
    info = env.compute_step(starting_states, actions)


if __name__ == "__main__":
    main()
    print("Done!")
