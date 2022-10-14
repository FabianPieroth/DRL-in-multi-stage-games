"""Test the `sequential_auction` game environment."""
import hydra
import torch

import src.utils_folder.io_utils as io_ut
from src.envs.sequential_auction import SequentialAuction

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"


def test_sequential_auction_in_bne():
    """Test the `sequential_auction` game environment by playing the BNE
    strategy. Collect the total rewards and compare with expectation.
    """
    io_ut.set_global_seed(0)
    batch_size: int = 2 ** 10  # The higher the lower the error tolerance should be

    config = io_ut.get_config()
    config["device"] = DEVICE
    rl_config = hydra.compose("../configs/rl_envs/sequential_auction.yaml")[""][""][""][
        "configs"
    ]
    env = SequentialAuction(rl_config["rl_envs"], device=DEVICE)

    states = env.sample_new_states(n=batch_size)
    observations = env.get_observations(states)

    rewards_total = {i: 0 for i in range(env.num_agents)}

    # Simulate game
    for stage in range(env.num_rounds_to_play):

        has_won_already = env._has_won_already(states, stage)
        actions = env.get_equilibrium_actions(
            stage, env.strategies_bne, observations, has_won_already
        )

        observations, rewards, _, states = env.compute_step(states, actions)

        for agent_id in range(env.num_agents):
            rewards_total[agent_id] += rewards[agent_id].mean().item()

    # Verify
    for agent_id, reward in rewards_total.items():
        assert (
            0.22 < reward < 0.28
        ), f"Bidder {agent_id} did not get the utility expected in BNE."
