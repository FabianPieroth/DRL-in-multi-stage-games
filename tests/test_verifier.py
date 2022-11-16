"""Test the verifier in the `sequential_auction` game environment."""
import copy

import hydra
import pytest
import torch

import src.utils.env_utils as env_ut
import src.utils.io_utils as io_ut
from src.envs.sequential_auction import SequentialAuction
from src.learners.multi_agent_learner import MultiAgentCoordinator

DEVICE = "cuda:1" if torch.cuda.is_available() else "CPU"
EPS = 0.01

ids_verifier, testdata_verifier = zip(
    *[
        ["signaling_contest", "signaling_contest"],
        ["sequential_auction", "sequential_auction"],
    ]
)


@pytest.mark.parametrize("environment", testdata_verifier, ids=ids_verifier)
def test_verifier_in_bne(environment):
    """Test the `signaling_contest` and the `sequential_auction` game
    environment by playing the BNE strategy. Collect the total rewards and
    compare with the analytic expected utility.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)

    config = io_ut.get_config()
    config = copy.deepcopy(config)

    config.rl_envs = hydra.compose(f"/rl_envs/{environment}.yaml").rl_envs
    config.device = DEVICE

    env = env_ut.get_env(config)
    ma_learner = MultiAgentCoordinator(config, env)

    utility_losses = ma_learner.verify_in_BNE()

    for utility_loss in utility_losses.values():
        assert abs(utility_loss) < EPS
