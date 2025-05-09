"""Test the verifier in the `sequential_auction` game environment."""
import hydra
import pytest
import torch

import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"
EPS = 0.01

ids_verifier, testdata_verifier = zip(
    *[
        ["signaling_contest_signaling", ("signaling_contest", "winning_bids")],
        ["sequential_auction", ("sequential_auction", "")],
    ]
)


@pytest.mark.parametrize("environment, add_info", testdata_verifier, ids=ids_verifier)
def test_verifier_in_bne(environment, add_info):
    """Test the `signaling_contest` and the `sequential_auction` game
    environment by playing the BNE strategy. Collect the total rewards and
    compare with the analytic expected utility.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)
    overrides = [
        f"device={DEVICE}",
        f"iteration_num=0",
        f"rl_envs={environment}",
        f"verifier.action_discretization={16}",
        f"verifier.obs_discretization={16}",
    ]
    if environment == "signaling_contest":
        overrides.append(f"rl_envs.information_case={add_info}")
    config = io_ut.get_config(overrides=overrides)

    ma_learner = coord_ut.start_ma_learning(config)
    utility_losses = ma_learner.verify_br_against_BNE()

    for utility_loss in utility_losses.values():
        assert abs(utility_loss) < EPS
