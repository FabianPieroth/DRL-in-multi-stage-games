"""Collection of known equilibria."""
from multiprocessing.sharedctypes import Value

import torch


def equilibrium_fpsb_symmetric_uniform(
    num_agents: int, num_units: int, player_position: int = 0
):
    """Equilibrium for FPSB symmetric uniform prior sequential auction."""

    assert (
        num_agents > num_units
    ), "For this BNE, there must be more bidders than items."

    def bid_function(stage: int, valuation: torch.Tensor, won: torch.Tensor = None):
        bid = ((num_agents - num_units) / (num_agents - stage)) * valuation

        if won is not None:
            bid[won, ...] = 0

        return bid.view(-1, 1)

    return bid_function


def truthful(num_agents: int, num_units: int, player_position: int = 0):
    """Truthful bidding."""

    def bid_function(stage: int, valuation: torch.Tensor, won: torch.Tensor = None):
        bid = ((num_agents - num_units) / (num_agents - stage - 1)) * valuation

        if won is not None:
            bid[won, ...] = 0

        return bid.view(-1, 1)

    return bid_function


def no_signaling_equilibrium(num_agents: int, prior_low: float, prior_high: float):
    """Equilibrium strategy for two stage signaling contest. First round all-pay,
    second round tullock contest. True valuations of winner revealed."""

    def bid_function(
        round: int,
        valuations: torch.Tensor,
        signal_info: torch.Tensor,
        lost: torch.Tensor = None,
    ):
        if round == 1:
            bid = valuations
        elif round == 2:
            bid = (valuations ** 2 * signal_info) / (valuations + signal_info) ** 2
        else:
            raise ValueError("Only two stage contest implemented!")

        if lost is not None:
            bid[lost, ...] = 0

        return bid.view(-1, 1)

    return bid_function


def signaling_equilibrium(num_agents: int, prior_low: float, prior_high: float):
    """Equilibrium strategy for two stage signaling contest. First round all-pay,
    second round tullock contest. Bid of winner revealed."""

    def bid_function(
        round: int,
        valuation: torch.Tensor,
        signal_info: torch.Tensor,
        lost: torch.Tensor = None,
    ):
        bid = ((num_agents - num_units) / (num_agents - stage - 1)) * valuation

        if lost is not None:
            bid[lost, ...] = 0

        return bid.view(-1, 1)

    return bid_function
