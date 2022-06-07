"""Collection of known equilibria."""
import math
import warnings

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

    warnings.warn(
        "Only 2 agents per group, prior_low=1.0 and prior_high=1.5 is implemented!"
    )

    def bid_function(
        round: int,
        valuations: torch.Tensor,
        signal_info: torch.Tensor,
        lost: torch.Tensor = None,
    ):
        if round == 1:
            term_1 = 27.0 * torch.log(valuations + 1.5) - 17.0 / 2.0 * valuations
            term_2 = -43.0 / 4.0 * math.log(2.5)
            term_3 = 3.5 * torch.pow(valuations, 2) - 2.0 * torch.pow(valuations, 3)
            term_4 = -4.0 * torch.log(valuations + 1.0) * (torch.pow(valuations, 4) - 1)
            term_5 = (
                4.0
                * torch.log(valuations + 1.5)
                * (torch.pow(valuations, 4) - 81.0 / 16.0)
                + 7.0
            )
            bid = term_1 + term_2 + term_3 + term_4 + term_5
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
        valuations: torch.Tensor,
        signal_info: torch.Tensor,
        lost: torch.Tensor = None,
    ):
        bid = ((num_agents - num_units) / (num_agents - stage - 1)) * valuation

        if lost is not None:
            bid[lost, ...] = 0

        return bid.view(-1, 1)

    return bid_function
