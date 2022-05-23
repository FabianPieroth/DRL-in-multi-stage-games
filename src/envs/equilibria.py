"""Collection of known equilibria."""
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
