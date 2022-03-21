"""Collection of known equilibria."""
import torch


def equilibrium_fpsb_symmetric_uniform(
    num_agents: int, num_units: int, player_position: int = 0
):
    """Equilibrium for FPSB symmetric uniform prior sequential auction."""

    assert (
        num_agents > num_units
    ), "For this BNE, there must be more bidders than items."

    def bid(stage: int, valuation: torch.Tensor):
        return (num_agents - num_units) / (num_agents - stage + 1) * valuation

    return bid
