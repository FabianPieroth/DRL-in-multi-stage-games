"""Collection of known equilibria."""
import math
import warnings

import numpy as np
import torch

RELU_LAYER = torch.nn.ReLU()
"""We delete positive bids in equilibrium when appropriate. That is necessary due to 
precision errors. See https://discuss.pytorch.org/t/numerical-error-between-batch-and-single-instance-computation/56735/4"""


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


def winning_effect_term(valuations: torch.Tensor) -> torch.Tensor:
    term_1 = 27.0 * torch.log(valuations + 1.5) - 17.0 / 2.0 * valuations
    term_2 = -43.0 / 4.0 * math.log(2.5)
    term_3 = 3.5 * torch.pow(valuations, 2) - 2.0 * torch.pow(valuations, 3)
    term_4 = -4.0 * torch.log(valuations + 1.0) * (torch.pow(valuations, 4) - 1.0)
    term_5 = (
        4.0 * torch.log(valuations + 1.5) * (torch.pow(valuations, 4) - 81.0 / 16.0)
        + 7.0
    )
    return term_1 + term_2 + term_3 + term_4 + term_5


def signaling_effect_term(valuations: torch.Tensor) -> torch.Tensor:
    expre_1 = torch.log(valuations + 1.5)
    expre_2 = 16.0 * torch.log(valuations + 1.0)
    expre_3 = 8.0 * torch.log(valuations + 1.0)
    term_1 = (
        17.0 * math.log(5.0) - expre_3 - 9.0 * expre_1 - 17.0 * math.log(2.0) + 33.0
    )
    term_2 = -16.0 * valuations - 135.0 / (2.0 * valuations + 3.0)
    term_3 = torch.pow(valuations, 2) * (expre_3 - 8.0 * expre_1 + 18.0)
    term_4 = -torch.pow(valuations, 4) * (expre_2 - 16.0 * expre_1)
    term_5 = -torch.pow(valuations, 3) * (16.0 * expre_1 - expre_2 + 8.0)
    return term_1 + term_2 + term_3 + term_4 + term_5


def no_signaling_equilibrium(num_agents: int, prior_low: float, prior_high: float):
    """Equilibrium strategy for two stage signaling contest. First round all-pay,
    second round tullock contest. True valuations of winner revealed."""

    warnings.warn(
        "Only 2 agents per group, prior_low=1.0 and prior_high=1.5 is implemented!"
    )

    def bid_function(
        round: int,
        valuations: torch.Tensor,
        opponent_vals: torch.Tensor = None,
        lost: torch.Tensor = None,
    ):
        if round == 1:
            bid = winning_effect_term(valuations)
        elif round == 2:
            bid = (valuations ** 2 * opponent_vals) / (valuations + opponent_vals) ** 2
        else:
            raise ValueError("Only two stage contest implemented!")

        if lost is not None:
            bid[lost, ...] = 0

        return RELU_LAYER(bid.view(-1, 1))

    return bid_function


def signaling_equilibrium(num_agents: int, prior_low: float, prior_high: float):
    """Equilibrium strategy for two stage signaling contest. First round all-pay,
    second round tullock contest. Bid of winner revealed."""

    warnings.warn(
        "Only 2 agents per group, prior_low=1.0 and prior_high=1.5 is implemented!"
    )

    def bid_function(
        round: int,
        valuations: torch.Tensor,
        opponent_vals: torch.Tensor = None,
        lost: torch.Tensor = None,
    ):
        if round == 1:
            bid = signaling_effect_term(valuations) + winning_effect_term(valuations)
        elif round == 2:
            bid = (valuations ** 2 * opponent_vals) / (valuations + opponent_vals) ** 2
        else:
            raise ValueError("Only two stage contest implemented!")

        if lost is not None:
            bid[lost, ...] = 0

        return RELU_LAYER(bid.view(-1, 1))

    return bid_function


def np_array_first_round_strategy(
    valuation: np.ndarray, with_signaling: bool = True
) -> np.ndarray:
    valuation = torch.tensor(valuation)
    res = winning_effect_term(valuation)
    if with_signaling:
        res += signaling_effect_term(valuation)
    return res.numpy()
