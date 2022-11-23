"""Collection of known equilibria."""
import math
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import numpy as np
import torch

RELU_LAYER = torch.nn.ReLU()
"""We delete positive bids in equilibrium when appropriate. That is necessary due to 
precision errors. See https://discuss.pytorch.org/t/numerical-error-between-batch-and-single-instance-computation/56735/4"""


class EquilibriumStrategy(ABC):
    """Base class for equilibrium strategies. It ensures the strategies to have the same
    interface as the learners.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.equ_method = self._init_equ_method()

    @abstractmethod
    def _init_equ_method(self) -> Callable:
        """Returns the equilibrium method for the repsective agent.

        Returns:
            Callable:   equ_method(self, observation: torch.Tensor) -> actions: torch.Tensor:
                        observation.shape=[batch_size, obs_size]
                        actions.shape=[batch_size, action_size]
        """

    @abstractmethod
    def _get_info_from_observation(self, observation: torch.Tensor) -> Tuple:
        """Takes the agent observation and extracts the relevant information
        to compute the equilibrium for the provided batch. Meant to improve
        readability, as one should specify where to find which relevant info.
        Also, needs to be revised whenever the observation is changed in the
        environment.
        TODO: Include some form of assertion to check if the observation has changed.

        Args:
            observation (torch.Tensor): shape=[batch_size, obs_size]

        Returns:
            Tuple: relevant info for equilibrium computation
        """

    def predict(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self.equ_method(observations), None


class SequetialAuctionEquilibrium(EquilibriumStrategy):
    def __init__(self, agent_id: int, config: Dict):
        self.num_agents = config["num_agents"]
        self.num_units = config["num_units"]
        self.player_position = agent_id
        self.equ_type = config["equ_type"]
        self.payments_start_index = config["payments_start_index"]
        self.reduced_obs_space = config["reduced_obs_space"]
        self.dummy_price_key = config["dummy_price_key"]
        self.valuations_start_index = config["valuations_start_index"]
        self.valuation_size = config["valuation_size"]
        assert (
            self.num_agents > self.num_units
        ), "For this BNE, there must be more bidders than items."
        super().__init__(agent_id)

    def _init_equ_method(self) -> Callable:
        if self.equ_type == "fpsb_symmetric_uniform":
            bid_function = self._get_fpsb_symmetric_uniform_equ()
        elif self.equ_type == "truthful":
            bid_function = self._get_truthful_equilibrium()
        else:
            raise ValueError("No valid equ_type selected - check: " + self.equ_type)
        return bid_function

    def _get_info_from_observation(
        self, observation: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Takes the agent observation and returns relevant info for equilibrium computation.
        Args:
            observation (torch.Tensor): 
        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]: stage, valuation, won
        """
        stage = self._obs2stage(observation)
        valuations = self._get_valuations_from_obs(observation)
        won = self._has_won_already_from_obs(observation)
        return stage, valuations, won

    def _obs2stage(self, observation: torch.Tensor) -> int:
        # NOTE: assumes all players and all games are in same stage
        if observation.shape[0] == 0:  # empty batch
            return -1
        if self.reduced_obs_space:
            return observation[0, 1].long().item()
        else:
            return (
                observation[0, self.payments_start_index :]
                .tolist()
                .index(self.dummy_price_key)
            )

    def _get_valuations_from_obs(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[
            :,
            self.valuations_start_index : self.valuations_start_index
            + self.valuation_size,
        ]

    def _has_won_already_from_obs(self, observation: torch.Tensor) -> torch.Tensor:
        """Check if the current player already has won in previous stages of the auction."""
        # NOTE: unit-demand hardcoded
        return observation[:, 2] > 0

    def _get_fpsb_symmetric_uniform_equ(self):
        def bid_function(observation: torch.Tensor):
            stage, valuation, won = self._get_info_from_observation(observation)
            bid = (
                (self.num_agents - self.num_units) / (self.num_agents - stage)
            ) * valuation

            if won is not None:
                bid[won, ...] = 0

            return bid.view(-1, 1)

        return bid_function

    def _get_truthful_equilibrium(self):
        def bid_function(observation: torch.Tensor):
            _, valuation, won = self._get_info_from_observation(observation)
            bid = valuation

            if won is not None:
                bid[won, ...] = 0

            return bid.view(-1, 1)

        return bid_function


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
    if num_agents != 4 or prior_low != 1.0 or prior_high != 1.5:
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

    if num_agents != 4 or prior_low != 1.0 or prior_high != 1.5:
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
