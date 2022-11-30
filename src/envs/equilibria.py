"""Collection of known equilibria."""
import math
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

import src.utils.torch_utils as th_ut

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
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self.equ_method(observation), None


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


class SignalingContestEquilibrium(EquilibriumStrategy):
    def __init__(self, agent_id: int, config: Dict):
        self.device = config["device"]
        self.prior_low = config["prior_low"]
        self.prior_high = config["prior_high"]
        self.num_agents = config["num_agents"]
        self.information_case = config["information_case"]
        self.stage_index = config["stage_index"]
        self.allocation_index = config["allocation_index"]
        self.valuation_size = config["valuation_size"]
        self.payments_start_index = config["payments_start_index"]
        self.is_signaling_equ = self._is_signaling_equilibrium()
        self.first_round_equ_strategy = self._init_first_round_equ_strategy()
        self.inverse_first_round_equ_strategy = th_ut.torch_inverse_func(
            func=self.first_round_equ_strategy,
            domain=(self.prior_low, self.prior_high),
            device=self.device,
        )
        super().__init__(agent_id)
        if self.num_agents != 4 or self.prior_low != 1.0 or self.prior_high != 1.5:
            warnings.warn(
                "Only 2 agents per group, prior_low=1.0 and prior_high=1.5 is implemented!"
            )

    def _is_signaling_equilibrium(self) -> bool:
        if self.information_case == "true_valuations":
            is_signaling = False
        elif self.information_case == "winning_bids":
            is_signaling = True
        else:
            raise ValueError(
                "No valid equ_type selected - check: " + self.information_case
            )
        return is_signaling

    def _init_equ_method(self) -> Callable:
        return self._get_equilibrium()

    def _init_first_round_equ_strategy(self) -> Callable:
        if self.is_signaling_equ:

            def first_round_strategy(valuations: torch.Tensor) -> torch.Tensor:
                bid = self.signaling_effect_term(valuations) + self.winning_effect_term(
                    valuations
                )
                return bid

        else:

            def first_round_strategy(valuations: torch.Tensor) -> torch.Tensor:
                bid = self.winning_effect_term(valuations)
                return bid

        return first_round_strategy

    def _get_equilibrium(self):
        def bid_function(observation: torch.Tensor):
            stage, valuations, lost, opponent_vals = self._get_info_from_observation(
                observation
            )
            if stage == 1:
                bid = self.first_round_equ_strategy(valuations)
            elif stage == 2:
                bid = (valuations ** 2 * opponent_vals) / (
                    valuations + opponent_vals
                ) ** 2
            else:
                raise ValueError("Only two stage contest implemented!")

            if lost is not None:
                bid[lost, ...] = 0

            return RELU_LAYER(bid.view(-1, 1))

        return bid_function

    def _get_info_from_observation(
        self, observation: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Takes the agent observation and returns relevant info for equilibrium computation.
        Args:
            observation (torch.Tensor): 
        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]: stage, valuation, lost
        """

        stage = self._obs2stage(observation)
        valuations = self._get_valuations_from_obs(observation)
        lost = self._has_lost_already_from_obs(observation)
        opponent_vals = self._get_oppo_vals_from_obs(observation)
        return stage, valuations, lost, opponent_vals

    def _obs2stage(self, observation: torch.Tensor) -> int:
        """Get the current stage from the observation."""
        if observation.shape[0] == 0:  # empty batch
            stage = -1
        elif observation[0, self.stage_index].detach().item() == 0:
            stage = 1
        else:
            stage = 2
        return stage

    def _get_valuations_from_obs(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, : self.valuation_size]

    def _has_lost_already_from_obs(self, observation: torch.Tensor) -> torch.Tensor:
        """Check if player already has lost in previous round based on his or
        her observation.
        """
        allocation_true = observation[:, self.allocation_index] == 0
        could_have_lost = observation[:, self.stage_index] > 0
        lost_already = torch.logical_and(allocation_true, could_have_lost)
        return lost_already

    def _get_oppo_vals_from_obs(self, observation: torch.Tensor) -> torch.Tensor:
        opponent_info = observation[:, self.payments_start_index :]
        if self.is_signaling_equ:
            opponent_vals = self.inverse_first_round_equ_strategy(opponent_info)
        else:
            opponent_vals = opponent_info
        return opponent_vals

    @staticmethod
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

    @staticmethod
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
