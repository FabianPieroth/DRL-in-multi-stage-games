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
        """Returns the equilibrium method for the respective agent.

        Returns:
            Callable:   equ_method(self, observation: torch.Tensor) -> actions: torch.Tensor:
                        observation.shape=[batch_size, obs_size]
                        actions.shape=[batch_size, action_size]
        """

    def predict(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self.equ_method(observation), None


class SequentialAuctionEquilibrium(EquilibriumStrategy):
    def __init__(self, agent_id: int, config: Dict):
        self.num_agents = config["num_agents"]
        self.num_units = config["num_units"]
        self.player_position = agent_id
        self.equ_type = config["equ_type"]
        self.obs_payments_start_index = (
            config["payments_start_index"] - config["valuation_size"]
        )
        self.reduced_obs_space = config["reduced_obs_space"]
        self.dummy_price_key = config["dummy_price_key"]
        self.valuations_start_index = config["valuations_start_index"]
        self.valuation_size = config["valuation_size"]
        self.risk_aversion = config["risk_aversion"]
        assert (
            self.num_agents > self.num_units
        ), "For this BNE, there must be more bidders than items."
        super().__init__(agent_id)

    def _init_equ_method(self) -> Callable:
        if self.equ_type == "fpsb_symmetric_uniform":
            bid_function = self._get_fpsb_symmetric_uniform_equ()
        elif self.equ_type == "fpsb_symmetric_uniform_single_stage_risk_averse":
            bid_function = (
                self._get_fpsb_symmetric_uniform_single_stage_risk_averse_equ()
            )
        elif self.equ_type == "second_price_symmetric_uniform":
            bid_function = self._get_second_price_symmetric_uniform_equ()
        elif self.equ_type == "second_price_3p_mineral_rights_prior":
            bid_function = self._get_bne_3p_mineral_rights_prior()
        elif self.equ_type == "first_price_2p_affiliated_values_uniform":
            bid_function = self._get_bne_2p_affiliated_values_prior()
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
                observation[0, self.obs_payments_start_index :]
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

    def _get_fpsb_symmetric_uniform_single_stage_risk_averse_equ(self):
        """BNE in the special case of a single-stage symmetric FPSB IPV auction
        where priors are symmetric uniform."""

        def bid_function(observation: torch.Tensor):
            stage, valuation, won = self._get_info_from_observation(observation)
            bid = (
                valuation
                * (self.num_agents - 1.0)
                / (self.num_agents - 1.0 + self.risk_aversion)
            )
            return bid.view(-1, 1)

        return bid_function

    def _get_second_price_symmetric_uniform_equ(self):
        """Surprisingly, the second price equilibrium is only truthful in the final stage!"""

        def bid_function(observation: torch.Tensor):
            stage, valuation, won = self._get_info_from_observation(observation)
            bid = (
                (self.num_agents - self.num_units) / (self.num_agents - stage - 1)
            ) * valuation

            if won is not None:
                bid[won, ...] = 0

            return bid.view(-1, 1)

        return bid_function

    def _get_bne_3p_mineral_rights_prior(self):
        """BNE in the 3-player 'Mineral Rights' setting.
        Reference: Krishna (2009), Example 6.1"""

        def bid_function(observation: torch.Tensor):
            stage, valuation, won = self._get_info_from_observation(observation)
            bid = (2 * valuation) / (2 + valuation)

            if won is not None:
                bid[won, ...] = 0

            return bid.view(-1, 1)

        return bid_function

    def _get_bne_2p_affiliated_values_prior(self):
        """Symmetric BNE in the 2p affiliated values model.
        Reference: Krishna (2009), Example 6.2"""

        def bid_function(observation: torch.Tensor):
            stage, noisy_valuation, won = self._get_info_from_observation(observation)
            bid = (2 / 3) * noisy_valuation

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


class BertrandCompetitionEquilibrium(EquilibriumStrategy):
    """We use the analytical equilibrium strategy from the paper by
    Arozamena and Weinschelbaum (2009), "Simultaneous vs. sequential
    price competition with incomplete information".
    """

    def __init__(self, agent_id: int, config: Dict):
        self.config = config
        self.device = config["device"]
        self.prior_low = config["prior_low"]
        self.prior_high = config["prior_high"]
        self.leader_inverse_first_round_strategy = (
            self._init_leader_inverse_first_round_strategy()
        )
        self.leader_first_round_equ_strategy = th_ut.torch_inverse_func(
            func=self.leader_inverse_first_round_strategy,
            domain=(self.prior_low, 2 * self.prior_high),
            device=self.device,
        )
        super().__init__(agent_id)

    def _init_leader_inverse_first_round_strategy(self) -> Callable:
        def leader_inverse_first_round_strategy(
            valuations: torch.Tensor
        ) -> torch.Tensor:
            bid = (
                -(10 - 6 * valuations - 4.5 * valuations ** 2 + 0.5 * valuations ** 3)
                / (5 + 10.5 * valuations - valuations ** 2 - 0.5 * valuations ** 3)
                + valuations
            )
            return bid

        return leader_inverse_first_round_strategy

    def _init_equ_method(self) -> Callable:
        if self.agent_id == 0:
            bid_function = self._get_leader_equilibrium()
        elif self.agent_id == 1:
            bid_function = self._get_follower_equilibrium()
        else:
            raise ValueError(
                "Only a single equilibrium implemented for exactly 2 agents!"
            )
        return bid_function

    def _get_leader_equilibrium(self) -> Callable:
        """See paper eq. (2)"""
        # F(b) = 0.5(b + b**2)
        # f(b) = 0.5 + b
        # Q(p) = 10 - p
        # q(p) = -p
        # b - phi1(b) = ((10 - b)(1 - 0.5*b - 0.5*b**2)) / ((10 - b)(0.5 + b) + b*(1 - 0.5*b - 0.5*b**2))
        def bid_function(observation: torch.Tensor):
            stage = self._obs2stage(observation)
            valuations = observation[:, 0]
            if stage == 0:
                bid = self.leader_first_round_equ_strategy(valuations.unsqueeze(-1))
            else:
                bid = torch.zeros(observation.shape[0], device=self.device)
            return bid.view(-1, 1)

        return bid_function

    def _get_follower_equilibrium(self) -> Callable:
        """Given b1, firm 2 has to match that bid to win. It will want to
            do so whenever b1 >= c2, and will thus set b2 = min{b1, pM(c2)},
            where pM(c2) is the monopoly price for unit cost c2. If b1 < c2,
            firm 2 will not match but rather set some price b2 > b1 so as to
            lose.
            """

        def bid_function(observation: torch.Tensor):
            stage = self._obs2stage(observation)
            if stage == 0:
                quote = torch.zeros(observation.shape[0], device=self.device)
            else:
                c2 = observation[:, 0]
                b1 = observation[:, 1]

                # NOTE: This is based on quantity Q(p) = 10 - p and profit
                # = Q(p) * (p - c)
                monopoly_price = 5 + c2 / 2

                # Match if b1 >= c2
                quote = torch.min(b1, monopoly_price)

                # If b1 < c2, set b2 > b1 so as to lose
                quote[b1 < c2] = c2[b1 < c2]
            return quote.view(-1, 1)

        return bid_function

    def _obs2stage(self, observation: torch.Tensor):
        """Get the current stage from the observation."""
        if observation.shape[0] == 0:  # empty batch
            return -1
        stage = 0 if observation[0, 1] == -1 else 1
        return stage
