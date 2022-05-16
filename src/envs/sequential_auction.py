"""
Simple sequential auction game following Krishna.

Single stage auction vendored from bnelearn [https://github.com/heidekrueger/bnelearn].
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

from src.envs.equilibria import equilibrium_fpsb_symmetric_uniform, truthful
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.utils import batched_index_select, tensor_norm


class Mechanism(ABC):
    """
    Auction Mechanism - Interpreted as a Bayesian game. A Mechanism collects
    bids from all players, then allocates available items as well as payments
    for each of the players.
    """

    def play(
        self, action_profile, smooth_market: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for `run` method"""
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for play for auction mechanisms"""
        raise NotImplementedError()


class FirstPriceAuction(Mechanism):
    """First Price Sealed Bid auction"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO: If multiple players submit the highest bid, the implementation chooses the first rather than at random
    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) First Price Sealed Bid Auction.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert (
            bids.dim() >= 3
        ), "Bid tensor must be at least 3d (*batch_dims x players x items)"

        # TODO can we prevent non-positive bids easily?
        # assert (bids >= 0).all().item(), "All bids must be nonnegative."
        bids[bids < 0] = 0
        # rule_violations = (bids <= 0).any(axis=2)

        device = bids.device

        # name dimensions
        *batch_dims, player_dim, item_dim = range(
            bids.dim()
        )  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)

        highest_bids, winning_bidders = bids.max(
            dim=player_dim, keepdim=True
        )  # both shapes: [batch_sizes, 1, n_items]

        payments_per_item.scatter_(player_dim, winning_bidders, highest_bids)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)

        # Don't allocate items that have a winning bid of zero.
        allocations.masked_fill_(mask=payments_per_item <= 0, value=0)
        payments.masked_fill_(mask=payments < 0, value=0)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class VickreyAuction(Mechanism):
    "Vickrey / Second Price Sealed Bid Auctions"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Vickrey/Second Price Sealed Bid Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert (
            bids.dim() >= 3
        ), "Bid tensor must be at least 3d (*batch_dims x players x items)"

        # TODO can we prevent non-positive bids easily?
        # assert (bids >= 0).all().item(), "All bids must be nonnegative."
        bids[bids < 0] = 0

        # name dimensions
        *batch_dims, player_dim, item_dim = range(
            bids.dim()
        )  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(
            *batch_sizes, n_players, n_items, device=bids.device
        )
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=bids.device)

        highest_bids, winning_bidders = bids.max(
            dim=player_dim, keepdim=True
        )  # shape of each: [batch_size, 1, n_items]

        # getting the second prices --> price is the lowest of the two highest bids
        top2_bids, _ = bids.topk(2, dim=player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)

        payments_per_item.scatter_(player_dim, winning_bidders, second_prices)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)

        # Don't allocate items that have a winning bid of zero.
        allocations.masked_fill_(mask=payments_per_item < 0, value=0)
        payments.masked_fill_(mask=payments < 0, value=0)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class SequentialAuction(BaseEnvForVec):
    """Sequential first price sealed bid auction.

    In each stage there is a single item sold.
    """

    DUMMY_PRICE_KEY = -1

    def __init__(
        self,
        config: Dict,
        device: str = "cpu",
        player_position: int = 0,
        reduced_observation_space: bool = False,
    ):
        super().__init__(config, device)
        self.num_rounds_to_play = self.config["num_rounds_to_play"]

        # list of indices which maps `player_position` to its strategy index
        self.policy_symmetries = [0] * self.num_agents
        self.mechanism_type = config["mechanism_type"]

        # set up mechanism
        if self.mechanism_type == "first":
            self.mechanism: Mechanism = FirstPriceAuction()
            self.equilibrium_profile = equilibrium_fpsb_symmetric_uniform
        elif self.mechanism_type in ["second", "vcg", "vickery"]:
            self.mechanism: Mechanism = VickreyAuction()
            self.equilibrium_profile = truthful
        else:
            raise NotImplementedError("Payment rule unknown.")

        # unit-demand
        self.valuation_size = self.config["valuation_size"]
        self.action_size = self.config["action_size"]

        # set up observation space
        # NOTE: does not support non unit-demand

        # dummy strategies: these can be overwritten by strategies that are
        # learned over time in repeated self-play.
        self.strategies = [
            lambda obs, deterministic=True: torch.zeros(
                (obs.shape[0], self.config["action_size"]), device=obs.device
            )
            for _ in range(self.num_agents)
        ]

        # setup analytical BNE
        self.strategies_bne = [
            self.equilibrium_profile(
                num_agents=self.num_agents,
                num_units=self.num_rounds_to_play,
                player_position=i,
            )
            for i in range(self.num_agents)
        ]

    def _get_num_agents(self) -> int:
        return self.config["num_agents"]

    def _init_observation_spaces(self):
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """
        low = (
            [0.0]
            + [0.0] * self.config["num_rounds_to_play"]
            + [-1.0] * (self.config["num_rounds_to_play"] * 2)
        )
        high = (
            [1.0]
            + [1.0] * self.config["num_rounds_to_play"]
            + [np.inf] * (self.config["num_rounds_to_play"] * 2)
        )
        return {
            agent_id: spaces.Box(low=np.float32(low), high=np.float32(high))
            for agent_id in range(self.num_agents)
        }

    def _init_action_spaces(self):
        """Returns dict with agent - action space pairs.
        Returns:
            Dict[int, Space]: agent_id: action space
        """
        sa_action_space = spaces.Box(
            low=np.float32([0] * self.config["action_size"]),
            high=np.float32([np.inf] * self.config["action_size"]),
        )
        return {agent_id: sa_action_space for agent_id in range(self.num_agents)}

    def to(self, device) -> Any:
        """Set device"""
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states consiting of
            * one valuation per agent
            * num_rounds_to_play * allocation per agent
            * prices of all stages (-1 for future stages TODO?)
                -> implicitly tells agents which the current round is

        :param n: Batch size of how many auction games are played in parallel.        
        :return: The new states, in shape=(n, num_agents*2 + num_rounds_to_play),
            where ...
        `current_round` and `num_rounds_to_play`.
        """
        # TODO: perhaps it's easier to split state into multiple tensors?
        # -> needs special treatment in `torch_vec_env`
        self.valuations_start_index = 0
        self.allocations_start_index = self.valuation_size
        self.payments_start_index = (
            self.valuation_size + self.valuation_size * self.num_rounds_to_play
        )
        # NOTE: We keep track of all (incl. zero) payments for reward calculations
        states = torch.zeros(
            (n, self.num_agents, self.payments_start_index + self.num_rounds_to_play),
            device=self.device,
        )

        # ipv symmetric uniform priors
        states[:, :, : self.valuation_size].uniform_(0, 1)

        # dummy prices
        states[:, :, self.payments_start_index :] = SequentialAuction.DUMMY_PRICE_KEY

        return states

    def compute_step(self, cur_states, actions: torch.Tensor):
        """Compute a step in the game.

        :param cur_states: The current states of the games.
        :param actions: Actions that the active player at
            `self.player_position` is choosing.
        :return observations:
        :return rewards:
        :return episode-done markers:
        :return updated_states:
        """
        # append opponents' actions
        action_profile = torch.stack(tuple(actions.values()), dim=1)

        # run auction round
        allocations, payments = self.mechanism.run(action_profile)

        # states are for all agents, obs and rewards for `player_position` only
        new_states = cur_states.detach().clone()

        # get current stage
        try:
            stage = (
                cur_states[0, 0, self.payments_start_index :]
                .tolist()
                .index(SequentialAuction.DUMMY_PRICE_KEY)
            )
        except ValueError as _:  # last round
            stage = self.num_rounds_to_play - 1

        # update payments
        new_states[:, :, self.payments_start_index + stage] = payments

        # update allocations
        new_states[
            :,
            :,
            self.allocations_start_index
            + stage * self.valuation_size : self.allocations_start_index
            + (stage + 1) * self.valuation_size,
        ] = allocations

        # reached last stage?
        if stage >= self.num_rounds_to_play - 1:
            dones = torch.ones((cur_states.shape[0]), device=cur_states.device).bool()
        else:
            dones = torch.zeros((cur_states.shape[0]), device=cur_states.device).bool()

        observations = self.get_observations(new_states)

        rewards = self._compute_rewards(new_states, stage)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played auction games for the player at
        `self.player_position`.

        TODO: do we want intermediate rewards or not?
        """

        return {
            agent_id: self._compute_sa_rewards(states, stage, agent_id)
            for agent_id in range(self.num_agents)
        }

    def _compute_sa_rewards(self, states: torch.Tensor, stage: int, agent_id: int):
        valuations = states[
            :,
            agent_id,
            self.valuations_start_index : self.valuations_start_index
            + self.valuation_size,
        ].clone()
        # only consider this stage's allocation
        allocations = states[
            :,
            agent_id,
            self.allocations_start_index
            + stage * self.valuation_size : self.allocations_start_index
            + (stage + 1) * self.valuation_size,
        ]
        payments = states[
            :,
            agent_id,
            self.payments_start_index + stage : self.payments_start_index + (stage + 1),
        ]

        # set value to zero if we already own the unit
        # NOTE: unit-demand hardcoded
        if stage > 0:
            # sum over allocations of all previous stages
            onwer_mask = (
                states[
                    :,
                    agent_id,
                    self.allocations_start_index : self.allocations_start_index
                    + stage * self.valuation_size,
                ].sum(axis=1)
                > 0
            )
            valuations[onwer_mask] = 0

        # quasi-linear utility
        return torch.sum(valuations * allocations - payments, dim=1)

    def get_observations(self, states: torch.Tensor) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim), where the private observations consist of the
            valuation and a vector of allocations and payments (for each stage)
            and the public observation consits of published prices.
        """
        # obs consits of: own valuations, own allocations, own payments and
        # published (here = highest payments)
        obs_public = states[:, :, self.payments_start_index :].max(axis=1).values
        observation_dict = {
            agent_id: torch.concat((states[:, agent_id, :], obs_public), axis=1)
            for agent_id in range(self.num_agents)
        }
        return observation_dict

    def render(self, state):
        return state

    def log_plotting(self, writer, step: int, num_samples: int = 500):
        """Evaluate and log current strategies."""
        seed = 69

        plt.style.use("ggplot")
        fig, axs = plt.subplots(
            nrows=1,
            ncols=self.num_rounds_to_play,
            sharey=True,
            figsize=(5 * self.num_rounds_to_play, 5),
        )
        fig.suptitle(f"Iteration {step}", fontsize="x-large")
        if self.num_rounds_to_play == 1:
            axs = [axs]

        self.seed(seed)
        states = self.sample_new_states(num_samples)

        for stage, ax in zip(range(self.num_rounds_to_play), axs):
            ax.set_title(f"Stage {stage + 1}")
            for player_position in range(self.num_agents):
                self.player_position = player_position
                observations = self.get_observations(states)
                order = observations[:, 0].sort(axis=0)[1]

                # get actual actions
                actions = self.strategies[player_position](
                    observations, deterministic=True
                )
                actions_mixed = self.strategies[player_position](
                    observations, deterministic=False
                )

                # get BNE actions
                actions_bne = self.strategies_bne[self.player_position](
                    stage, observations[:, 0]
                )

                # covert to numpy
                observations = observations[order, 0].detach().cpu().view(-1).numpy()
                actions_array = actions.view(-1, 1)[order, ...].detach().cpu().numpy()
                actions_mixed = (
                    actions_mixed.view(-1)[order, ...].detach().cpu().numpy()
                )
                actions_bne = actions_bne.view(-1, 1)[order, ...].detach().cpu().numpy()

                # plotting
                if stage == 0:
                    drawing, = ax.plot(
                        observations,
                        actions_array,
                        linestyle="dotted",
                        marker="o",
                        markevery=32,
                        label=f"bidder {player_position} PPO",
                    )
                else:
                    has_won_already = (
                        (states[order, player_position, 1 : stage + 1].sum(axis=-1) > 0)
                        .cpu()
                        .numpy()
                    )
                    drawing, = ax.plot(
                        observations[~has_won_already],
                        actions_array[~has_won_already],
                        linestyle="dotted",
                        marker="o",
                        markevery=32,
                        label=f"bidder {player_position} PPO",
                    )
                    ax.plot(
                        observations[has_won_already],
                        actions_array[has_won_already],
                        linestyle="dotted",
                        marker="x",
                        markevery=32,
                        label=f"bidder {player_position} PPO (won)",
                        color=drawing.get_color(),
                    )
                ax.plot(
                    observations,
                    actions_mixed,
                    ".",
                    alpha=0.2,
                    color=drawing.get_color(),
                )
                ax.plot(
                    observations,
                    actions_bne,
                    linestyle="--",
                    marker="*",
                    markevery=32,
                    color=drawing.get_color(),
                    label=f"bidder {player_position} BNE",
                )
            lin = np.linspace(0, 1, 2)
            ax.plot(lin, lin, "--", color="grey")
            ax.set_xlabel("valuation $v$")
            if stage == 0:
                ax.set_ylabel("bid $b$")
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.05, 1.05])

            # apply actions to get to next stage
            _, _, _, states = self.compute_step(states, actions)

        handles, labels = ax.get_legend_handles_labels()
        axs[0].legend(handles, labels, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{step}.png")
        writer.add_figure("images", fig, step)
        plt.close()

        # reset seed
        self.seed(int(time.time()))

    def log_vs_bne(self, logger, num_samples: int = 100):
        """Evaluate learned strategies vs BNE."""
        # TODO: Currently not working for multi-stage: need cases?
        seed = 69

        # calculate utility in self-play (learned strategies only)
        actual_utility = 0
        self.seed(seed)
        states = self.sample_new_states(num_samples)
        observations = self.get_observations(states)
        for stage in range(self.num_rounds_to_play):
            actions_actual = self.strategies[self.player_position](
                observations, deterministic=True
            )
            observations, rewards, _, states = self.compute_step(states, actions_actual)
        actual_utility += rewards.mean().item()
        logger.record("eval/utility_actual", actual_utility)

        # calculate the utility that the BNE strategy of the current player
        # would achieve
        bne_utility = 0
        self.seed(seed)
        states = self.sample_new_states(num_samples)
        observations = self.get_observations(states)
        for stage in range(self.num_rounds_to_play):
            actions_bne = self.strategies_bne[self.player_position](
                stage, observations[:, 0]
            )
            observations, rewards, _, states = self.compute_step(states, actions_bne)
        bne_utility += rewards.mean().item()
        logger.record("eval/utility_bne", bne_utility)

        # calculate distance in action space
        L2 = tensor_norm(actions_actual, actions_bne)
        logger.record("eval/action_norm_last_stage", L2)

        # reset seed
        self.seed(int(time.time()))
