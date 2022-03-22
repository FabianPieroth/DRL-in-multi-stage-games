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

from src.envs.equilibria import equilibrium_fpsb_symmetric_uniform
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.utils import tensor_norm


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


class FirstPriceSealedBidAuction(Mechanism):
    """First Price Sealed Bid auction"""

    # def __init__(self, **kwargs):
    #     self.smoothing = .01
    #     super().__init__(**kwargs)

    # def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
    #     # TODO can we prevent non-positive bids easily?
    #     # assert (bids >= 0).all().item(), "All bids must be nonnegative."
    #     bids[bids < 0] = 0
    #     # rule_violations = (bids <= 0).any(axis=2)

    #     # name dimensions
    #     *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
    #     *batch_sizes, n_players, n_items = bids.shape

    #     payments = bids.clone()

    #     allocations = torch.nn.Softmax(dim=-2)(bids / self.smoothing)

    #     # Don't allocate items that have a winning bid of zero.
    #     allocations.masked_fill_(mask=bids < 0, value=0)
    #     payments.masked_fill_(mask=bids < 0, value=0)

    #     return (allocations, payments[:, :, 0])  # payments: batches x players, allocation: batch x players x items

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
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class SequentialFPSBAuction(BaseEnvForVec):
    """Sequential first price sealed bid auction.

    In each stage there is a single item sold.
    """

    DUMMY_PRICE_KEY = -1

    def __init__(self, config: Dict, device, player_position: int = 0):
        super().__init__(config, device)
        self.rl_env_config = config
        self.num_rounds_to_play = self.rl_env_config["num_rounds_to_play"]

        self.num_agents = self.rl_env_config["num_agents"]
        assert self.num_agents == 2, "Only two bidders supported so far."

        self.mechanism: Mechanism = FirstPriceSealedBidAuction()

        # unit-demand
        self.valuation_size = 1
        self.action_size = 1

        # observations: valuation, allocation, previous prices
        self.num_rounds_to_play
        self.observation_space = spaces.Box(
            low=np.array(
                [0]
                + [0] * self.num_rounds_to_play
                + [-1] * (self.num_rounds_to_play * 2)
            ),
            high=np.array(
                [1]
                + [1] * self.num_rounds_to_play
                + [np.inf] * (self.num_rounds_to_play * 2)
            ),
        )

        # actions
        self.action_space = spaces.Box(
            low=np.array([0] * self.action_size),
            high=np.array([np.inf] * self.action_size),
        )

        # positions
        self.player_position = player_position

        # dummy strategies: these can be overwritten by strategies that are
        # learned over time in repeated self-play.
        self.strategies = [
            lambda obs, deterministic=True: torch.zeros(
                (obs.shape[0], self.action_size), device=obs.device
            )
            for _ in range(self.num_agents)
        ]

        # Setup analytical BNE
        self.strategies_bne = [
            equilibrium_fpsb_symmetric_uniform(
                num_agents=self.num_agents,
                num_units=self.num_rounds_to_play,
                player_position=i,
            )
            for i in range(self.num_agents)
        ]

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

        # ipv symmetric unifrom priors
        states[:, :, : self.valuation_size].uniform_(0, 1)

        # dummy prices
        states[
            :, :, self.payments_start_index :
        ] = SequentialFPSBAuction.DUMMY_PRICE_KEY

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
        action_profile = actions.view(-1, 1, self.action_size).repeat(
            1, self.num_agents, 1
        )
        for opponent_position, opponent_strategy in enumerate(self.strategies):
            if opponent_position != self.player_position:
                opponent_obs = self.get_observations(
                    cur_states, player_position=opponent_position
                )
                action_profile[:, opponent_position, :] = opponent_strategy(
                    opponent_obs, deterministic=True
                ).view(-1, self.action_size)

        # run auction round
        allocations, payments = self.mechanism.run(action_profile)

        # states are for all agents, obs and rewards for `player_position` only
        new_states = cur_states.detach().clone()

        # get current stage
        try:
            stage = (
                cur_states[0, 0, self.payments_start_index :]
                .tolist()
                .index(SequentialFPSBAuction.DUMMY_PRICE_KEY)
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

    def _compute_rewards(self, state: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played auction games for the player at
        `self.player_position`.

        TODO: do we want intermediate rewards or not?
        """
        valuations = state[
            :,
            self.player_position,
            self.valuations_start_index : self.valuations_start_index
            + self.valuation_size,
        ]
        # only consider this stage's allocation
        allocations = state[
            :,
            self.player_position,
            self.allocations_start_index
            + stage * self.valuation_size : self.allocations_start_index
            + (stage + 1) * self.valuation_size,
        ]
        payments = state[
            :,
            self.player_position,
            self.payments_start_index + stage : self.payments_start_index + (stage + 1),
        ]

        # quasi-linear utility
        rewards = valuations * allocations - payments

        return rewards.view(-1)

    def get_observations(
        self, states: torch.Tensor, player_position: int = None
    ) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, state_dim).
        """
        if player_position is None:
            player_position = self.player_position

        # obs consits of: own valuations, own allocations, own payments and
        # published (here = highest payments)
        obs_private = states[:, player_position, :]
        obs_public = states[:, :, self.payments_start_index :].max(axis=1).values

        return torch.concat((obs_private, obs_public), axis=1)

    def render(self, state):
        return state

    def log_plotting(self, writer, step: int, n: int = 500):
        """Evaluate and log current strategies."""
        seed = 69

        fig, axs = plt.subplots(nrows=1, ncols=self.num_rounds_to_play, sharey=True)
        if self.num_rounds_to_play == 1:
            axs = [axs]

        self.seed(seed)
        states = self.sample_new_states(n)

        for stage, ax in zip(range(self.num_rounds_to_play), axs):
            ax.set_title(f"Stage {stage + 1}")
            for player_position in range(self.num_agents):
                self.player_position = player_position
                observations = self.get_observations(states)
                observations = observations.sort(axis=0)[0].view(-1, 4)

                # get actual actions
                try:
                    actions = self.strategies[player_position](
                        observations, deterministic=True
                    )
                    actions_mixed = self.strategies[player_position](
                        observations, deterministic=False
                    )
                except:
                    actions = self.strategies[player_position](observations)
                    actions_mixed = actions

                # get BNE actions
                actions_bne = self.strategies_bne[self.player_position](
                    stage, observations[:, 0]
                )

                # covert to numpy
                observations = observations[:, 0].cpu().view(-1).numpy()
                actions = actions.view(-1).detach().cpu().numpy()
                actions_mixed = actions_mixed.view(-1).detach().cpu().numpy()
                actions_bne = actions_bne.view(-1).detach().cpu().numpy()

                # plotting
                drawing, = ax.plot(
                    observations,
                    actions,
                    linestyle="--",
                    marker="o",
                    markevery=32,
                    label=f"bidder {player_position} PPO",
                )
                ax.plot(
                    observations,
                    actions_mixed,
                    ".",
                    alpha=0.3,
                    color=drawing.get_color(),
                )
                ax.plot(
                    observations,
                    actions_bne,
                    "--",
                    color=drawing.get_color(),
                    label=f"bidder {player_position} BNE",
                )
            lin = np.linspace(0, 1, 2)
            ax.plot(lin, lin, "--", color="grey", label="truthful")
            ax.set_xlabel("valuation $v$")
            if stage == 0:
                ax.set_ylabel("bid $b$")
                ax.legend(loc="upper left")
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.05, 0.55])

        plt.tight_layout()
        plt.close()
        # plt.savefig(f"{path}/plot_{step}.png")
        writer.add_figure("images", fig, step)

        # reset seed
        self.seed(int(time.time()))

    def log_vs_bne(self, logger, n: int = 100):
        """Evaluate learned strategies vs BNE."""
        seed = 69

        # calculate utility in self-play (learned strategies only)
        actual_utility = 0
        self.seed(seed)
        states = self.sample_new_states(n)
        for stage in range(self.num_rounds_to_play):
            observations = self.get_observations(states)
            try:
                actions_actual = self.strategies[self.player_position](
                    observations, deterministic=True
                )
            except:
                actions_actual = self.strategies[self.player_position](observations)
            observations, rewards, dones, states = self.compute_step(
                states, actions_actual
            )
        actual_utility += rewards.mean().item()
        logger.record("eval/utility_actual", actual_utility)

        # calculate the utility that the BNE strategy of the current player
        # would achieve
        bne_utility = 0
        self.seed(seed)
        states = self.sample_new_states(n)
        for stage in range(self.num_rounds_to_play):
            observations = self.get_observations(states)
            actions_bne = self.strategies_bne[self.player_position](
                stage, observations[:, 0]
            )
            observations, rewards, dones, states = self.compute_step(
                states, actions_bne
            )
        bne_utility += rewards.mean().item()
        logger.record("eval/utility_bne", bne_utility)

        # calculate distance in action space
        L2 = tensor_norm(actions_actual, actions_bne)
        logger.record("eval/action_norm_last_stage", L2)

        # reset seed
        self.seed(int(time.time()))
