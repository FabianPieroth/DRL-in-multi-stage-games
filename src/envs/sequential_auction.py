"""
Simple sequential auction game following Krishna.

Single stage auction vendored from bnelearn [https://github.com/heidekrueger/bnelearn].
"""
import time
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

import src.utils.distributions_and_priors as dap_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import SequentialAuctionEquilibrium
from src.envs.mechanisms import FirstPriceAuction, Mechanism, VickreyAuction
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.utils import tensor_norm


class SequentialAuction(VerifiableEnv, BaseEnvForVec):
    """Sequential first price sealed bid auction.

    In each stage there is a single item sold.
    """

    DUMMY_PRICE_KEY = -1
    ACTION_LOWER_BOUND = 0.0
    ACTION_UPPER_BOUND = 1.1
    OBSERVATION_DIM = None
    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_rounds_to_play = config.num_rounds_to_play
        self.mechanism = self._init_mechanism(config)

        # NOTE: unit-demand only atm
        self.valuation_size = config.valuation_size
        self.signal_size = self.valuation_size
        self.action_size = config.action_size
        self.payments_start_index = self.get_payments_start_index()
        self.valuations_start_index = 0
        self.state_signal_start_index = (
            self.valuations_start_index + self.valuation_size
        )
        self.risk_aversion = config.risk_aversion
        self.sampler = self._init_sampler(config, device)

        # If the opponents are all symmetric, we may just sample from one
        # opponent with the max order statistic corresponding to the same
        # competition as the original opponents
        self.collapse_symmetric_opponents = config.collapse_symmetric_opponents

        super().__init__(config, device)

    def _init_mechanism(self, config):
        if config.mechanism_type == "first":
            mechanism: Mechanism = FirstPriceAuction()
        elif config.mechanism_type in ["second", "vcg", "vickery"]:
            mechanism: Mechanism = VickreyAuction()
        else:
            raise NotImplementedError("Payment rule unknown.")
        return mechanism

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        return {
            agent_id: self._get_agent_equilibrium_strategy(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _get_agent_equilibrium_strategy(
        self, agent_id: int
    ) -> SequentialAuctionEquilibrium:
        equilibrium_config = {
            "num_agents": self.config.num_agents,
            "num_units": self.num_rounds_to_play,
            "reduced_obs_space": self.reduced_observation_space,
            "payments_start_index": self.payments_start_index,
            "dummy_price_key": SequentialAuction.DUMMY_PRICE_KEY,
            "valuations_start_index": self.valuations_start_index,
            "valuation_size": self.valuation_size,
            "risk_aversion": self.risk_aversion,
        }
        if (
            self.config.mechanism_type == "first"
            and self.risk_aversion == 1.0
            and self.config.sampler.name == "symmetric_uniform"
        ):
            equilibrium_config["equ_type"] = "fpsb_symmetric_uniform"
        elif (
            self.config.mechanism_type == "first"
            and self.num_rounds_to_play == 1
            and self.config.sampler.name == "symmetric_uniform"
        ):
            equilibrium_config[
                "equ_type"
            ] = "fpsb_symmetric_uniform_single_stage_risk_averse"
        elif (
            self.config.mechanism_type in ["second", "vcg", "vickery"]
            and self.risk_aversion == 1.0
            and self.config.sampler.name == "symmetric_uniform"
        ):
            # TODO: @Nils: Is the equilibrium in second price also for risk? I would have thought it only to work for risk-neutral
            equilibrium_config["equ_type"] = "second_price_symmetric_uniform"
        elif (
            self.config.mechanism_type in ["second", "vcg", "vickery"]
            and self.num_rounds_to_play == 1
            and self.config.num_agents == 3
            and self.config.sampler.name == "mineral_rights_common_value"
        ):
            equilibrium_config["equ_type"] = "second_price_3p_mineral_rights_prior"
        elif (
            self.config.mechanism_type == "first"
            and self.num_rounds_to_play == 1
            and self.config.num_agents == 2
            and self.config.sampler.name == "affiliated_uniform"
        ):
            equilibrium_config["equ_type"] = "first_price_2p_affiliated_values_uniform"
        else:
            print("No analytical equilibrium available.")
            return None
        return SequentialAuctionEquilibrium(agent_id, equilibrium_config)

    # TODO: @Nils: Is this redundant?
    def _init_dummy_strategies(self):
        return [
            lambda obs, deterministic=True: torch.zeros(
                (obs.shape[0], self.config.action_size), device=obs.device
            )
            for _ in range(self.num_agents)
        ]

    def _get_num_agents(self) -> int:
        num_agents = self.config.num_agents
        # `num_actual_agents` may be larger than `num_agents` when we use
        # `collapse_symmetric_opponents`: Then `num_agents` will be lowered to
        # correspond to the sole learner.
        self.num_actual_agents = num_agents

        if self.collapse_symmetric_opponents:
            assert (
                self.config.sampler.name == "symmetric_uniform"
            ), "Collapse_opponents is currently hard-coded for uniform_symmetric prior"
            # Overwrite: external usage of this `BaseEnvForVec` should only
            # interact via a single learner.
            num_agents = 1
        return num_agents

    def _init_sampler(self, config, device):
        num_agents = config.num_agents
        if config.collapse_symmetric_opponents:
            num_agents = 2
        return dap_ut.get_sampler(
            num_agents,
            self.valuation_size,
            self.signal_size,
            config.sampler,
            default_device=device,
        )

    def _init_observation_spaces(self):
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """
        # NOTE: does not support non unit-demand
        self.reduced_observation_space = self.config.reduced_observation_space
        observation_spaces_dict = {}
        for agent_id in range(self.num_agents):
            val_low = self.sampler.support_bounds[agent_id, 0, 0].cpu().detach().item()
            val_high = self.sampler.support_bounds[agent_id, 0, 1].cpu().detach().item()
            if self.reduced_observation_space:
                # observations: valuation, stage, allocation (up to now)
                low = [val_low, 0.0, 0.0]
                high = [val_high, self.num_rounds_to_play, 1.0]
            else:
                # observations: valuation, allocation (including in which stage
                # obtained), previous prices (including in which stage payed)
                low = (
                    [val_low]
                    + [0.0] * self.num_rounds_to_play
                    + [-1.0] * (self.num_rounds_to_play * 2)
                )
                high = (
                    [val_high]
                    + [1.0] * self.num_rounds_to_play
                    + [val_high] * (self.num_rounds_to_play * 2)
                )
            self.OBSERVATION_DIM = len(low)
            observation_spaces_dict[agent_id] = spaces.Box(
                low=np.float32(low), high=np.float32(high)
            )
        return observation_spaces_dict

    def _init_action_spaces(self):
        """Returns dict with agent - action space pairs.
        Returns:
            Dict[int, Space]: agent_id: action space
        """
        action_spaces_dict = {}
        for agent_id in range(self.num_agents):
            val_low = self.sampler.support_bounds[agent_id, 0, 0].cpu().detach().item()
            val_high = self.sampler.support_bounds[agent_id, 0, 1].cpu().detach().item()
            action_spaces_dict[agent_id] = spaces.Box(
                low=np.float32([val_low] * self.config.action_size),
                high=np.float32([val_high] * self.config.action_size),
            )
        return action_spaces_dict

    def to(self, device) -> Any:
        """Set device"""
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states consisting of
            * one valuation per agent
            * one signal per agent
            * num_rounds_to_play * allocation per agent
            * prices of all stages (-1 for future stages)
                -> implicitly tells agents which the current round is

        :param n: Batch size of how many auction games are played in parallel.
        :return: The new states, in shape=(n, num_agents, -1), where the last
            dimension consists of the valuation, the allocations, and payments.
            Latter of which are kept track of over all stages.
        """
        self.allocations_start_index = self.valuation_size + self.signal_size
        # NOTE: We keep track of all (incl. zero) payments for reward calculations
        states = torch.zeros(
            (
                n,
                2 if self.collapse_symmetric_opponents else self.num_agents,
                self.payments_start_index + self.num_rounds_to_play,
            ),
            device=self.device,
        )

        # draw valuations and signals
        valuations, signals = self.sampler.draw_profiles(n)
        states[:, :, : self.valuation_size] = valuations
        states[
            :, :, self.valuation_size : self.valuation_size + self.signal_size
        ] = signals

        if self.collapse_symmetric_opponents:
            # The maximum of multiple uniform random variables follows a Beta
            # distribution
            m = torch.distributions.Beta(
                torch.tensor([self.num_actual_agents - 1], device=self.device),
                torch.tensor([1.0], device=self.device),
            )
            # NOTE: This samples from the valuations directly. In general, these may not be public to the agents.
            # If necessary, use signals instead.
            states[:, 1, : self.valuation_size] = m.sample((n,))
            states[
                :, 1, self.valuation_size : self.valuation_size + self.signal_size
            ] = states[:, 1, : self.valuation_size].clone()

        # dummy prices
        states[:, :, self.payments_start_index :] = SequentialAuction.DUMMY_PRICE_KEY

        return states

    def get_payments_start_index(self) -> int:
        """The payments start in die state after:
        valuations, signals, allocations for each round and item

        Returns:
            int: _description_
        """
        return (
            self.valuation_size
            + self.signal_size
            + self.valuation_size * self.num_rounds_to_play
        )

    def set_winners_bids_to_zero(
        self, states, actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """NOTE: We assume that all agents lose when they submit zero bids. If this is not the case, some agents may
        win again even though they won already in a previous round!"""
        # get current stage
        stage = self._state2stage(states)

        # set previous rounds winners' bids to zero
        has_won_already = self._has_won_already_from_state(states, stage)
        agent_ids = list(set(actions.keys()) & set(has_won_already.keys()))
        for agent_id in agent_ids:
            actions[agent_id][has_won_already[agent_id]] = 0.0
        return actions

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
        device = cur_states.device  # may differ from `self.device`
        player_positions_of_actions = set(actions.keys())

        if self.collapse_symmetric_opponents:
            # simulate and append opponent actions
            opponent_obs = self.get_observations(cur_states)[1]
            with torch.no_grad():
                opponent_actions = self.learners[0].policy._predict(
                    opponent_obs.to(self.learners[0].device)
                )
            actions[1] = opponent_actions

        actions = self.set_winners_bids_to_zero(cur_states, actions)

        action_profile = torch.stack(tuple(actions.values()), dim=1)

        # get current stage
        stage = self._state2stage(cur_states)

        # run auction round
        allocations, payments = self.mechanism.run(action_profile)

        # states are for all agents, obs and rewards for `player_position` only
        new_states = cur_states.detach().clone()

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

        if self.collapse_symmetric_opponents:
            # the only thing the current player knows is that the opponents
            # faced in the next round are weaker -> different Beta distribution
            # of opponents' bids
            highest_opponent = new_states[:, 1, : self.valuation_size]
            m = torch.distributions.Beta(
                torch.tensor(
                    [max(1, self.num_actual_agents - 2 - stage)], device=device
                ),
                torch.tensor([1.0], device=device),
            )
            batch_size = cur_states.shape[0]
            new_states[:, 1, : self.valuation_size] = highest_opponent * m.sample(
                (batch_size,)
            )
            new_states[
                :, 1, self.valuation_size : self.valuation_size + self.signal_size
            ] = new_states[:, 1, : self.valuation_size].clone()

            # force opponent's allocation to zero again s.t. it competes again
            # in next stage even if it has won already a good
            new_states[
                :,
                1,
                self.allocations_start_index
                + stage * self.valuation_size : self.allocations_start_index
                + (stage + 1) * self.valuation_size,
            ] = 0

        # reached last stage?
        if stage >= self.num_rounds_to_play - 1:
            dones = torch.ones((cur_states.shape[0]), device=device).bool()
        else:
            dones = torch.zeros((cur_states.shape[0]), device=device).bool()

        observations = {
            k: self.get_observations(new_states)[k] for k in player_positions_of_actions
        }

        rewards = self._compute_rewards(new_states, stage)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played auction games for the player at
        `self.player_position`.

        NOTE: We calculate intermediate rewards and not only returns once the
        game terminates.
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
        ]
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

        # quasi-linear utility
        payoff = valuations * allocations - payments

        # Handle risk aversion and the case for negative utilities
        rewards = (
            payoff.relu() ** self.risk_aversion - (-payoff).relu() ** self.risk_aversion
        )

        return rewards.view(-1)

    def get_observations(self, states: torch.Tensor) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim), where the private observations consist of the
            valuation and a vector of allocations and payments (for each stage)
            and the public observation consists of published prices.
        """
        num_agents = (
            self.num_agents + 1
            if self.collapse_symmetric_opponents
            else self.num_agents
        )

        if self.reduced_observation_space:
            # Observation consists of: own valuation, stage number, and own
            # allocations
            stage = self._state2stage(states)
            won = self._has_won_already_from_state(states, stage)
            batch_size = states.shape[0]
            observation_dict = {}
            for agent_id in range(num_agents):
                observation_dict[agent_id] = torch.zeros(
                    (batch_size, 3), device=states.device
                )
                observation_dict[agent_id][
                    :,
                    self.valuations_start_index : self.valuations_start_index
                    + self.valuation_size,
                ] = states[
                    :,
                    agent_id,
                    self.state_signal_start_index : self.state_signal_start_index
                    + self.signal_size,
                ]
                observation_dict[agent_id][:, 1] = stage
                observation_dict[agent_id][:, 2] = won[agent_id]

        else:
            # Observation consists of: own signal, own allocations, own
            # payments and published (here = highest payments)
            obs_public = states[:, :, self.payments_start_index :].max(axis=1).values
            observation_dict = {
                agent_id: torch.concat(
                    (
                        states[
                            :,
                            agent_id,
                            self.state_signal_start_index : self.state_signal_start_index
                            + self.signal_size,
                        ],
                        states[:, agent_id, self.allocations_start_index :],
                        obs_public,
                    ),
                    axis=1,
                )
                for agent_id in range(num_agents)
            }

        return observation_dict

    def render(self, state):
        return state

    def _state2stage(self, states):
        """Get the current stage from the state."""
        # NOTE: assumes all players and all games are in same stage
        if states.shape[0] == 0:  # empty batch
            return -1
        try:
            # NOTE: only works for fixed length / each batch at same stage
            stage = (
                states[0, 0, self.payments_start_index :]
                .tolist()
                .index(SequentialAuction.DUMMY_PRICE_KEY)
            )
        except ValueError as _:  # last round
            stage = self.num_rounds_to_play - 1
        return stage

    def get_obs_discretization_shape(
        self, agent_id: int, obs_discretization: int, stage: int
    ) -> Tuple[int]:
        """We only consider the agent's valuation space and loss/win."""
        if stage == 0:
            return (obs_discretization,)
        else:
            return (2,)

    def get_obs_bin_indices(
        self,
        agent_obs: torch.Tensor,
        agent_id: int,
        stage: int,
        obs_discretization: int,
    ) -> torch.LongTensor:
        """Determines the bin indices for the given observations with discrete
        values between 0 and obs_discretization.

        Args:
            agent_obs (torch.Tensor): shape=(batch_size, obs_size)
            agent_id (int): 
            stage (int): 
            obs_discretization (int): number of discretization points

        Returns:
            torch.LongTensor: shape=(batch_size, relevant_obs_size)
        """
        device = agent_obs.device
        if stage == 0:
            relevant_obs_indices = (0,)
            num_discretization = obs_discretization
        else:
            if self.reduced_observation_space:
                relevant_obs_indices = (2,)
            else:
                raise NotImplementedError(
                    "Needs to be handled differently. Win/lose is given per round here."
                )
                relevant_obs_indices = (stage,)
            num_discretization = 2
        relevant_new_stage_obs = agent_obs[:, relevant_obs_indices]
        low = self.observation_spaces[agent_id].low[relevant_obs_indices]
        high = self.observation_spaces[agent_id].high[relevant_obs_indices]
        obs_grid = torch.linspace(low, high, num_discretization, device=device)
        # TODO: Only works for one dimensional additional obs in every stage
        return torch.bucketize(relevant_new_stage_obs, obs_grid)

    def _has_won_already_from_state(
        self, state: torch.Tensor, stage: int
    ) -> Dict[int, torch.Tensor]:
        """Check if the current player already has won in previous stages of the auction."""
        num_agents = (
            self.num_agents + 1
            if self.collapse_symmetric_opponents
            else self.num_agents
        )

        # NOTE: unit-demand hardcoded
        low = self.allocations_start_index
        high = self.allocations_start_index + stage

        return {
            agent_id: state[:, agent_id, low:high].sum(axis=-1) > 0
            for agent_id in range(num_agents)
        }

    def custom_evaluation(self, learners, env, writer, iteration: int, config: Dict):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            iteration: current training iteration
        """
        self.plot_strategies_vs_bne(learners, writer, iteration, config)
        if self.equilibrium_strategies_known:
            self.log_metrics_to_equilibrium(learners)

    def plot_strategies_vs_bne(
        self, strategies, writer, iteration: int, config, num_samples: int = 2 ** 12
    ):
        """Evaluate and log current strategies."""
        seed = 69

        plt.style.use("ggplot")
        fig, axs = plt.subplots(
            nrows=1,
            ncols=self.num_rounds_to_play,
            sharey=True,
            figsize=(5 * self.num_rounds_to_play, 5),
        )
        fig.suptitle(f"Iteration {iteration}", fontsize="x-large")
        if self.num_rounds_to_play == 1:
            axs = [axs]

        self.seed(seed)
        states = self.sample_new_states(num_samples)

        for stage, ax in zip(range(self.num_rounds_to_play), axs):
            ax.set_title(f"Stage {stage + 1}")
            observations = self.get_observations(states)
            ma_deterministic_actions = th_ut.get_ma_actions(
                strategies, observations, True
            )
            ma_deterministic_actions = self.set_winners_bids_to_zero(
                states, ma_deterministic_actions
            )
            ma_stddevs = th_ut.get_ma_learner_stddevs(strategies, observations)
            ma_stddevs = self.set_winners_bids_to_zero(states, ma_stddevs)

            unique_strategies = set(strategies.values())
            # NOTE: This breaks under asymmetries and partial policy sharing
            # (like in an LLG auction)
            for agent_id, strategy in enumerate(unique_strategies):
                agent_obs = observations[agent_id]
                increasing_order = agent_obs[:, 0].sort(axis=0)[1]

                # sort
                agent_obs = agent_obs[increasing_order]

                has_won_already = self._has_won_already_from_state(
                    states[increasing_order], stage
                )[agent_id]

                # get actual actions
                deterministic_actions = ma_deterministic_actions[agent_id][
                    increasing_order
                ]
                stddevs = ma_stddevs[agent_id][increasing_order]

                # get BNE actions
                if self.equilibrium_strategies_known:
                    actions_bne = th_ut.get_ma_actions(
                        self.equilibrium_strategies, {agent_id: agent_obs}
                    )[agent_id]

                # covert to numpy
                agent_obs = agent_obs[:, 0].detach().cpu().view(-1).numpy()
                actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
                stddevs = stddevs.view(-1).detach().cpu().numpy()
                if self.equilibrium_strategies_known:
                    actions_bne = actions_bne.view(-1, 1).detach().cpu().numpy()
                has_won_already = has_won_already.cpu().numpy()

                # plotting
                drawing, = ax.plot(
                    agent_obs[~has_won_already],
                    actions_array[~has_won_already],
                    linestyle="-",
                    label=f"bidder {agent_id+1} {strategy}",
                )
                if self.equilibrium_strategies_known:
                    ax.plot(
                        agent_obs[~has_won_already],
                        actions_bne[~has_won_already],
                        linestyle="--",
                        color=drawing.get_color(),
                        label=f"bidder {agent_id+1} BNE",
                    )
                ax.fill_between(
                    agent_obs[~has_won_already],
                    (
                        actions_array[~has_won_already].squeeze()
                        - stddevs[~has_won_already]
                    ).clip(min=0),
                    (
                        actions_array[~has_won_already].squeeze()
                        + stddevs[~has_won_already]
                    ).clip(min=0),
                    alpha=0.2,
                    color=drawing.get_color(),
                )
                if stage > 0:
                    ax.plot(
                        agent_obs[has_won_already],
                        actions_array[has_won_already],
                        linestyle="dotted",
                        color=drawing.get_color(),
                        alpha=0.5,
                        label=f"bidder {agent_id+1} {strategy} (won)",
                    )
                    if self.equilibrium_strategies_known:
                        ax.plot(
                            agent_obs[has_won_already],
                            actions_bne[has_won_already],
                            linestyle="--",
                            color=drawing.get_color(),
                            alpha=0.5,
                            label=f"bidder {agent_id+1} BNE (won)",
                        )
                    ax.fill_between(
                        agent_obs[has_won_already],
                        (
                            actions_array[has_won_already].squeeze()
                            - stddevs[has_won_already]
                        ).clip(min=0),
                        (
                            actions_array[has_won_already].squeeze()
                            + stddevs[has_won_already]
                        ).clip(min=0),
                        alpha=0.1,
                        color=drawing.get_color(),
                    )
            # NOTE: We take support of prior for agent 0 to do the plots!
            lower_bound, upper_bound = (
                self.sampler.support_bounds[0, 0, 0].item(),
                self.sampler.support_bounds[0, 0, 1].item(),
            )
            lin = np.linspace(lower_bound, upper_bound, 2)
            ax.plot(lin, lin, "--", color="grey", alpha=0.5)
            ax.set_xlabel("valuation $v$")
            if stage == 0:
                ax.set_ylabel("bid $b$")
            ax.set_xlim([lower_bound, upper_bound])
            ax.set_ylim([lower_bound - 0.05, upper_bound + 0.05])

            # apply actions to get to next stage
            _, _, _, states = self.compute_step(states, ma_deterministic_actions)

        handles, labels = ax.get_legend_handles_labels()
        axs[0].legend(handles, labels, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{iteration}.png")
        writer.add_figure("images", fig, iteration)
        plt.close()

        # reset seed
        self.seed(int(time.time()))

    def log_metrics_to_equilibrium(self, strategies, num_samples: int = 4096):
        """Evaluate learned strategies vs BNE."""
        seed = 69
        self.seed(seed)

        learned_utilities, equ_utilities, l2_distances = self.do_equilibrium_and_actual_rollout(
            strategies, num_samples
        )

        self._log_metric_dict_to_individual_learners(
            strategies, equ_utilities, "eval/utility_equilibrium"
        )
        self._log_metric_dict_to_individual_learners(
            strategies, learned_utilities, "eval/utility_actual"
        )
        self._log_l2_distances(strategies, l2_distances)

        # reset seed
        self.seed(int(time.time()))

    def do_equilibrium_and_actual_rollout(self, learners, num_samples: int):
        """Staring from state `states` we want to compute
            1. the action space L2 loss
            2. the rewards in actual play and in BNE
        Note that we need to keep track of counterfactual BNE states as these
        may be different from the states under actual play.
        """
        actual_states = self.sample_new_states(num_samples)
        actual_observations = self.get_observations(actual_states)

        equ_states = actual_states.clone()
        equ_observations = self.get_observations(equ_states)

        l2_distances = {i: [None] * self.num_rounds_to_play for i in learners.keys()}
        actual_rewards_total = {i: 0 for i in learners.keys()}
        equ_rewards_total = {i: 0 for i in learners.keys()}

        for stage in range(self.num_rounds_to_play):

            equ_actions_in_actual_play = th_ut.get_ma_actions(
                self.equilibrium_strategies, actual_observations
            )
            equ_actions_in_equ = th_ut.get_ma_actions(
                self.equilibrium_strategies, equ_observations
            )
            actual_actions = th_ut.get_ma_actions(learners, actual_observations)
            actual_actions = self.set_winners_bids_to_zero(
                actual_states, actual_actions
            )

            actual_observations, actual_rewards, _, actual_states = self.compute_step(
                actual_states, actual_actions
            )
            equ_observations, equ_rewards, _, equ_states = self.compute_step(
                equ_states, equ_actions_in_equ
            )

            for agent_id in learners.keys():
                l2_distances[agent_id][stage] = tensor_norm(
                    actual_actions[agent_id], equ_actions_in_actual_play[agent_id]
                )

                actual_rewards_total[agent_id] += actual_rewards[agent_id].mean().item()
                equ_rewards_total[agent_id] += equ_rewards[agent_id].mean().item()

        return actual_rewards_total, equ_rewards_total, l2_distances

    def _log_l2_distances(self, learners, distances_l2):
        for stage in range(self.num_rounds_to_play):
            for agent_id, learner in learners.items():
                learner.logger.record(
                    "eval/action_equ_L2_distance_stage_" + str(stage),
                    distances_l2[agent_id][stage],
                )

    def _get_mix_equ_learned_actions(
        self, agent_id, ma_deterministic_learned_actions, ma_equilibrium_actions
    ):
        mixed_equ_learned_actions = {}
        for agent_idx in ma_deterministic_learned_actions.keys():
            if agent_idx == agent_id:
                mixed_equ_learned_actions[agent_idx] = ma_equilibrium_actions[agent_idx]
            else:
                mixed_equ_learned_actions[agent_idx] = ma_deterministic_learned_actions[
                    agent_idx
                ]
        return mixed_equ_learned_actions

    @staticmethod
    def _log_metric_dict_to_individual_learners(
        learners, metric_dict: Dict[int, float], key_prefix: str = ""
    ):
        for agent_id, learner in learners.items():
            learner.logger.record(key_prefix, metric_dict[agent_id])

    def plot_br_strategy(
        self, br_strategies: Dict[int, Callable]
    ) -> Optional[plt.Figure]:
        num_vals = 128
        valuations = torch.linspace(0.0, 1.0, num_vals, device=self.device)
        agent_obs = torch.cat(
            (valuations.unsqueeze(-1), torch.zeros((num_vals, 3), device=self.device)),
            dim=1,
        )
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(4.5, 4.5), clear=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel("valuation $v$")
        ax.set_ylabel("bid $b$")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])

        colors = [
            (0 / 255.0, 150 / 255.0, 196 / 255.0),
            (248 / 255.0, 118 / 255.0, 109 / 255.0),
            (150 / 255.0, 120 / 255.0, 170 / 255.0),
            (255 / 255.0, 215 / 255.0, 130 / 255.0),
        ]
        line_types = ["-", "--", "-.", ":"]

        for agent_id, agent_br in br_strategies.items():
            if agent_id > 3:
                color = (0 / 255.0, 150 / 255.0, 196 / 255.0)
                line_type = "-"
            else:
                color = colors[agent_id]
                line_type = line_types[agent_id]
            agent_actions = agent_br[0](agent_obs)
            xs = valuations.detach().cpu().numpy()
            ys = agent_actions.squeeze().detach().cpu().numpy()
            ax.plot(
                xs,
                ys,
                label="BR agent " + str(agent_id),
                linestyle=line_type,
                color=color,
            )
        plt.legend()
        ax.set_aspect(1)
        return fig

    def __str__(self):
        return "SequentialAuction"
