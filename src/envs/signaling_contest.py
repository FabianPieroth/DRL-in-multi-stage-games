"""
Simple signaling contest with two stages and four player.
"""
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

import src.utils.policy_utils as pl_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import SignalingContestEquilibrium
from src.envs.mechanisms import AllPayAuction, Mechanism, TullockContest
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.utils import tensor_norm


class SignalingContest(BaseEnvForVec, VerifiableEnv):
    """Two Stage Contest with different information sets."""

    DUMMY_PRICE_KEY = -1
    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.valuation_size = config["valuation_size"]
        self.action_size = config["action_size"]
        self.prior_low, self.prior_high = config["prior_bounds"]
        self.ACTION_LOWER_BOUND, self.ACTION_UPPER_BOUND = 0, 2 * self.prior_high
        self.num_stages = 2
        # obs indices
        self.group_split_index = int(config["num_agents"] / 2)
        self.allocation_index = self.valuation_size
        self.stage_index = self.valuation_size + 1
        self.payments_start_index = self.valuation_size + self.allocation_index + 1
        self.relu_layer = torch.nn.ReLU()

        super().__init__(config, device)

        self.all_pay_mechanism = self._init_all_pay_mechanism()
        self.tullock_contest_mechanism = self._init_tullock_contest_mechanism()

    def _init_all_pay_mechanism(self) -> Mechanism:
        return AllPayAuction(self.device)

    def _init_tullock_contest_mechanism(self) -> Mechanism:
        impact_fun = lambda x: x ** self.config["impact_factor"]
        return TullockContest(impact_fun, self.device, self.config["use_valuation"])

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        equilibrium_config = {
            "device": self.device,
            "prior_low": self.prior_low,
            "prior_high": self.prior_high,
            "num_agents": self.num_agents,
            "information_case": self.config["information_case"],
            "stage_index": self.stage_index,
            "allocation_index": self.allocation_index,
            "valuation_size": self.valuation_size,
            "payments_start_index": self.payments_start_index,
        }
        return {
            agent_id: SignalingContestEquilibrium(agent_id, equilibrium_config)
            for agent_id in range(self.num_agents)
        }

    def _is_equilibrium_ensured_to_exist(self):
        if not (self.is_support_ratio_bounded() and self.does_min_density_bound_hold()):
            warnings.warn(
                "The sufficient conditions for a separating equilibrium do not hold! An equilibrium is not ensured to exist!"
            )

    def is_support_ratio_bounded(self) -> bool:
        ratio_support = self.prior_high / self.prior_low
        return 1.0 < ratio_support and ratio_support < 4.0 ** (1 / 3)

    def does_min_density_bound_hold(self) -> bool:
        min_density = 1.0 / (self.prior_high - self.prior_low)
        ratio_support = self.prior_high / self.prior_low
        density_factor = (self.num_agents / 2 - 1) * min_density * self.prior_low
        ratio_factor = max(
            (ratio_support - 1) * ratio_support ** 4 / 2,
            (ratio_support ** 2 - ratio_support)
            / (8.0 - 4.0 * ratio_support ** (3 / 2)),
        )
        return ratio_factor < density_factor

    def _get_num_agents(self) -> int:
        assert (
            self.config["num_agents"] % 2 == 0
        ), "The contest demands currently an even number of agents!"
        return self.config["num_agents"]

    def _init_observation_spaces(self):
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
                - valuation
                - win/loss
                - stage
                - bid/valuation of winning opponent
        """
        low = [self.prior_low] + [0.0] + [0.0] + [self.ACTION_LOWER_BOUND]
        high = [self.prior_high] + [1.0] + [1.0] + [self.ACTION_UPPER_BOUND]
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
            low=np.float32([self.ACTION_LOWER_BOUND] * self.config["action_size"]),
            high=np.float32([self.ACTION_UPPER_BOUND] * self.config["action_size"]),
        )
        return {agent_id: sa_action_space for agent_id in range(self.num_agents)}

    def to(self, device) -> Any:
        """Set device"""
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Samples number n initial states.
        [n, valuation + allocation + stage + winning bids + winning valuations]
        """
        states = torch.zeros(
            (n, self.num_agents, self.valuation_size + 1 + 1 + 2 * self.action_size),
            device=self.device,
        )

        # ipv symmetric uniform priors
        states[:, :, : self.valuation_size].uniform_(self.prior_low, self.prior_high)

        # no allocations
        states[:, :, self.valuation_size] = 0.0
        # First stage starting
        states[:, :, self.stage_index] = 0.0

        # dummy prices and winning valuations
        states[:, :, self.payments_start_index :] = SignalingContest.DUMMY_PRICE_KEY

        return states

    def adapt_ma_actions_for_env(
        self,
        ma_actions: Dict[int, torch.Tensor],
        states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        ma_actions = self.set_losers_bids_to_zero(states, ma_actions)
        ma_actions = self.clip_bids_to_positive(ma_actions)
        return ma_actions

    def clip_bids_to_positive(
        self, ma_actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        return {
            agent_id: self.relu_layer(sa_actions)
            for agent_id, sa_actions in ma_actions.items()
        }

    def set_losers_bids_to_zero(
        self, states, actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """NOTE: Theoretically agents may still win the second round even if they lost in the first
        but they still will not get any reward. It does skew the probabilities slightly for the first
        rounds winners though!"""

        # set previous rounds winners' bids to zero
        has_lost_already = self._has_lost_already_from_state(states)
        agent_ids = list(set(actions.keys()) & set(has_lost_already.keys()))
        for agent_id in agent_ids:
            actions[agent_id][has_lost_already[agent_id]] = 0.0
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
        actions = self.set_losers_bids_to_zero(cur_states, actions)
        action_profile = torch.stack(tuple(actions.values()), dim=1)
        new_states = cur_states.detach().clone()
        cur_stage = self._state2stage(cur_states)

        if cur_stage == 1:
            winning_info, allocations, payments = self._get_first_round_info(
                cur_states, action_profile
            )
            # store winning_info to new_states
            new_states[:, :, self.allocation_index] = winning_info[:, :, 0]
            new_states[:, :, self.stage_index] += 1.0
            new_states[:, :, self.payments_start_index :] = winning_info[:, :, 1:]
            dones = torch.zeros((cur_states.shape[0]), device=cur_states.device).bool()
        elif cur_stage == 2:
            allocations_prev_round = cur_states[
                :, :, self.allocation_index : self.allocation_index + 1
            ]
            aggregated_allocations = torch.sum(allocations_prev_round.squeeze(), axis=1)
            allocations = torch.zeros(
                (cur_states.shape[0], cur_states.shape[1], self.valuation_size),
                device=self.device,
            )
            payments = action_profile

            # Case 2: One winner in first round
            allocations[aggregated_allocations == 1] = allocations_prev_round[
                aggregated_allocations == 1
            ]

            # Case 3: Two winners in first round
            first_round_winner_indices = self._get_first_round_two_winner_indices(
                allocations_prev_round, aggregated_allocations
            )
            first_round_winner_bids = torch.gather(
                action_profile[aggregated_allocations == 2],
                dim=1,
                index=first_round_winner_indices,
            )
            sec_round_winning_probs, _ = self.tullock_contest_mechanism.run(
                first_round_winner_bids.detach().clone()
            )
            if self.config["sample_tullock_allocations"]:
                raise NotImplementedError("Enable sampling for winning allocations!")
            else:
                sec_round_allocations = sec_round_winning_probs
            allocations[aggregated_allocations == 2] = allocations[
                aggregated_allocations == 2
            ].scatter_(1, first_round_winner_indices, sec_round_allocations)
            dones = torch.ones((cur_states.shape[0]), device=cur_states.device).bool()
        else:
            raise ValueError("The setting only considers two stages at the moment!")

        rewards = self._compute_rewards(new_states, allocations, payments, cur_stage)

        observations = self.get_observations(new_states)

        return observations, rewards, dones, new_states

    def _get_first_round_two_winner_indices(
        self, allocations_prev_round, aggregated_allocations
    ):
        allocations_with_two_winners = allocations_prev_round[
            aggregated_allocations == 2
        ]
        first_rou_winner_indices = (allocations_with_two_winners).nonzero(
            as_tuple=True
        )[1]
        first_rou_winner_indices = torch.reshape(
            first_rou_winner_indices, ((allocations_with_two_winners).shape[0], 2)
        ).unsqueeze(-1)
        return first_rou_winner_indices

    def _get_first_round_info(self, cur_states, action_profile):
        w_info_A, allocations_A, payments_A = self._get_info_from_all_pay_auction(
            cur_states, 0, self.group_split_index, action_profile
        )
        w_info_B, allocations_B, payments_B = self._get_info_from_all_pay_auction(
            cur_states, self.group_split_index, self.num_agents, action_profile
        )
        wining_info = self._merge_group_winning_infos(
            w_info_A, allocations_A, w_info_B, allocations_B
        )
        return (
            wining_info,
            torch.concat([allocations_A, allocations_B], axis=1),
            torch.concat([payments_A, payments_B], axis=1),
        )

    def _merge_group_winning_infos(
        self,
        w_info_A: torch.Tensor,
        allocations_A: torch.Tensor,
        w_info_B: torch.Tensor,
        allocations_B: torch.Tensor,
    ) -> torch.Tensor:
        """Write opponent val/bid into info.

        Args:
            w_info_A (torch.Tensor): [num_envs, num_agents_group_A, (own) val/bid]
            allocations_A (torch.Tensor): [num_envs, num_agents_group_A, allocations]
            w_info_B (torch.Tensor): [num_envs, num_agents_group_A, (own) val/bid]
            allocations_B (torch.Tensor): [num_envs, num_agents_group_B, allocations]

        Returns:
            torch.Tensor: [num_envs, num_agents, allocation + opponent val/bid]
        """
        data_group_A = torch.concat([allocations_A, w_info_B], axis=2)
        data_group_B = torch.concat([allocations_B, w_info_A], axis=2)
        return torch.concat([data_group_A, data_group_B], axis=1)

    def provide_env_verifier_info(
        self, stage: int, agent_id: int, obs_discretization: int
    ) -> Tuple:
        discr_shapes = self._get_ver_obs_discretization_shape(obs_discretization, stage)
        obs_indices = self._get_ver_obs_dim_indices(stage)
        return discr_shapes, obs_indices

    def _get_ver_obs_discretization_shape(
        self, obs_discretization: int, stage: int
    ) -> Tuple[int]:
        """For the verifier, we return a discretized observation space."""
        if stage == 0:
            return (obs_discretization,)
        elif stage == 1:
            return (2, obs_discretization)
        else:
            raise ValueError("The contest is only implemented for two stages!")

    def _get_ver_obs_dim_indices(self, stage: int) -> Tuple[int]:
        if stage == 0:
            obs_indices = (0,)
        else:
            obs_indices = (1, 3)
        return obs_indices

    def get_ver_boundaries(
        self, stage: int, agent_id: int, obs_indices: Tuple[int]
    ) -> Tuple[float]:
        low = [
            self.observation_spaces[agent_id].low[obs_index]
            for obs_index in obs_indices
        ]
        high = [
            self.observation_spaces[agent_id].high[obs_index]
            for obs_index in obs_indices
        ]
        if stage == 1:
            if self.config["information_case"] == "true_valuations":
                low[-1], high[-1] = self.prior_low, self.prior_high
            elif self.config["information_case"] == "winning_bids":
                low[-1], high[-1] = 0.0, 0.5 * self.prior_high
            else:
                raise ValueError("Unknown information case!")
        return {"low": tuple(low), "high": tuple(high)}

    def _get_info_from_all_pay_auction(
        self, cur_states, low_split_index, high_split_index, action_profile
    ):
        sliced_action_profile = action_profile[:, low_split_index:high_split_index, :]
        group_allocations, group_payments = self.all_pay_mechanism.run(
            sliced_action_profile
        )
        winner_info = self._get_winning_information(
            group_allocations,
            sliced_action_profile,
            cur_states[:, low_split_index:high_split_index, :],
        )
        return winner_info, group_allocations, group_payments

    def _split_actions_into_first_round_groups(
        self, action_profile: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        action_profile_A = action_profile[:, : self.group_split_index, :]
        action_profile_B = action_profile[:, self.group_split_index :, :]
        return action_profile_A, action_profile_B

    def _get_winning_information(
        self,
        allocations: torch.Tensor,
        bids: torch.Tensor,
        rel_cur_states: torch.Tensor,
    ) -> torch.Tensor:
        winner_info = torch.zeros(
            (
                rel_cur_states.shape[0],
                rel_cur_states.shape[1],
                self.valuation_size + self.action_size,
            ),
            device=rel_cur_states.device,
        )
        winner_mask_ind_agent = (allocations == 1.0).squeeze()
        winner_mask_env = torch.any(winner_mask_ind_agent, axis=1)

        winner_info[:, :, 0][winner_mask_env] = (
            bids[winner_mask_ind_agent].detach().clone()
        )
        winner_info[:, :, 1][winner_mask_env] = rel_cur_states[
            :, :, : self.valuation_size
        ][winner_mask_ind_agent]
        return winner_info

    def _compute_rewards(
        self,
        states: torch.Tensor,
        allocations: torch.Tensor,
        payments: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        return {
            agent_id: self._compute_sa_rewards(
                states, allocations, payments, stage, agent_id
            )
            for agent_id in range(self.num_agents)
        }

    def _compute_sa_rewards(
        self,
        states: torch.Tensor,
        allocations: torch.Tensor,
        payments: torch.Tensor,
        stage: int,
        agent_id: int,
    ):
        sa_payments = payments[:, agent_id].squeeze()
        if stage == 1:
            rewards = -sa_payments
        else:
            sa_valuations = states[:, agent_id, : self.valuation_size].squeeze()
            sa_allocations = allocations[:, agent_id, :].squeeze()
            rewards = sa_valuations * sa_allocations - sa_payments
        return rewards

    def get_observations(
        self,
        states: torch.Tensor,
        player_positions: List = None,
        information_case: str = None,
    ) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim)
        """
        batch_size = states.shape[0]
        player_positions = (
            list(range(self.num_agents))
            if player_positions is None
            else player_positions
        )
        information_case = (
            self.config.information_case
            if information_case is None
            else information_case
        )

        observation_dict = {}
        for agent_id in player_positions:
            slicing_indices = self._get_obs_slicing_indices(information_case)
            # shape = (batch_size, valuation_size + allocation_index + 1 + action_size)
            observation_dict[agent_id] = (
                states[:, agent_id, :]
                .index_select(1, slicing_indices)
                .to(device=states.device)
                .detach()
            )

        return observation_dict

    def _get_obs_slicing_indices(self, information_case: str):
        if information_case == "true_valuations":
            slice_indices = [
                0,
                self.valuation_size,
                self.stage_index,
                self.valuation_size + self.allocation_index + 1 + self.action_size,
            ]
        elif information_case == "winning_bids":
            slice_indices = [
                0,
                self.valuation_size,
                self.stage_index,
                self.valuation_size + self.allocation_index + 1,
            ]
        else:
            raise ValueError
        indexing_tensor = torch.tensor(slice_indices, device=self.device)
        return indexing_tensor

    def render(self, state):
        return state

    def _state2stage(self, states):
        """Get the current stage from the state."""
        if states.shape[0] == 0:  # empty batch
            stage = -1
        elif states[0, 0, self.stage_index].detach().item() == 0:
            stage = 1
        else:
            stage = 2
        return stage

    def _obs2stage(self, obs):
        """Get the current stage from the observation."""
        if obs.shape[0] == 0:  # empty batch
            stage = -1
        elif obs[0, self.stage_index].detach().item() == 0:
            stage = 1
        else:
            stage = 2
        return stage

    def _has_lost_already_from_state(
        self, state: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Check if player already has lost in previous round based on
        current state.
        """
        lost_already_dict = {}
        for agent_id in range(self.num_agents):
            allocation_true = state[:, agent_id, self.allocation_index] == 0
            could_have_lost = state[:, agent_id, self.stage_index] > 0
            lost_already_dict[agent_id] = torch.logical_and(
                allocation_true, could_have_lost
            )
        return lost_already_dict

    def _has_lost_already_from_obs(
        self, obs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Check if player already has lost in previous round based on his or
        her observation.
        """
        lost_already_dict = {}
        for agent_id in obs.keys():
            allocation_true = obs[agent_id][:, self.allocation_index] == 0
            could_have_lost = obs[agent_id][:, self.stage_index] > 0
            lost_already_dict[agent_id] = torch.logical_and(
                allocation_true, could_have_lost
            )
        return lost_already_dict

    def custom_evaluation(self, learners, env, writer, iteration: int, config: Dict):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            iteration: current training iteration
        """
        self.plot_strategies_vs_equilibrium(learners, writer, iteration, config)

    def plot_strategies_vs_equilibrium(
        self, learners, writer, iteration: int, config, num_samples: int = 500
    ):
        """Evaluate and log current strategies."""
        seed = 69
        self.seed(seed)

        plt.style.use("ggplot")
        total_num_second_round_plots = 5
        ax_second_round_rotations = [
            (30, -60),
            (10, -100),
            (-10, 80),
            (30, 120),
            (30, 45),
        ]
        cmap = plt.get_cmap("gnuplot")
        agent_plot_colors = [cmap(i) for i in np.linspace(0, 1, self.num_agents)]
        plt.rcParams["figure.figsize"] = (8, 5.5)
        fig = plt.figure(
            figsize=plt.figaspect(1.0 + total_num_second_round_plots), dpi=600
        )
        ax_first_round = fig.add_subplot(1 + total_num_second_round_plots, 1, 1)
        ax_second_round_list = [
            fig.add_subplot(
                1 + total_num_second_round_plots, 1, 2 + plot_id, projection="3d"
            )
            for plot_id in range(total_num_second_round_plots)
        ]
        fig.suptitle(f"Iteration {iteration}", fontsize="x-large")
        states = self.sample_new_states(num_samples)

        for round in range(1, 3):
            observations = self.get_observations(states)
            ma_deterministic_actions = self.get_ma_actions_for_env(
                learners, observations=observations, deterministic=True, states=states
            )

            ma_stddevs = th_ut.get_ma_learner_stddevs(learners, observations)
            ma_stddevs = self.adapt_ma_actions_for_env(ma_stddevs, states=states)

            num_agents_to_plot = len(set(self.learners.values()))
            for agent_id in range(num_agents_to_plot):
                agent_obs = observations[agent_id]
                increasing_order = agent_obs[:, 0].sort(axis=0)[1]

                # sort
                agent_obs = agent_obs[increasing_order]

                has_lost_already = self._has_lost_already_from_obs(
                    {agent_id: agent_obs}
                )[agent_id]
                deterministic_actions = ma_deterministic_actions[agent_id][
                    increasing_order
                ]
                agent_stddevs = ma_stddevs[agent_id][increasing_order]

                # convert to numpy
                agent_vals = agent_obs[:, 0].detach().cpu().view(-1).numpy()
                opponent_info = agent_obs[:, 3].detach().cpu().view(-1).numpy()
                actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
                agent_stddevs = agent_stddevs.view(-1).detach().cpu().numpy()
                has_lost_already = has_lost_already.cpu().numpy()

                algo_name = pl_ut.get_algo_name(agent_id, config)

                if round == 1:
                    self._plot_first_round_strategy(
                        ax_first_round,
                        agent_id,
                        has_lost_already,
                        agent_stddevs,
                        agent_vals,
                        actions_array,
                        algo_name,
                        agent_plot_colors[agent_id],
                    )
                elif round == 2:
                    for ax_second_round, rotation in zip(
                        ax_second_round_list, ax_second_round_rotations
                    ):
                        ax_second_round.view_init(rotation[0], rotation[1])
                        ax_second_round.dist = 13
                        self._plot_second_round_strategy(
                            ax_second_round,
                            agent_id,
                            has_lost_already,
                            agent_vals,
                            opponent_info,
                            actions_array,
                            algo_name,
                            agent_plot_colors[agent_id],
                        )

            # apply actions to get to next stage
            _, _, _, states = self.compute_step(states, ma_deterministic_actions)

        handles, labels = ax_first_round.get_legend_handles_labels()
        ax_first_round.legend(handles, labels, ncol=2, prop={"size": 6})
        handles, labels = ax_second_round_list[0].get_legend_handles_labels()
        ax_second_round_list[0].legend(handles, labels, ncol=2, prop={"size": 3})
        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{iteration}.png")
        writer.add_figure("images", fig, iteration)
        plt.close()

        # reset seed
        self.seed(int(time.time()))

    def sufficient_points_to_plot3d(self, mask: np.ndarray) -> bool:
        return np.sum(mask) > 3

    def _plot_second_round_strategy(
        self,
        ax,
        agent_id,
        has_lost_already,
        agent_vals,
        opponent_info,
        actions_array,
        algo_name,
        color,
    ):
        ax.set_title("Second round")
        # check whether there are at least three points to plot in the mask!
        mask = np.logical_and(~has_lost_already, opponent_info != 0)
        if self.sufficient_points_to_plot3d(mask):
            surf = ax.plot_trisurf(
                agent_vals[mask],
                opponent_info[mask],
                actions_array[mask].squeeze(),
                linewidth=0.3,
                antialiased=True,
                alpha=0.5,
                color=color,
                label=f"bidder {agent_id} " + algo_name,
            )
            # ## due to bug in matplotlib ## #
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            # ############################## #
        mask = np.logical_and(has_lost_already, opponent_info != 0)
        if self.sufficient_points_to_plot3d(mask):
            surf = ax.plot_trisurf(
                agent_vals[mask],
                opponent_info[mask],
                actions_array[mask].squeeze(),
                linewidth=0.2,
                alpha=0.1,
                antialiased=True,
                color=color,
                label=f"bidder {agent_id} " + algo_name + " (lost)",
            )
            # ## due to bug in matplotlib ## #
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            # ############################## #
        if agent_id == 0:
            self._plot_second_round_equ_strategy_surface(ax, agent_id, 100)

        if self.config["information_case"] == "true_valuations":
            y_label = "opponent $v$"
        elif self.config["information_case"] == "winning_bids":
            y_label = "opponent $b$"
        else:
            raise ValueError
        ax.set_xlabel("valuation $v$", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel("bid $b$", fontsize=12)

    def _plot_second_round_equ_strategy_surface(self, ax, agent_id, plot_precision):
        val_x, opp_info, bid_z = self._get_actions_and_grid_in_second_stage(
            self.equilibrium_strategies[agent_id], plot_precision
        )
        surf = ax.plot_surface(
            val_x.cpu().numpy(), opp_info.cpu().numpy(), bid_z.cpu().numpy(), alpha=0.8
        )
        # ## due to bug in matplotlib ## #
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        # ############################## #s

    def _get_actions_and_grid_in_second_stage(
        self, sa_learner, precision: int, won_first_round: float = 1.0
    ):
        val_x, opp_info = self._get_meshgrid_for_second_round_equ(precision)
        # flatten mesh for forward
        val_x, opp_info = (
            val_x.reshape(precision ** 2),
            opp_info.reshape(precision ** 2),
        )
        sa_obs = self.get_obs_from_val_and_opp_info(
            val_x, opp_info, stage=1.0, won_first_round=won_first_round
        )
        bid_z, _ = sa_learner.predict(sa_obs, deterministic=True)
        bid_z = bid_z.reshape(precision, precision)
        val_x = val_x.reshape(precision, precision)
        opp_info = opp_info.reshape(precision, precision)
        return val_x, opp_info, bid_z

    def get_obs_from_val_and_opp_info(
        self, vals, opp_info, stage: float, won_first_round: float
    ):
        sa_obs = torch.zeros((vals.shape[0], 4), device=self.device)
        sa_obs[:, 0] = vals
        sa_obs[:, 1] = won_first_round
        sa_obs[:, 2] = stage
        sa_obs[:, 3] = opp_info
        return sa_obs

    def _get_meshgrid_for_second_round_equ(self, precision):
        val_xs = torch.linspace(self.prior_low, self.prior_high, steps=precision)
        if self.config["information_case"] == "true_valuations":
            info_ys = torch.linspace(self.prior_low, self.prior_high, steps=precision)
            val_x, opp_info = torch.meshgrid(val_xs, info_ys, indexing="xy")
        elif self.config["information_case"] == "winning_bids":
            info_ys = np.linspace(0.000001, 0.297682, num=precision)
            val_x, opp_info = torch.meshgrid(
                val_xs, torch.tensor(info_ys, dtype=torch.float32), indexing="xy"
            )

        else:
            raise ValueError()
        return val_x, opp_info

    def _plot_first_round_strategy(
        self,
        ax,
        agent_id,
        has_lost_already,
        ma_stddevs,
        agent_vals,
        actions_array,
        algo_name,
        agent_color,
    ):
        ax.set_title("First round")
        (drawing,) = ax.plot(
            agent_vals[~has_lost_already],
            actions_array[~has_lost_already],
            linestyle="-",
            label=f"bidder {agent_id} " + algo_name,
            color=agent_color,
        )
        ax.fill_between(
            agent_vals[~has_lost_already],
            (
                actions_array[~has_lost_already].squeeze()
                - ma_stddevs[~has_lost_already]
            ).clip(min=0),
            (
                actions_array[~has_lost_already].squeeze()
                + ma_stddevs[~has_lost_already]
            ).clip(min=0),
            alpha=0.2,
            color=drawing.get_color(),
        )
        self._plot_first_round_equilibrium_strategy(ax, agent_id, drawing)
        lin = np.linspace(0, self.prior_high, 2)
        ax.plot(lin, lin, "--", color="grey")
        ax.set_xlabel("valuation $v$")
        ax.set_ylabel("bid $b$")
        ax.set_xlim([self.prior_low - 0.1, self.prior_high + 0.1])

    def _plot_first_round_equilibrium_strategy(
        self, ax, agent_id: int, drawing, precision: int = 200
    ):
        val_xs, bid_ys = self._get_actions_and_grid_in_first_stage(
            self.equilibrium_strategies[agent_id], precision
        )
        ax.plot(
            val_xs.detach().cpu().numpy().squeeze(),
            bid_ys.squeeze().detach().cpu().numpy(),
            linestyle="--",
            color=drawing.get_color(),
            label=f"bidder {agent_id+1} equ",
        )

    def _get_actions_and_grid_in_first_stage(self, sa_learner, precision: int):
        val_xs = torch.linspace(
            self.prior_low, self.prior_high, steps=precision, device=self.device
        )
        opp_info = -1.0 * torch.ones_like(val_xs)
        sa_obs = self.get_obs_from_val_and_opp_info(
            val_xs, opp_info, stage=0.0, won_first_round=0.0
        )
        bid_ys, _ = sa_learner.predict(sa_obs, deterministic=True)
        return val_xs, bid_ys

    def plot_br_strategy(
        self, br_strategies: Dict[int, Dict[int, Callable]]
    ) -> Optional[plt.Figure]:
        num_vals = 128
        valuations = torch.linspace(
            self.prior_low, self.prior_high, num_vals, device=self.device
        )
        agent_obs = torch.cat(
            (valuations.unsqueeze(-1), torch.zeros((num_vals, 3), device=self.device)),
            dim=1,
        )
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(4.5, 4.5), clear=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel("valuation $v$")
        ax.set_ylabel("bid $b$")
        ax.set_xlim([self.prior_low, self.prior_high])
        ax.set_ylim([-0.05, self.prior_high * 1 / 3])

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
        return "SignalingContest"
