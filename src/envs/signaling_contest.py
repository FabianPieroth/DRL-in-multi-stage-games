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
from pynverse import inversefunc

import src.utils.policy_utils as pl_ut
from src.envs.equilibria import (
    SignalingContestEquilibrium,
    no_signaling_equilibrium,
    np_array_first_round_strategy,
    signaling_equilibrium,
)
from src.envs.mechanisms import AllPayAuction, Mechanism, TullockContest
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.utils import tensor_norm


class SignalingContest(BaseEnvForVec, VerifiableEnv):
    """Two Stage Contest with different information sets.
    """

    DUMMY_PRICE_KEY = -1
    OBSERVATION_DIM = 2
    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.valuation_size = config["valuation_size"]
        self.action_size = config["action_size"]
        self.prior_low, self.prior_high = config["prior_bounds"]
        self.ACTION_LOWER_BOUND, self.ACTION_UPPER_BOUND = 0, 2 * self.prior_high
        self.num_rounds_to_play = 2
        # obs indizes
        self.group_split_index = int(config["num_agents"] / 2)
        self.allocation_index = self.valuation_size
        self.stage_index = self.valuation_size + 1
        self.payments_start_index = self.valuation_size + self.allocation_index + 1

        super().__init__(config, device)

        self.all_pay_mechanism = self._init_all_pay_mechanism()
        self.tullock_contest_mechanism = self._init_tullock_contest_mechanism()

        self.equilibrium_strategies_deprecated = (
            self._init_equilibrium_strategies_deprecated()
        )

    def _init_all_pay_mechanism(self) -> Mechanism:
        return AllPayAuction(self.device)

    def _init_tullock_contest_mechanism(self) -> Mechanism:
        impact_fun = lambda x: x ** self.config["impact_factor"]
        return TullockContest(impact_fun, self.device, self.config["use_valuation"])

    def _init_equilibrium_strategies_deprecated(self):
        equilibrium_profile = self._get_equilibrium_profile(
            self.config["information_case"]
        )
        return {
            agent_id: equilibrium_profile(
                num_agents=self.num_agents,
                prior_low=self.prior_low,
                prior_high=self.prior_high,
            )
            for agent_id in range(self.num_agents)
        }

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        return {
            agent_id: self._get_agent_equilibrium_strategy(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _get_agent_equilibrium_strategy(
        self, agent_id: int
    ) -> SignalingContestEquilibrium:
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
        return SignalingContestEquilibrium(agent_id, equilibrium_config)

    def _get_equilibrium_profile(self, information_case: str):
        if information_case == "true_valuations":
            return no_signaling_equilibrium
        elif information_case == "winning_bids":
            self._is_equilibrium_ensured_to_exist()
            return signaling_equilibrium
        else:
            raise ValueError("No valid information case provided!")

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

    def get_obs_discretization_shape(
        self, agent_id: int, obs_discretization: int, stage: int
    ) -> Tuple[int]:
        """For the verifier, we return a discretized observation space."""
        if stage == 0:
            return (obs_discretization,)
        elif stage == 1:
            return (2, obs_discretization)
        else:
            raise ValueError("The contest is only implemented for two stages!")

    def get_obs_bin_indices(
        self,
        agent_obs: torch.Tensor,
        agent_id: int,
        stage: int,
        obs_discretization: int,
    ) -> torch.LongTensor:
        """Determines the bin indices for the given observations with discrete values between 0 and obs_discretization.

        Args:
            agent_obs (torch.Tensor): shape=(batch_size, obs_size)
            agent_id (int): 
            stage (int): 
            obs_discretization (int): number of discretization points

        Returns:
            torch.LongTensor: shape=(batch_size, )
        """
        if stage == 0:
            relevant_obs_indices = (0,)
            discretization_nums = (obs_discretization,)
        else:
            relevant_obs_indices = (1, 3)
            discretization_nums = (2, obs_discretization)
        obs_bins = torch.zeros(
            (agent_obs.shape[0], len(relevant_obs_indices)),
            dtype=torch.long,
            device=self.device,
        )
        for k, obs_dim in enumerate(relevant_obs_indices):
            obs_bins[:, k] = self._get_single_dim_obs_bins(
                agent_obs, agent_id, discretization_nums[k], obs_dim, stage
            )
        return obs_bins

    def _get_single_dim_obs_bins(
        self, agent_obs, agent_id, num_discretization, obs_dim, stage: int
    ) -> torch.LongTensor:
        low, high = self._get_bounds_for_obs_bins(agent_id, obs_dim, stage)
        obs_grid = torch.linspace(low, high, num_discretization, device=self.device)
        single_dim_obs_bins = torch.bucketize(agent_obs[:, obs_dim], obs_grid)
        return single_dim_obs_bins

    def _get_bounds_for_obs_bins(
        self, agent_id: int, obs_dim: int, stage: int
    ) -> Tuple[float]:
        low = self.observation_spaces[agent_id].low[obs_dim]
        high = self.observation_spaces[agent_id].high[obs_dim]
        if stage > 0 and obs_dim == 3:
            if self.config["information_case"] == "true_valuations":
                low, high = self.prior_low, self.prior_high
            elif self.config["information_case"] == "winning_bids":
                low, high = 0.0, 0.5 * self.prior_high
            else:
                raise ValueError("Unknown information case!")
        return low, high

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
        """Check if all players already have lost in previous round based on
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
        self.log_metrics_to_equilibrium(learners)

    def get_ma_equilibrium_actions(
        self, observations: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        equ_actions = {}
        player_positions = list(observations.keys())
        stage = self._obs2stage(observations[player_positions[0]])
        has_lost_already = self._has_lost_already_from_obs(observations)
        for agent_id, obs in observations.items():
            agent_vals, opponent_vals = obs[:, 0], obs[:, 3]
            equ_actions[agent_id] = self.equilibrium_strategies_deprecated[agent_id](
                stage, agent_vals, opponent_vals, has_lost_already[agent_id]
            )
        return equ_actions

    @staticmethod
    def get_ma_learner_predictions(
        learners,
        observations,
        deterministic: bool = True,
        clip_negativ_bids: bool = False,
    ):
        relu = torch.nn.ReLU()
        action_dict = {}
        for agent_id, obs in observations.items():
            sa_action_pred = learners[0].predict(obs, deterministic)[0]
            if clip_negativ_bids:
                sa_action_pred = relu(sa_action_pred)
            action_dict[agent_id] = sa_action_pred
        return action_dict

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
        ax_second_round_colors = [cmap(i) for i in np.linspace(0, 1, self.num_agents)]
        plt.rcParams["figure.figsize"] = (8, 5.5)
        fig = plt.figure(
            figsize=plt.figaspect(1.0 + total_num_second_round_plots), dpi=300
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
            ma_deterministic_actions = self.get_ma_learner_predictions(
                learners, observations, True
            )
            ma_mixed_actions = self.get_ma_learner_predictions(
                learners, observations, False
            )
            num_agents_to_plot = self.num_agents
            if self.config["plot_only_one_agent"]:
                num_agents_to_plot = 1
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
                mixed_actions = ma_mixed_actions[agent_id][increasing_order]

                # convert to numpy
                agent_vals = agent_obs[:, 0].detach().cpu().view(-1).numpy()
                opponent_info = agent_obs[:, 3].detach().cpu().view(-1).numpy()
                actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
                mixed_actions = mixed_actions.view(-1).detach().cpu().numpy()
                has_lost_already = has_lost_already.cpu().numpy()

                algo_name = pl_ut.get_algo_name(agent_id, config)

                if round == 1:
                    self._plot_first_round_strategy(
                        ax_first_round,
                        agent_id,
                        has_lost_already,
                        mixed_actions,
                        agent_vals,
                        actions_array,
                        algo_name,
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
                            ax_second_round_colors[agent_id],
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

        mask = np.logical_and(~has_lost_already, opponent_info != 0)
        ax.scatter(
            agent_vals[mask],
            opponent_info[mask],
            actions_array[mask],
            marker=".",
            color=color,
            label=f"bidder {agent_id} " + algo_name,
            s=8,
        )
        mask = np.logical_and(has_lost_already, opponent_info != 0)
        ax.scatter(
            agent_vals[mask],
            opponent_info[mask],
            actions_array[mask],
            marker="1",
            color=color,
            label=f"bidder {agent_id} " + algo_name + " (lost)",
            s=6,
        )
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
        val_x, info_y, bid_opponent_info = self._get_meshgrid_for_second_round_equ(
            plot_precision
        )
        bid_z = self.equilibrium_strategies_deprecated[agent_id](
            round=2, valuations=val_x, opponent_vals=bid_opponent_info, lost=None
        )
        bid_z = bid_z.reshape(plot_precision, plot_precision)
        ax.plot_surface(val_x.numpy(), info_y.numpy(), bid_z.numpy(), alpha=0.2)

    def _get_meshgrid_for_second_round_equ(self, plot_precision):
        val_xs = torch.linspace(self.prior_low, self.prior_high, steps=plot_precision)
        if self.config["information_case"] == "true_valuations":
            info_ys = torch.linspace(
                self.prior_low, self.prior_high, steps=plot_precision
            )
            val_x, info_y = torch.meshgrid(val_xs, info_ys, indexing="xy")
            bid_opponent_info = info_y
        elif self.config["information_case"] == "winning_bids":
            info_ys = np.linspace(0.000001, 0.297682, num=plot_precision)
            inverse_bids = inversefunc(
                np_array_first_round_strategy,
                y_values=info_ys,
                domain=[self.prior_low, self.prior_high],
            )
            inverse_bids = np.repeat(inverse_bids[:, None], plot_precision, axis=1)
            bid_opponent_info = torch.tensor(inverse_bids)

            val_x, info_y = torch.meshgrid(
                val_xs, torch.tensor(info_ys, dtype=torch.float32), indexing="xy"
            )
        else:
            raise ValueError()
        return val_x, info_y, bid_opponent_info

    def _plot_first_round_strategy(
        self,
        ax,
        agent_id,
        has_lost_already,
        mixed_actions,
        agent_vals,
        actions_array,
        algo_name,
    ):
        ax.set_title("First round")
        drawing, = ax.plot(
            agent_vals[~has_lost_already],
            actions_array[~has_lost_already],
            linestyle="dotted",
            marker="o",
            markevery=32,
            label=f"bidder {agent_id} " + algo_name,
        )
        ax.plot(
            agent_vals[~has_lost_already],
            mixed_actions[~has_lost_already],
            ".",
            alpha=0.2,
            color=drawing.get_color(),
        )
        self._plot_first_round_equilibrium_strategy(ax, agent_id)
        lin = np.linspace(0, self.prior_high, 2)
        ax.plot(lin, lin, "--", color="grey")
        ax.set_xlabel("valuation $v$")
        ax.set_ylabel("bid $b$")
        ax.set_xlim([self.prior_low - 0.1, self.prior_high + 0.1])

    def _plot_first_round_equilibrium_strategy(self, ax, agent_id):
        val_xs = torch.linspace(
            self.prior_low, self.prior_high, steps=100, device=self.device
        )
        bid_ys = self.equilibrium_strategies_deprecated[agent_id](
            round=1, valuations=val_xs, opponent_vals=None, lost=None
        )
        equ_xs = val_xs.detach().cpu().numpy().squeeze()
        equ_bid_y = bid_ys.detach().cpu().numpy().squeeze()
        ax.plot(equ_xs, equ_bid_y, linewidth=1)

    def log_metrics_to_equilibrium(self, learners, num_samples: int = 2 ** 16):
        """Evaluate learned strategies vs BNE."""
        seed = 69
        self.seed(seed)

        learned_utilities, equ_utilities, l2_distances = self.do_equilibrium_and_actual_rollout(
            learners, num_samples
        )

        self._log_metric_dict_to_individual_learners(
            learners, equ_utilities, "eval/utility_equilibrium"
        )
        self._log_metric_dict_to_individual_learners(
            learners, learned_utilities, "eval/utility_actual"
        )
        self._log_l2_distances(learners, l2_distances)

        # reset seed
        self.seed(int(time.time()))

    def do_equilibrium_and_actual_rollout(self, learners, num_samples: int):
        """Staring from state `states` we want to compute
            1. the action space L2 loss
            2. the rewards in actual play and in BNE
        Note that we need to keep track of counterfactual BNE states as these
        may be different from the states under actual play.
        """
        num_rounds = 2
        actual_states = self.sample_new_states(num_samples)
        actual_observations = self.get_observations(actual_states)
        equ_states = actual_states.clone()
        equ_observations = self.get_observations(equ_states)

        l2_distances = {i: [None] * num_rounds for i in learners.keys()}
        actual_rewards_total = {i: 0 for i in learners.keys()}
        equ_rewards_total = {i: 0 for i in learners.keys()}

        for round_iter in range(num_rounds):

            equ_actions_in_actual_play = self.get_ma_equilibrium_actions(
                actual_observations
            )

            equ_actions_in_equ = self.get_ma_equilibrium_actions(equ_observations)

            actual_actions = self.get_ma_learner_predictions(
                learners, actual_observations, True, clip_negativ_bids=True
            )

            actual_observations, actual_rewards, _, actual_states = self.compute_step(
                actual_states, actual_actions
            )
            equ_observations, equ_rewards, _, equ_states = self.compute_step(
                equ_states, equ_actions_in_equ
            )

            for agent_id in learners.keys():
                l2_distances[agent_id][round_iter] = tensor_norm(
                    actual_actions[agent_id], equ_actions_in_actual_play[agent_id]
                )

                actual_rewards_total[agent_id] += actual_rewards[agent_id].mean().item()
                equ_rewards_total[agent_id] += equ_rewards[agent_id].mean().item()

        return actual_rewards_total, equ_rewards_total, l2_distances

    def _log_l2_distances(self, learners, distances_l2):
        for round_iter in range(2):
            for agent_id, learner in learners.items():
                learner.logger.record(
                    "eval/action_equ_L2_distance_round_" + str(round_iter + 1),
                    distances_l2[agent_id][round_iter],
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
