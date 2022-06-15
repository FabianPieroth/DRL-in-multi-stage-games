"""
Simple sequential auction game following Krishna.

Single stage auction vendored from bnelearn [https://github.com/heidekrueger/bnelearn].
"""
import time
import warnings
from multiprocessing.sharedctypes import Value
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from pynverse import inversefunc

from src.envs.equilibria import (
    equilibrium_fpsb_symmetric_uniform,
    no_signaling_equilibrium,
    np_array_first_round_strategy,
    signaling_equilibrium,
)
from src.envs.mechanisms import AllPayAuction, Mechanism, TullockContest
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.utils import tensor_norm


class SignalingContest(BaseEnvForVec):
    """Two Stage Contest with different information sets.
    """

    DUMMY_PRICE_KEY = -1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.valuation_size = config["valuation_size"]
        self.action_size = config["action_size"]
        self.prior_low, self.prior_high = config["prior_bounds"]
        super().__init__(config, device)

        self.all_pay_mechanism = self._init_all_pay_mechanism()
        self.tullock_contest_mechanism = self._init_tullock_contest_mechanism()

        self.equilibrium_strategies = self._init_equilibrium_strategies()

    def _init_all_pay_mechanism(self) -> Mechanism:
        return AllPayAuction(self.device)

    def _init_tullock_contest_mechanism(self) -> Mechanism:
        impact_fun = lambda x: x ** self.config["impact_factor"]
        return TullockContest(impact_fun, self.device, self.config["use_valuation"])

    def _init_equilibrium_strategies(self):
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
                - win/loss/not-played first round
                - bid/valuation of winning opponent
        """
        low = [self.prior_low] + [-1.0] + [-1.0]
        high = [self.prior_high] + [1.0] + [np.inf]
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
        """Samples number n initial states. 
        [n, valuation + allocation + winning bids + winning valuations]
        """
        self.group_split_index = int(self.num_agents / 2)
        self.allocation_index = self.valuation_size
        self.payments_start_index = self.valuation_size + self.allocation_index
        states = torch.zeros(
            (n, self.num_agents, self.valuation_size + 1 + 2 * self.action_size),
            device=self.device,
        )

        # ipv symmetric uniform priors
        states[:, :, : self.valuation_size].uniform_(self.prior_low, self.prior_high)

        # No rounds played until now - no allocations
        states[:, :, self.valuation_size] = -1.0

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
            new_states[:, :, self.valuation_size :] = winning_info
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
        """if self.config["information_case"] == "true_valuations":
            winner_info.squeeze()[winner_mask_env] = rel_cur_states[
                :, :, : self.valuation_size
            ][winner_mask_ind_agent]
        elif self.config["information_case"] == "winning_bids":
            winner_info.squeeze()[winner_mask_env] = (
                bids[winner_mask_ind_agent].detach().clone()
            )
        else:
            raise ValueError("No valid information case provided!")"""
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

    def get_observations(self, states: torch.Tensor) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim)
        """
        batch_size = states.shape[0]
        observation_dict = {}
        for agent_id in range(self.num_agents):
            observation_size = (
                self.valuation_size + self.action_size + self.allocation_index
            )
            observation_dict[agent_id] = torch.zeros(
                (batch_size, observation_size), device=states.device
            )
            observation_dict[agent_id] = self._get_obs_info_from_state(
                agent_id, states, self.config["information_case"]
            ).detach()
        return observation_dict

    def _get_obs_info_from_state(
        self, agent_id: int, states: torch.Tensor, information_case: str
    ) -> torch.Tensor:
        slicing_indices = self._get_obs_slicing_indices(information_case)
        return states[:, agent_id, :].index_select(1, slicing_indices)

    def _get_obs_slicing_indices(self, information_case: str):
        if information_case == "true_valuations":
            slice_indices = [
                0,
                self.valuation_size,
                self.valuation_size + self.allocation_index + self.action_size,
            ]
        elif information_case == "winning_bids":
            slice_indices = [
                0,
                self.valuation_size,
                self.valuation_size + self.allocation_index,
            ]
        else:
            raise ValueError
        indexing_tensor = torch.Tensor(slice_indices).to(self.device).long()
        return indexing_tensor

    def render(self, state):
        return state

    def _state2stage(self, cur_states):
        """Get the current stage from the state."""
        if cur_states.shape[0] == 0:  # empty batch
            stage = -1
        elif cur_states[0, 0, self.allocation_index].detach().item() == -1.0:
            stage = 1
        else:
            stage = 2
        return stage

    def _has_lost_already(self, state: torch.Tensor):
        """Check if player already has lost in previous round."""
        # NOTE: unit-demand hardcoded
        return {
            agent_id: state[:, 0, self.allocation_index] == 0
            for agent_id in range(state.shape[1])
        }

    def custom_evaluation(self, learners, env, writer, iteration: int, config: Dict):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            iteration: current training iteration
        """
        pass
        self.plot_strategies_vs_equilibrium(learners, writer, iteration, config)
        self.log_metrics_to_equilibrium(learners)

    def get_ma_equilibrium_actions(
        self, round: int, states: torch.Tensor
    ) -> torch.Tensor:
        equ_actions = {}
        has_lost_already = self._has_lost_already(states)
        for agent_id in range(states.shape[1]):
            agent_true_val_info = self._get_obs_info_from_state(
                agent_id, states, "true_valuations"
            )
            agent_vals = agent_true_val_info[:, 0]
            opponent_vals = agent_true_val_info[:, 2]
            equ_actions[agent_id] = self.equilibrium_strategies[agent_id](
                round, agent_vals, opponent_vals, has_lost_already[agent_id]
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

                has_lost_already = self._has_lost_already(states[increasing_order])[
                    agent_id
                ]

                deterministic_actions = ma_deterministic_actions[agent_id][
                    increasing_order
                ]
                mixed_actions = ma_mixed_actions[agent_id][increasing_order]

                # convert to numpy
                agent_vals = agent_obs[:, 0].detach().cpu().view(-1).numpy()
                opponent_info = agent_obs[:, 2].detach().cpu().view(-1).numpy()
                actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
                mixed_actions = mixed_actions.view(-1).detach().cpu().numpy()
                has_lost_already = has_lost_already.cpu().numpy()

                if isinstance(config["algorithms"], str):
                    algo_name = config["algorithms"]
                else:
                    algo_name = config["algorithms"][agent_id]

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
        ax.scatter(
            agent_vals[~has_lost_already],
            opponent_info[~has_lost_already],
            actions_array[~has_lost_already],
            marker=".",
            color=color,
            label=f"bidder {agent_id} " + algo_name,
            s=8,
        )
        ax.scatter(
            agent_vals[has_lost_already],
            opponent_info[has_lost_already],
            actions_array[has_lost_already],
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
        bid_z = self.equilibrium_strategies[agent_id](
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
                np_array_first_round_strategy, y_values=info_ys, domain=[1.0, 1.5]
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
        bid_ys = self.equilibrium_strategies[agent_id](
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

        l2_distances = {i: [None] * num_rounds for i in learners.keys()}
        actual_rewards_total = {i: 0 for i in learners.keys()}
        equ_rewards_total = {i: 0 for i in learners.keys()}

        for round_iter in range(num_rounds):

            equ_actions_in_actual_play = self.get_ma_equilibrium_actions(
                round_iter + 1, actual_states
            )

            equ_actions_in_equ = self.get_ma_equilibrium_actions(
                round_iter + 1, equ_states
            )

            actual_actions = self.get_ma_learner_predictions(
                learners, actual_observations, True, clip_negativ_bids=True
            )

            actual_observations, actual_rewards, _, actual_states = self.compute_step(
                actual_states, actual_actions
            )
            _, equ_rewards, _, equ_states = self.compute_step(
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
