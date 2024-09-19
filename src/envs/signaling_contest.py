"""
Simple signaling contest with two stages and four player.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from tensordict import TensorDict

import src.utils.distributions_and_priors as dap_ut
import src.utils.evaluation_utils as ev_ut
import src.utils.policy_utils as pl_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import SignalingContestEquilibrium
from src.envs.mechanisms import AllPayAuction, Mechanism, TullockContest
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv


class SignalingContest(BaseEnvForVec, VerifiableEnv):
    """Two-stage contest with different information sets following
    https://ideas.repec.org/p/qed/wpaper/1184.html:

    > This paper analyzes the signaling effect of bidding in a two-round
    > elimination contest. Before the final round, bids in the preliminary
    > round are revealed and act as signals of the contestants' private
    > valuations. Depending on his valuation, a incentive to bluff or sandbag
    > in the preliminary round in order to gain an advantage in the final
    > round. I analyze this signaling effect and characterize the equilibrium
    > in this game. Compared to the benchmark model, in which private
    > valuations are revealed automatically before the final round and thus no
    > signaling of bids takes place, I find that strong contestants bluff and
    > weak contestants sandbag. In a separating equilibrium, bids in the
    > preliminary round fully reveal the contestants' private valuations.
    > However, this signaling effect makes the equilibrium bidding strategy in
    > the preliminary round steeper for high valuations and flatter for low
    > valuations compared to the benchmark model.
    """

    DUMMY_PRICE_KEY = -1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.valuation_size = config["valuation_size"]
        self.num_stages = 2
        # obs indices
        self.group_split_index = int(config["num_agents"] / 2)
        self.allocation_index = self.valuation_size
        self.stage_index = self.valuation_size + 1
        self.payments_start_index = self.valuation_size + self.allocation_index + 1
        self.relu_layer = torch.nn.ReLU()

        self.cara_risk_aversion = config.cara_risk_aversion
        self.sampler = self._init_sampler(config, device)
        self.prior_low = self.sampler.support_bounds[:, :, 0].squeeze()
        self.prior_high = self.sampler.support_bounds[:, :, 1].squeeze()
        if config.sampler.name in ["mineral_rights_common_value", "affiliated_uniform"]:
            self.ACTION_UPPER_BOUNDS = self.prior_high
        else:
            self.ACTION_UPPER_BOUNDS = 2 * self.prior_high
        super().__init__(config, device)

        self.all_pay_mechanism = self._init_all_pay_mechanism()
        self.tullock_contest_mechanism = self._init_tullock_contest_mechanism()

    def _init_sampler(self, config, device):
        num_agents = config.num_agents
        return dap_ut.get_sampler(
            num_agents,
            self.valuation_size,
            self.valuation_size,
            config.sampler,
            default_device=device,
        )

    def _init_all_pay_mechanism(self) -> Mechanism:
        return AllPayAuction(self.device)

    def _init_tullock_contest_mechanism(self) -> Mechanism:
        impact_fun = lambda x: x ** self.config["impact_factor"]
        return TullockContest(impact_fun, self.device, self.config["use_valuation"])

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        equilibrium_config = {
            "device": self.device,
            "prior_low": self.prior_low[0].item(),
            "prior_high": self.prior_high[0].item(),
            "num_agents": self.num_agents,
            "information_case": self.config["information_case"],
            "stage_index": self.stage_index,
            "allocation_index": self.allocation_index,
            "valuation_size": self.valuation_size,
            "payments_start_index": self.payments_start_index,
        }
        if (
            self.sampler.sampler_config.name == "symmetric_uniform"
            and self.cara_risk_aversion == 0.0
        ):
            return {
                agent_id: SignalingContestEquilibrium(agent_id, equilibrium_config)
                for agent_id in range(self.num_agents)
            }
        else:
            print("No analytical equilibrium available.")
            return {agent_id: None for agent_id in range(self.num_agents)}

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
        obs_space_dict = {}
        for agent_id in range(self.num_agents):
            low = [self.prior_low[agent_id].item()] + [0.0] + [0.0] + [0.0]
            high = (
                [self.prior_high[agent_id].item()]
                + [1.0]
                + [1.0]
                + [max(self.ACTION_UPPER_BOUNDS).item()]
            )
            obs_space_dict[agent_id] = spaces.Box(
                low=np.float32(low), high=np.float32(high)
            )
        return obs_space_dict

    def _init_action_spaces(self):
        """Returns dict with agent - action space pairs.
        Returns:
            Dict[int, Space]: agent_id: action space
        """
        action_space_dict = {}
        for agent_id in range(self.num_agents):
            action_space_dict[agent_id] = spaces.Box(
                low=np.float32([0.0]),
                high=np.float32([self.ACTION_UPPER_BOUNDS[agent_id].item()]),
            )
        return action_space_dict

    def to(self, device) -> Any:
        """Set device"""
        self.device = device
        return self

    def sample_new_states(self, n: int) -> TensorDict:
        """Samples number n initial states.
        [n, num_agents, valuation + valuation_signal + allocation + stage + winning bids + winning valuations/signals/bids]
        """
        # draw valuations
        valuations, val_signals = self.sampler.draw_profiles(n)
        states = TensorDict(
            {
                "vals": valuations,
                "val_signals": val_signals,
                "allocation": torch.zeros((n, self.num_agents, 1), device=self.device),
                "stage": torch.zeros((n, self.num_agents, 1), device=self.device),
                "winning_bids": torch.ones((n, self.num_agents, 1), device=self.device)
                * SignalingContest.DUMMY_PRICE_KEY,
                "winners_info": torch.ones((n, self.num_agents, 1), device=self.device)
                * SignalingContest.DUMMY_PRICE_KEY,
            },
            batch_size=n,
            device=self.device,
        )
        return states

    def states_dict_to_old_tensor_format(
        self, states: TensorDict, signals_instead_of_vals=False
    ) -> torch.Tensor:
        """We adapt the information of the env by allowing for the agents to
        receive a signal about their valuation instead of their true valuation.
        This increases the state size.
        This method turns the state into the old format so that other methods
        can stay identical.

        Returns:
            torch.Tensor: [n, num_agents,
            valuation/signal + allocation + stage + winning bids + winning valuations]
        """
        signal_type = "vals"
        if signals_instead_of_vals:
            signal_type = "val_signals"
        keys_to_cat = (
            signal_type,
            "allocation",
            "stage",
            "winning_bids",
            "winners_info",
        )
        return torch.cat([states[key] for key in keys_to_cat], dim=-1)

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

        cur_states_old_format = self.states_dict_to_old_tensor_format(
            cur_states, signals_instead_of_vals=True
        )

        cur_stage = self._state2stage(cur_states)

        if cur_stage == 1:
            winning_info, allocations, payments = self._get_first_round_info(
                cur_states_old_format, action_profile
            )
            # store winning_info to new_states
            new_states["allocation"][:, :, 0] = winning_info[:, :, 0]
            new_states["stage"] += 1.0
            new_states["winning_bids"][:, :, 0] = winning_info[:, :, 1]
            new_states["winners_info"][:, :, 0] = winning_info[:, :, 2]
            dones = torch.zeros(
                (cur_states.batch_size), device=cur_states.device
            ).bool()
        elif cur_stage == 2:
            new_states["stage"] += 1.0
            allocations_prev_round = cur_states["allocation"]
            aggregated_allocations = torch.sum(allocations_prev_round.squeeze(), axis=1)
            allocations = torch.zeros(
                (
                    cur_states.batch_size[0],
                    allocations_prev_round.shape[1],
                    self.valuation_size,
                ),
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
            dones = torch.ones(cur_states.batch_size, device=cur_states.device).bool()
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
    ) -> Dict[str, Tuple[float]]:
        """Get the boundaries for the verifier."""
        low = [
            self.observation_spaces[agent_id].low[obs_index]
            for obs_index in obs_indices
        ]
        high = [
            self.observation_spaces[agent_id].high[obs_index]
            for obs_index in obs_indices
        ]
        if stage == 1:
            if self.config["information_case"] == "winners_signal":
                low[-1], high[-1] = (
                    min(self.prior_low).item(),
                    max(self.prior_high).item(),
                )
            elif self.config["information_case"] == "winning_bids":
                low[-1], high[-1] = 0.0, 0.5 * max(self.prior_high).item()
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
            (rel_cur_states.shape[0], rel_cur_states.shape[1], self.valuation_size + 1),
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
            sa_valuations = states["vals"][:, agent_id, :].squeeze()
            sa_allocations = allocations[:, agent_id, :].squeeze()
            rewards = sa_valuations * sa_allocations - sa_payments

        # We implement the CARA utility function for risk-averse bidders
        if self.cara_risk_aversion != 0.0:
            rewards = (
                1.0 - torch.exp(-self.cara_risk_aversion * rewards)
            ) / self.cara_risk_aversion
        return rewards

    def get_observations(
        self,
        states: torch.Tensor,
        player_positions: List = None,
        information_case: str = None,
    ) -> torch.Tensor:
        """Return the observations to the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim)
        """
        states_old_format = self.states_dict_to_old_tensor_format(
            states, signals_instead_of_vals=True
        )
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
            # shape = (batch_size, valuation_size + allocation_index + 1 + 1)
            observation_dict[agent_id] = (
                states_old_format[:, agent_id, :]
                .index_select(1, slicing_indices)
                .to(device=states_old_format.device)
                .detach()
            )

        return observation_dict

    def _get_obs_slicing_indices(self, information_case: str):
        if information_case == "winners_signal":
            slice_indices = [
                0,
                self.valuation_size,
                self.stage_index,
                self.valuation_size + self.allocation_index + 1 + 1,
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
        if states["stage"].shape[0] == 0:  # empty batch
            stage = -1
        elif states["stage"][0, 0, 0].detach().item() == 0:
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
            allocation_true = state["allocation"][:, agent_id, 0] == 0
            could_have_lost = state["stage"][:, agent_id, 0] > 0
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
        cmap = plt.get_cmap("gnuplot")
        agent_plot_colors = [cmap(i) for i in np.linspace(0, 1, self.num_agents)]

        ax_second_round_rotations = [
            (30, -60),
            (10, -100),
            (-10, 80),
            (30, 120),
            (30, 45),
        ]

        states_list, observations_list, actions_list, _ = ev_ut.run_algorithms(
            env=self,
            algorithms=learners,
            num_envs=num_samples,
            num_steps=self.num_stages,
            deterministic=True,
        )

        ma_stddevs_list = []
        for stage in range(self.num_stages):
            ma_stddevs = th_ut.get_ma_learner_stddevs(
                learners, observations_list[stage]
            )
            ma_stddevs = self.adapt_ma_actions_for_env(
                ma_stddevs, states=states_list[stage]
            )
            ma_stddevs_list.append(ma_stddevs)

        first_round_figure = self._plot_first_round_strategy(
            writer,
            iteration,
            config,
            ma_stddevs_list[0],
            states_list[0],
            observations_list[0],
            actions_list[0],
            agent_plot_colors,
        )

        second_round_figure_list = self._plot_second_round_strategy(
            writer,
            iteration,
            config,
            ma_stddevs_list[1],
            states_list[1],
            observations_list[1],
            actions_list[1],
            agent_plot_colors,
            ax_second_round_rotations,
        )

        # writer.add_figure("images", fig, iteration)
        plt.close()

    def sort_and_convert_to_numpy(
        self, ma_stddevs, observations, ma_deterministic_actions, agent_id
    ):
        agent_obs = observations[agent_id]
        increasing_order = agent_obs[:, 0].sort(axis=0)[1]

        # sort
        agent_obs = agent_obs[increasing_order]

        has_lost_already = self._has_lost_already_from_obs({agent_id: agent_obs})[
            agent_id
        ]
        deterministic_actions = ma_deterministic_actions[agent_id][increasing_order]
        agent_stddevs = ma_stddevs[agent_id][increasing_order]

        # convert to numpy
        agent_vals = agent_obs[:, 0].detach().cpu().view(-1).numpy()
        opponent_info = agent_obs[:, 3].detach().cpu().view(-1).numpy()
        actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
        agent_stddevs = agent_stddevs.view(-1).detach().cpu().numpy()
        has_lost_already = has_lost_already.cpu().numpy()
        return has_lost_already, agent_stddevs, agent_vals, opponent_info, actions_array

    def store_indvidual_plots(self, fig, log_dir, iteration, axes_list):
        for k, axis in enumerate(axes_list):
            extent = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig.savefig(
                f"{log_dir}/plot_{iteration}_axis_{k+1}.png", bbox_inches=extent
            )

    def sufficient_points_to_plot3d(self, mask: np.ndarray) -> bool:
        return np.sum(mask) > 3

    def _plot_second_round_strategy(
        self,
        writer,
        iteration,
        config,
        ma_stddevs,
        states,
        observations,
        ma_actions,
        agent_plot_colors,
        ax_second_round_rotations,
    ):
        second_round_figure_list = []
        plt.style.use("ggplot")
        for k, rotation in enumerate(ax_second_round_rotations):
            figure_plt = plt.figure(figsize=(4.5, 4.5), clear=True, dpi=600)
            ax = figure_plt.add_subplot(111, projection="3d")
            ax.view_init(rotation[0], rotation[1])
            ax.dist = 13
            num_agents_to_plot = len(set(self.learners.values()))
            for agent_id in range(num_agents_to_plot):
                (
                    has_lost_already,
                    agent_stddevs,
                    agent_vals,
                    opponent_info,
                    actions_array,
                ) = self.sort_and_convert_to_numpy(
                    ma_stddevs, observations, ma_actions, agent_id
                )

                algo_name = pl_ut.get_algo_name(agent_id, config)
                self._draw_second_round_agent_strategy_on_axis(
                    ax,
                    agent_id,
                    has_lost_already,
                    agent_vals,
                    opponent_info,
                    actions_array,
                    algo_name,
                    agent_plot_colors[agent_id],
                )

            ax.set_title("Round 2")
            ax.set_zlim([0.0 - 0.05, 0.45 + 0.05])
            if self.config["information_case"] == "winners_signal":
                if self.config.sampler.name in [
                    "mineral_rights_common_value",
                    "affiliated_uniform",
                ]:
                    y_label = "opponent $x$"
                else:
                    y_label = "opponent $v$"
            elif self.config["information_case"] == "winning_bids":
                y_label = "opponent $b$"
            else:
                raise ValueError(
                    f"No valid information case selected: {self.config['information_case']}!"
                )
            if self.config.sampler.name in [
                "mineral_rights_common_value",
                "affiliated_uniform",
            ]:
                ax.set_xlabel("observation $x$", fontsize=10)
            else:
                ax.set_xlabel("valuation $v$", fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_zlabel("bid $b$", fontsize=10)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, ncol=2, loc="best")

            figure_plt.tight_layout()
            figure_plt.savefig(f"{writer.log_dir}/{iteration}_second_round_{k}.png")

            second_round_figure_list.append(figure_plt)
        return second_round_figure_list

    def _draw_second_round_agent_strategy_on_axis(
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
        label = algo_name + f" contestant {agent_id}"
        if len(set(self.learners.values())) > 2:
            label = algo_name + f" {agent_id}"
        # check whether there are at least three points to plot in the mask!
        mask = np.logical_and(~has_lost_already, opponent_info != 0)
        if self.sufficient_points_to_plot3d(mask):
            surf = ax.plot_trisurf(
                agent_vals[mask],
                opponent_info[mask],
                actions_array[mask].squeeze(),
                linewidth=0.4,
                antialiased=True,
                alpha=0.7,
                color=color,
                label=label,
            )
            # ## due to bug in matplotlib ## #
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            # ############################## #
        if not self.config["prettify_plots"]:
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
                    label=label + " (lost)",
                )
                # ## due to bug in matplotlib ## #
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d
                # ############################## #
        if agent_id == 0 and self.equilibrium_strategies[agent_id]:
            self._plot_second_round_equ_strategy_surface(ax, agent_id, 100)

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
            val_x.reshape(precision**2),
            opp_info.reshape(precision**2),
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
        val_xs = torch.linspace(
            min(self.prior_low).item(), max(self.prior_high).item(), steps=precision
        )
        if self.config["information_case"] == "winners_signal":
            info_ys = torch.linspace(
                min(self.prior_low).item(), max(self.prior_high).item(), steps=precision
            )
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
        writer,
        iteration,
        config,
        ma_stddevs,
        states,
        observations,
        ma_actions,
        agent_plot_colors,
    ):
        plt.style.use("ggplot")
        figure_plt = plt.figure(figsize=(4.5, 4.5), clear=True, dpi=600)
        ax = figure_plt.add_subplot(111)
        num_agents_to_plot = len(set(self.learners.values()))
        for agent_id in range(num_agents_to_plot):
            (
                has_lost_already,
                agent_stddevs,
                agent_vals,
                opponent_info,
                actions_array,
            ) = self.sort_and_convert_to_numpy(
                ma_stddevs, observations, ma_actions, agent_id
            )

            algo_name = pl_ut.get_algo_name(agent_id, config)
            self._draw_first_round_agent_strategy_on_axis(
                ax,
                agent_id,
                has_lost_already,
                agent_stddevs,
                agent_vals,
                actions_array,
                algo_name,
                agent_plot_colors[agent_id],
            )

        ax.set_title("Round 1")
        if not self.config["prettify_plots"]:
            lin = np.linspace(0, max(self.prior_high).item(), 2)
            ax.plot(lin, lin, "--", color="grey")
        else:
            ax.set_ylim([0.0 - 0.05, 0.7 + 0.05])
        if self.config.sampler.name in [
            "mineral_rights_common_value",
            "affiliated_uniform",
        ]:
            ax.set_xlabel("observation $x$", fontsize=12)
        else:
            ax.set_xlabel("valuation $v$", fontsize=12)
        ax.set_ylabel("bid $b$")
        ax.set_xlim(
            [min(self.prior_low).item() - 0.1, max(self.prior_high).item() + 0.1]
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=2, loc="best")
        figure_plt.tight_layout()
        figure_plt.savefig(f"{writer.log_dir}/first_round_{iteration}.png")
        return figure_plt

    def _draw_first_round_agent_strategy_on_axis(
        self,
        ax,
        agent_id,
        has_lost_already,
        agent_stddevs,
        agent_vals,
        actions_array,
        algo_name,
        agent_color,
    ):
        label = algo_name + f" contestant {agent_id}"
        if len(set(self.learners.values())) > 2:
            label = algo_name + f" {agent_id}"
        (drawing,) = ax.plot(
            agent_vals[~has_lost_already],
            actions_array[~has_lost_already],
            linestyle="-",
            label=label,
            color=agent_color,
        )
        ax.fill_between(
            agent_vals[~has_lost_already],
            (
                actions_array[~has_lost_already].squeeze()
                - agent_stddevs[~has_lost_already]
            ).clip(min=0),
            (
                actions_array[~has_lost_already].squeeze()
                + agent_stddevs[~has_lost_already]
            ).clip(min=0),
            alpha=0.2,
            color=drawing.get_color(),
        )
        if self.equilibrium_strategies[agent_id]:
            self._plot_first_round_equilibrium_strategy(ax, agent_id, drawing, label)

    def _plot_first_round_equilibrium_strategy(
        self, ax, agent_id: int, drawing, label: str, precision: int = 200
    ):
        val_xs, bid_ys = self._get_actions_and_grid_in_first_stage(
            self.equilibrium_strategies[agent_id], precision
        )
        ax.plot(
            val_xs.detach().cpu().numpy().squeeze(),
            bid_ys.squeeze().detach().cpu().numpy(),
            linestyle="--",
            color=drawing.get_color(),
            label=label + " equ",
        )

    def _get_actions_and_grid_in_first_stage(self, sa_learner, precision: int):
        val_xs = torch.linspace(
            self.prior_low[sa_learner.agent_id].item(),
            self.prior_high[sa_learner.agent_id].item(),
            steps=precision,
            device=self.device,
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
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(4.5, 4.5), clear=True)
        ax = fig.add_subplot(111)
        if self.config.sampler.name in [
            "mineral_rights_common_value",
            "affiliated_uniform",
        ]:
            ax.set_xlabel("observation $x$", fontsize=12)
        else:
            ax.set_xlabel("valuation $v$", fontsize=12)
        ax.set_ylabel("bid $b$")
        ax.set_xlim([min(self.prior_low).item(), max(self.prior_high).item()])
        ax.set_ylim([-0.05, max(self.prior_high).item() * 1 / 3])

        colors = [
            (0 / 255.0, 150 / 255.0, 196 / 255.0),
            (248 / 255.0, 118 / 255.0, 109 / 255.0),
            (150 / 255.0, 120 / 255.0, 170 / 255.0),
            (255 / 255.0, 215 / 255.0, 130 / 255.0),
        ]
        line_types = ["-", "--", "-.", ":"]

        for agent_id, agent_br in br_strategies.items():
            valuations = torch.linspace(
                self.prior_low[agent_id].item(),
                self.prior_high[agent_id].item(),
                num_vals,
                device=self.device,
            )
            agent_obs = torch.cat(
                (
                    valuations.unsqueeze(-1),
                    torch.zeros((num_vals, 3), device=self.device),
                ),
                dim=1,
            )
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
