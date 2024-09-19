"""Bertrand competition"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from tensordict import TensorDict

import src.utils.distributions_and_priors as dap_ut
import src.utils.evaluation_utils as ev_ut
import src.utils.logging_write_utils as wr_ut
import src.utils.policy_utils as pl_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import BertrandCompetitionEquilibrium
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv


class BertrandCompetition(VerifiableEnv, BaseEnvForVec):
    """Bertrand competition as in https://doi.org/10.1016/j.econlet.2009.03.017

    ``We compare equilibria with sequential and simultaneous moves under
    homogeneous-good Bertrand competition when unit costs are private
    information. Under an alternative interpretation, we examine the
    consequences of awarding a right of first refusal in a first-price
    procurement auction with endogenous quantity.''
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_stages = 2
        self.valuation_size = 1
        self.observation_size = config["observation_size"]
        self.cara_risk_aversion = config.cara_risk_aversion
        self.sampler = self._init_sampler(config, device)
        self.relu_layer = torch.nn.ReLU()

        super().__init__(config, device)

    def _get_num_agents(self) -> int:
        assert (
            self.config["num_agents"] == 2
        ), "The Betrand Stackelberg game is only implemented for two agents!"
        return self.config["num_agents"]

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        return {
            agent_id: self._get_agent_equilibrium_strategy(agent_id)
            for agent_id in range(self.num_agents)
        }

    def get_prior_bounds(self, agent_id: int) -> Tuple[float]:
        val_low = self.sampler.support_bounds[agent_id, 0, 0].cpu().detach().item()
        val_high = self.sampler.support_bounds[agent_id, 0, 1].cpu().detach().item()
        return val_low, val_high

    def _get_agent_equilibrium_strategy(self, agent_id: int):
        val_low, val_high = self.get_prior_bounds(agent_id)
        equilibrium_config = {
            "num_agents": self.config.num_agents,
            "num_stages": self.num_stages,
            "prior_low": val_low,
            "prior_high": val_high,
            "device": self.device,
        }
        if self.cara_risk_aversion == 0.0 and self.config.sampler.name == "bertrand":
            return BertrandCompetitionEquilibrium(agent_id, equilibrium_config)
        else:
            print("No analytical equilibrium available.")
            return None

    def _init_sampler(self, config, device):
        return dap_ut.get_sampler(
            config.num_agents,
            self.valuation_size,
            self.observation_size,
            config.sampler,
            default_device=device,
        )

    def _init_observation_spaces(self):
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """
        observation_spaces_dict = {}
        for agent_id in range(self.num_agents):
            val_low = self.sampler.support_bounds[agent_id, 0, 0].cpu().detach().item()
            val_high = self.sampler.support_bounds[agent_id, 0, 1].cpu().detach().item()

            # observations: valuation, firm 1 quote(Agent 1) / stage-info (Agent 0)
            observation_spaces_dict[agent_id] = spaces.Box(
                low=np.float32([val_low, -1.0]),
                high=np.float32([val_high, 2 * val_high]),
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
            if self.config.sampler.name in [
                "mineral_rights_common_value",
                "affiliated_uniform",
            ]:
                action_spaces_dict[agent_id] = spaces.Box(
                    low=np.float32([val_low]), high=np.float32([2 * val_high])
                )
            else:
                action_spaces_dict[agent_id] = spaces.Box(
                    low=np.float32([val_low]), high=np.float32([1.3 * val_high])
                )
        return action_spaces_dict

    def to(self, device) -> Any:
        """Set device"""
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states consisting of one valuation per agent

        :param n: Batch size of how many auction games are played in parallel.
        :return: The new states, in shape=(n, num_agents, 2), where the last
            dimension consists of the valuation and the opponent firm's quote (initialized
            at -1).
        """
        # draw valuations and signals
        costs, cost_signals = self.sampler.draw_profiles(n)
        states = TensorDict(
            {
                "costs": costs,
                "cost_signals": cost_signals,
                "quoted_prices": -torch.ones(
                    (n, self.num_agents, 1), device=self.device
                )
                # keep 2nd entry to -1 as to detect which stage it is (firm 1 must
                # quote a value above 0.)
            },
            batch_size=n,
            device=self.device,
        )
        return states

    def compute_step(self, cur_states: torch.Tensor, actions: Dict[int, torch.Tensor]):
        """Compute a step in the game.

        :param cur_states: The current states of the games.
        :param actions: Actions that the active player at
            `self.player_position` is choosing.
        :return observations:
        :return rewards:
        :return episode-done markers:
        :return updated_states:
        """
        self.adapt_ma_actions_for_env(ma_actions=actions, states=cur_states)
        batch_size = cur_states.shape[0]

        # get current stage
        stage = self._state2stage(cur_states)

        new_states = cur_states.clone()

        # 1. stage: add firm 1's quotes to firm 2's info
        if stage == 0:
            new_states["quoted_prices"][:, 1, 0] = actions[0].squeeze()
            new_states["quoted_prices"][:, 0, 0] = 1  # Set stage info for agent 0
            rewards = {
                0: torch.zeros(batch_size, device=self.device),
                1: torch.zeros(batch_size, device=self.device),
            }
            dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 2. stage: firm 2's quotes to firm 1's info
        if stage == 1:
            new_states["quoted_prices"][:, 0, 0] = actions[1].squeeze()
            rewards = self._compute_rewards(new_states, stage)
            dones = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        observations = self.get_observations(new_states)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played competition."""
        firm1_wins = states["quoted_prices"][:, 1, 0] < states["quoted_prices"][:, 0, 0]
        quantity = 10 - states["quoted_prices"][:, :, 0].min(axis=1).values
        leader_reward = (
            firm1_wins
            * quantity
            * (states["quoted_prices"][:, 1, 0] - states["costs"][:, 0, 0])
        )
        follower_reward = (
            torch.logical_not(firm1_wins)
            * quantity
            * (states["quoted_prices"][:, 0, 0] - states["costs"][:, 1, 0])
        )
        leader_reward = self.apply_cara_risk_aversion(leader_reward)
        follower_reward = self.apply_cara_risk_aversion(follower_reward)
        return {
            0: leader_reward,
            1: follower_reward,
        }

    def apply_cara_risk_aversion(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.cara_risk_aversion != 0.0:
            rewards = (
                1.0 - torch.exp(-self.cara_risk_aversion * rewards)
            ) / self.cara_risk_aversion
        return rewards

    def get_observations(self, states: torch.Tensor) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :returns observations: Observations consisting of own valuation and
            other firm's quote.
        """
        obs_dict = {}
        for agent_id in range(self.num_agents):
            obs_dict[agent_id] = torch.cat(
                [
                    states["cost_signals"][:, agent_id, :].clone(),
                    states["quoted_prices"][:, agent_id, :].clone(),
                ],
                dim=-1,
            )
        return obs_dict

    def adapt_ma_actions_for_env(
        self,
        ma_actions: Dict[int, torch.Tensor],
        states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        ma_actions = self.set_non_active_bids_to_zero(states, ma_actions)
        return ma_actions

    def set_non_active_bids_to_zero(
        self, states: torch.Tensor, actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Improve learning signal by setting bids of non-active players to zero."""
        # get current stage
        stage = self._state2stage(states)
        if stage == 0:
            actions[1][:] *= 0.0
        else:
            actions[0][:] *= 0.0
        return actions

    def render(self, state):
        return state

    def _state2stage(self, states):
        """Get the current stage from the state."""
        if states.shape[0] == 0:  # empty batch
            return -1
        stage = 0 if states["quoted_prices"][0, 1, 0] == -1 else 1
        return stage

    def provide_env_verifier_info(
        self, stage: int, agent_id: int, obs_discretization: int
    ) -> Tuple:
        """Get the discretization shape and the indices of the observation."""
        discr_shapes = self._get_ver_obs_discretization_shape(
            obs_discretization, agent_id, stage
        )
        obs_indices = self._get_ver_obs_dim_indices(agent_id, stage)
        return discr_shapes, obs_indices

    def _get_ver_obs_discretization_shape(
        self, obs_discretization: int, agent_id: int, stage: int
    ) -> Tuple[int]:
        """We only consider the agent's valuation space and loss/win."""
        if agent_id == 0:
            if stage == 0:
                return (obs_discretization,)
            else:
                return (1,)
        else:
            if stage == 0:
                return (1,)
            else:
                return (obs_discretization, obs_discretization)

    def _get_ver_obs_dim_indices(self, agent_id: int, stage: int) -> Tuple[int]:
        """Get the indices of the observation that are considered for
        verification."""
        if agent_id == 0:
            obs_indices = (0,)
        else:
            if stage == 0:
                obs_indices = (0,)
            else:
                obs_indices = (0, 1)
        return obs_indices

    def get_ver_boundaries(
        self, stage: int, agent_id: int, obs_indices: Tuple[int]
    ) -> Dict[str, Tuple[float]]:
        """Use default, unless we are in stage 1 and consider
        the leader's bid in the observation for the follower.

        Args:
            stage (int): _description_
            agent_id (int): _description_
            obs_indices (Tuple[int]): _description_

        Returns:
            Dict[str, Tuple[float]]: _description_
        """
        low = tuple(
            [
                self.observation_spaces[agent_id].low[obs_index]
                for obs_index in obs_indices
            ]
        )
        high = tuple(
            [
                self.observation_spaces[agent_id].high[obs_index]
                for obs_index in obs_indices
            ]
        )
        if (
            stage == 1
            and agent_id == 1
            and self.config.sampler.name == "bertrand"
            and self.cara_risk_aversion == 0.0
        ):
            val_low, val_high = self.get_prior_bounds(agent_id)
            low = tuple([val_low, 0.4])
            high = tuple([val_high, 1.1])
        elif (
            stage == 1
            and agent_id == 1
            and self.config.sampler.name == "mineral_rights_common_value"
            and self.cara_risk_aversion == 0.0
        ):
            val_low, val_high = self.get_prior_bounds(agent_id)
            low = tuple([val_low, 0.8])
            high = tuple([val_high, 1.2])
        elif (
            stage == 1
            and agent_id == 1
            and self.config.sampler.name == "affiliated_uniform"
            and self.cara_risk_aversion == 0.0
        ):
            val_low, val_high = self.get_prior_bounds(agent_id)
            low = tuple([val_low, 1.2])
            high = tuple([val_high, 4.0])
        elif (
            stage == 1
            and agent_id == 1
            and self.config.sampler.name == "bertrand"
            and self.cara_risk_aversion != 0.0
        ):
            val_low, val_high = self.get_prior_bounds(agent_id)
            low = tuple([val_low, 0.0])
            high = tuple([val_high, 1.2])
        return {"low": low, "high": high}

    def clip_bids_to_positive(
        self, ma_actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        return {
            agent_id: self.relu_layer(sa_actions)
            for agent_id, sa_actions in ma_actions.items()
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

    def l2_loss_adaption_callback(
        self,
        states_list: List[Optional[Dict[int, torch.Tensor]]],
        observations_list: List[Dict[int, torch.Tensor]],
        equ_actions_list: List[Dict[int, torch.Tensor]],
        learner_actions_list: List[Dict[int, torch.Tensor]],
    ) -> Tuple[List[Dict[int, torch.Tensor]], List[Dict[int, torch.Tensor]]]:
        """The followers equilibrium strategy is not unique. If the leader's quote b1 is lower than
        the follower's costs c2, then any b2 > b1 is in equilibrium. Therefore, we adapt the
        actions so that the l2-distance in this case is zero.
        """
        follower_obs = observations_list[1][1]
        c2 = follower_obs[:, 0]
        b1 = follower_obs[:, 1]
        mask = torch.logical_and(b1 < c2, b1 < learner_actions_list[1][1].squeeze())
        # set follower's actions to 0.0
        equ_actions_list[1][1][mask.unsqueeze(-1)] = 0.0
        learner_actions_list[1][1][mask.unsqueeze(-1)] = 0.0
        return equ_actions_list, learner_actions_list

    def plot_strategies_vs_bne(
        self, learners, writer, iteration: int, config, num_samples: int = 2**10
    ):
        """Evaluate and log current strategies."""

        ax_second_round_rotations = [
            (30, -60),
            (10, -100),
            (-10, 80),
            (30, 120),
            (30, 45),
        ]
        agent_plot_colors = ["red", "blue"]

        states_list, observations_list, actions_list, _ = ev_ut.run_algorithms(
            env=self,
            algorithms=learners,
            num_envs=num_samples,
            num_steps=self.num_stages,
            deterministic=True,
        )

        agent_stddevs_list = []
        for stage in range(self.num_stages):
            ma_stddevs = th_ut.get_ma_learner_stddevs(
                learners, observations_list[stage]
            )
            ma_stddevs = self.adapt_ma_actions_for_env(
                ma_stddevs, states=states_list[stage]
            )
            ma_stddevs = self.clip_bids_to_positive(ma_stddevs)
            agent_stddevs_list.append(ma_stddevs)

        leader_figure = self._plot_first_round_strategy(
            writer,
            iteration,
            config,
            agent_stddevs_list[0][0],
            states_list[0],
            actions_list[0][0],
            agent_plot_colors[0],
        )

        follower_figures = self._plot_second_round_strategy(
            writer,
            iteration,
            ax_second_round_rotations,
            config,
            states_list[1],
            actions_list[1][1],
            agent_plot_colors[1],
        )
        plt.close()

    def _plot_first_round_strategy(
        self,
        writer,
        iteration,
        config,
        leader_stddevs,
        states,
        leader_actions,
        agent_color,
    ):
        plt.style.use("ggplot")
        figure_plt = plt.figure(figsize=(4.5, 4.5), clear=True, dpi=600)
        ax = figure_plt.add_subplot(111)
        self._draw_first_round_strategy_on_axis(
            ax, config, leader_stddevs, states, leader_actions, agent_color
        )

        ax.set_title("Bertrand Leader")
        if self.config.sampler.name in [
            "mineral_rights_common_value",
            "affiliated_uniform",
        ]:
            ax.set_xlabel("L's observation $x_1$", fontsize=12)
        else:
            ax.set_xlabel("L's cost $c_1$", fontsize=12)
        ax.set_ylabel("L's price $p_1$")
        val_low, val_high = self.get_prior_bounds(agent_id=0)
        ax.set_xlim([val_low - 0.1, val_high + 0.1])
        if (
            self.config["prettify_plots"]
            and self.config.sampler.name == "mineral_rights_common_value"
        ):
            ax.set_ylim(0.4 - 0.05, 1.1 + 0.05)
        elif self.config["prettify_plots"] and self.cara_risk_aversion > 0.0:
            ax.set_ylim(0.15 - 0.05, 1.1 + 0.05)
        ax.legend(loc="best")

        figure_plt.tight_layout()
        figure_plt.savefig(f"{writer.log_dir}/{iteration}_leader.png")
        return figure_plt

    def _draw_first_round_strategy_on_axis(
        self, ax, config, leader_stddevs, states, leader_actions, agent_color
    ):
        leader_cost_signals = states["cost_signals"][:, 0, 0]
        sorted_leader_cost_signals, increasing_order = leader_cost_signals.sort(axis=0)
        sorted_leader_cost_signals = (
            sorted_leader_cost_signals.detach().cpu().view(-1).numpy()
        )
        sorted_leader_actions = (
            leader_actions[increasing_order].detach().cpu().view(-1).numpy()
        )
        sorted_leader_stddevs = (
            leader_stddevs[increasing_order].detach().cpu().view(-1).numpy()
        )
        algo_name = pl_ut.get_algo_name(0, config)
        (drawing,) = ax.plot(
            sorted_leader_cost_signals,
            sorted_leader_actions,
            linestyle="-",
            label=algo_name + " leader",
            color=agent_color,
        )
        ax.fill_between(
            sorted_leader_cost_signals,
            (sorted_leader_actions.squeeze() - sorted_leader_stddevs).clip(min=0),
            (sorted_leader_actions.squeeze() + sorted_leader_stddevs).clip(min=0),
            alpha=0.2,
            color=drawing.get_color(),
        )
        self._plot_first_round_equilibrium_strategy(ax, drawing)

    def _plot_first_round_equilibrium_strategy(
        self, plot_axis, drawing, precision: int = 200
    ):
        if self.equilibrium_strategies[0] is not None:
            leader_vals, leader_actions = self._get_actions_and_grid_in_first_stage(
                self.equilibrium_strategies[0], precision
            )
            plot_axis.plot(
                leader_vals.squeeze().detach().cpu().numpy(),
                leader_actions.squeeze().detach().cpu().numpy(),
                linestyle="--",
                color=drawing.get_color(),
                label=f"Leader equ",
            )

    def _get_actions_and_grid_in_first_stage(self, sa_learner, precision: int):
        val_low, val_high = self.get_prior_bounds(agent_id=0)
        val_xs = torch.linspace(val_low, val_high, steps=precision, device=self.device)
        opp_info = -1.0 * torch.ones_like(val_xs)
        sa_obs = torch.stack((val_xs, opp_info), dim=-1)
        bid_ys, _ = sa_learner.predict(sa_obs, deterministic=True)
        return val_xs, bid_ys

    def _plot_second_round_strategy(
        self,
        writer,
        iteration,
        ax_second_round_rotations,
        config,
        states,
        follower_actions,
        color,
    ):
        second_round_figure_list = []
        plt.style.use("ggplot")
        for k, rotation in enumerate(ax_second_round_rotations):
            figure_plt = plt.figure(figsize=(4.5, 4.5), clear=True, dpi=600)
            ax = figure_plt.add_subplot(111, projection="3d")
            ax.view_init(rotation[0], rotation[1])
            ax.dist = 13
            self._draw_second_round_strategy_on_axis(
                ax, config, states, follower_actions, color
            )
            ax.set_title("Bertrand Follower")
            if self.config.sampler.name in [
                "mineral_rights_common_value",
                "affiliated_uniform",
            ]:
                ax.set_xlabel("F's observation $x_2$", fontsize=10)
            else:
                ax.set_xlabel("F's cost $c_2$", fontsize=10)
            ax.set_ylabel("L's price $p_1$", fontsize=10)
            ax.set_zlabel("F's price $p_2$", fontsize=10)

            # val_low, val_high = self.get_prior_bounds(agent_id=0)
            # ax.set_xlim([val_low - 0.1, val_high + 0.1])
            if (
                self.config["prettify_plots"]
                and self.config.sampler.name == "mineral_rights_common_value"
            ):
                ax.set_zlim(0.4 - 0.05, 1.1 + 0.05)
            elif self.config["prettify_plots"] and self.cara_risk_aversion > 0.0:
                ax.set_zlim(0.15 - 0.05, 1.1 + 0.05)
            ax.legend(loc="best")

            figure_plt.tight_layout()
            figure_plt.savefig(f"{writer.log_dir}/{iteration}_follower_{k}.png")

            second_round_figure_list.append(figure_plt)
        return second_round_figure_list

    def _draw_second_round_strategy_on_axis(
        self, ax, config, states, follower_actions, color
    ):
        algo_name = pl_ut.get_algo_name(1, config)
        follower_vals = states["cost_signals"][:, 1, 0]
        leader_actions = states["quoted_prices"][:, 1, 0]

        sorted_follower_vals, increasing_order = follower_vals.sort(axis=0)
        sorted_follower_vals = sorted_follower_vals.detach().cpu().view(-1).numpy()
        sorted_follower_actions = (
            follower_actions[increasing_order].squeeze().detach().cpu().view(-1).numpy()
        )
        sorted_leader_actions = (
            leader_actions[increasing_order].detach().cpu().view(-1).numpy()
        )
        surf = ax.plot_trisurf(
            sorted_follower_vals,
            sorted_leader_actions,
            sorted_follower_actions,
            linewidth=0.3,
            antialiased=True,
            alpha=0.5,
            color=color,
            edgecolor=color,
            label=algo_name + " follower",
        )
        # ## due to bug in matplotlib ## #
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        # ############################## #
        self._plot_second_round_equ_strategy_surface(ax, 50)

    def _plot_second_round_equ_strategy_surface(self, ax, plot_precision):
        if self.equilibrium_strategies[1] is not None:
            (
                follower_val,
                leader_action,
                follower_action,
            ) = self._get_actions_and_grid_in_second_stage(
                self.equilibrium_strategies[1], plot_precision
            )
            surf = ax.plot_surface(
                follower_val.cpu().numpy(),
                leader_action.cpu().numpy(),
                follower_action.cpu().numpy(),
                alpha=0.2,
                color="red",
                edgecolor="black",
            )
            # ## due to bug in matplotlib ## #
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            # ############################## #s

    def _get_actions_and_grid_in_second_stage(self, sa_strategy, precision: int):
        follower_val, leader_action = self._get_meshgrid_for_second_round_equ(precision)
        # flatten mesh for forward
        follower_val, leader_action = (
            follower_val.reshape(precision**2),
            leader_action.reshape(precision**2),
        )
        sa_obs = torch.stack((follower_val, leader_action), dim=-1)
        follower_action, _ = sa_strategy.predict(sa_obs, deterministic=True)
        follower_action = follower_action.reshape(precision, precision)
        follower_val = follower_val.reshape(precision, precision)
        leader_action = leader_action.reshape(precision, precision)
        return follower_val, leader_action, follower_action

    def _get_meshgrid_for_second_round_equ(self, precision):
        val_low, val_high = self.get_prior_bounds(agent_id=1)
        follower_vals = torch.linspace(val_low, val_high, steps=precision)
        leader_actions = np.linspace(0.5359, 0.9999, num=precision)
        follower_grid_vals, leader_grid_actions = torch.meshgrid(
            follower_vals,
            torch.tensor(leader_actions, dtype=torch.float32),
            indexing="xy",
        )
        return follower_grid_vals, leader_grid_actions

    def plot_br_strategy(
        self, br_strategies: Dict[int, Callable]
    ) -> Optional[plt.Figure]:
        num_vals = 128
        val_low, val_high = self.get_prior_bounds(agent_id=0)
        cost_signals = torch.linspace(val_low, val_high, num_vals, device=self.device)
        agent_obs = torch.cat(
            (
                cost_signals.unsqueeze(-1),
                torch.zeros((num_vals, 3), device=self.device),
            ),
            dim=1,
        )
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(4.5, 4.5), clear=True)
        ax = fig.add_subplot(111)
        if self.config.sampler.name in [
            "mineral_rights_common_value",
            "affiliated_uniform",
        ]:
            ax.set_xlabel("L's observation $x_1$", fontsize=12)
        else:
            ax.set_xlabel("L's cost $c_1$", fontsize=12)
        ax.set_ylabel("L's price $p_1$")
        ax.set_xlim([val_low, val_high])
        ax.set_ylim([val_low - 0.05, val_high + 1.05])

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
            xs = cost_signals.detach().cpu().numpy()
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
        return "BertrandCompetition"
