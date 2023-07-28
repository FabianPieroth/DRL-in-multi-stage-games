"""
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

import src.utils.distributions_and_priors as dap_ut
import src.utils.evaluation_utils as ev_ut
import src.utils.policy_utils as pl_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import BertrandCompetitionEquilibrium
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv


class BertrandCompetition(VerifiableEnv, BaseEnvForVec):
    """Bertrand Competition as in https://doi.org/10.1016/j.econlet.2009.03.017
    """

    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_stages = 2
        self.valuation_size = 1
        self.observation_size = config["observation_size"]
        self.action_size = 1
        self.prior_low = config.sampler.prior_low
        self.prior_high = config.sampler.prior_high
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

    def _get_agent_equilibrium_strategy(self, agent_id: int):
        val_low = self.sampler.support_bounds[agent_id, 0, 0].cpu().detach().item()
        val_high = self.sampler.support_bounds[agent_id, 0, 1].cpu().detach().item()
        equilibrium_config = {
            "num_agents": self.config.num_agents,
            "num_stages": self.num_stages,
            "prior_low": val_low,
            "prior_high": val_high,
            "device": self.device,
        }
        return BertrandCompetitionEquilibrium(agent_id, equilibrium_config)

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
            action_spaces_dict[agent_id] = spaces.Box(
                low=np.float32([val_low] * self.action_size),
                high=np.float32([2 * val_high] * self.action_size),
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
        states = -torch.ones((n, self.num_agents, 2), device=self.device)
        # keep 2nd entry to -1 as to detect which stage it is (firm 1 must
        # quote a value above 0.)

        # draw valuations
        valuations, _ = self.sampler.draw_profiles(n)
        states[:, :, [0]] = valuations

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
            new_states[:, 1, 1] = actions[0].squeeze()
            new_states[:, 0, 1] = 1  # Set stage info for agent 0
            rewards = {
                0: torch.zeros(batch_size, device=self.device),
                1: torch.zeros(batch_size, device=self.device),
            }
            dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 2. stage: firm 2's quotes to firm 1's info
        if stage == 1:
            new_states[:, 0, 1] = actions[1].squeeze()
            rewards = self._compute_rewards(new_states, stage)
            dones = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        observations = self.get_observations(new_states)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played competition."""
        firm1_wins = states[:, 1, 1] < states[:, 0, 1]
        quantity = 10 - states[:, :, 1].min(axis=1).values
        return {
            0: firm1_wins * quantity * (states[:, 1, 1] - states[:, 0, 0]),
            1: torch.logical_not(firm1_wins)
            * quantity
            * (states[:, 0, 1] - states[:, 1, 0]),
        }

    def get_observations(self, states: torch.Tensor) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :returns observations: Observations consisting of own valuation and
            other firm's quote.
        """
        return {
            agent_id: states[:, agent_id, :].clone()
            for agent_id in range(self.num_agents)
        }

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
        stage = 0 if states[0, 1, 1] == -1 else 1
        return stage

    def provide_env_verifier_info(
        self, stage: int, agent_id: int, obs_discretization: int
    ) -> Tuple:
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
        if stage == 1 and agent_id == 1:
            low = tuple([self.prior_low] * len(obs_indices))
            high = tuple([self.prior_high, 1.3])
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
        self, learners, writer, iteration: int, config, num_samples: int = 2 ** 12
    ):
        """Evaluate and log current strategies."""

        plt.style.use("ggplot")
        total_num_second_round_plots = 5
        ax_second_round_rotations = [
            (30, -60),
            (10, -100),
            (-10, 80),
            (30, 120),
            (30, 45),
        ]
        agent_plot_colors = ["red", "blue"]
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

        self._plot_first_round_strategy(
            ax_first_round,
            config,
            agent_stddevs_list[0][0],
            states_list[0],
            actions_list[0][0],
            agent_plot_colors[0],
        )

        for ax_second_round, rotation in zip(
            ax_second_round_list, ax_second_round_rotations
        ):
            ax_second_round.view_init(rotation[0], rotation[1])
            ax_second_round.dist = 13
            self._plot_second_round_strategy(
                ax_second_round,
                config,
                states_list[1],
                actions_list[1][1],
                agent_plot_colors[1],
            )
        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{iteration}.png")
        writer.add_figure("images", fig, iteration)
        plt.close()

    def _plot_first_round_strategy(
        self, ax, config, leader_stddevs, states, leader_actions, agent_color
    ):
        leader_vals = states[:, 0, 0]
        sorted_leader_vals, increasing_order = leader_vals.sort(axis=0)
        sorted_leader_vals = sorted_leader_vals.detach().cpu().view(-1).numpy()
        sorted_leader_actions = (
            leader_actions[increasing_order].detach().cpu().view(-1).numpy()
        )
        sorted_leader_stddevs = (
            leader_stddevs[increasing_order].detach().cpu().view(-1).numpy()
        )
        ax.set_title("First stage")
        algo_name = pl_ut.get_algo_name(0, config)
        (drawing,) = ax.plot(
            sorted_leader_vals,
            sorted_leader_actions,
            linestyle="-",
            label=f"Leader " + algo_name,
            color=agent_color,
        )
        ax.fill_between(
            sorted_leader_vals,
            (sorted_leader_actions.squeeze() - sorted_leader_stddevs).clip(min=0),
            (sorted_leader_actions.squeeze() + sorted_leader_stddevs).clip(min=0),
            alpha=0.2,
            color=drawing.get_color(),
        )
        self._plot_first_round_equilibrium_strategy(ax, drawing)
        ax.set_xlabel("Firm 1's Costs")
        ax.set_ylabel("Firm 1's Quote")
        ax.set_xlim([self.prior_low - 0.1, self.prior_high + 0.1])
        ax.legend()

    def _plot_first_round_equilibrium_strategy(
        self, plot_axis, drawing, precision: int = 200
    ):
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
        val_xs = torch.linspace(
            self.prior_low, self.prior_high, steps=precision, device=self.device
        )
        opp_info = -1.0 * torch.ones_like(val_xs)
        sa_obs = torch.stack((val_xs, opp_info), dim=-1)
        bid_ys, _ = sa_learner.predict(sa_obs, deterministic=True)
        return val_xs, bid_ys

    def _plot_second_round_strategy(self, ax, config, states, follower_actions, color):
        algo_name = pl_ut.get_algo_name(1, config)
        follower_vals, leader_actions = states[:, 1, 0], states[:, 1, 1]

        sorted_follower_vals, increasing_order = follower_vals.sort(axis=0)
        sorted_follower_vals = sorted_follower_vals.detach().cpu().view(-1).numpy()
        sorted_follower_actions = (
            follower_actions[increasing_order].squeeze().detach().cpu().view(-1).numpy()
        )
        sorted_leader_actions = (
            leader_actions[increasing_order].detach().cpu().view(-1).numpy()
        )
        ax.set_title("Second stage")
        surf = ax.plot_trisurf(
            sorted_follower_vals,
            sorted_leader_actions,
            sorted_follower_actions,
            linewidth=0.3,
            antialiased=True,
            alpha=0.5,
            color=color,
            edgecolor=color,
            label=f"Follower " + algo_name,
        )
        # ## due to bug in matplotlib ## #
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        # ############################## #
        self._plot_second_round_equ_strategy_surface(ax, 50)

        ax.set_xlabel("F's Costs")
        ax.set_ylabel("L's Quote")
        ax.set_zlabel("F's Quote")

    def _plot_second_round_equ_strategy_surface(self, ax, plot_precision):
        follower_val, leader_action, follower_action = self._get_actions_and_grid_in_second_stage(
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
            follower_val.reshape(precision ** 2),
            leader_action.reshape(precision ** 2),
        )
        sa_obs = torch.stack((follower_val, leader_action), dim=-1)
        follower_action, _ = sa_strategy.predict(sa_obs, deterministic=True)
        follower_action = follower_action.reshape(precision, precision)
        follower_val = follower_val.reshape(precision, precision)
        leader_action = leader_action.reshape(precision, precision)
        return follower_val, leader_action, follower_action

    def _get_meshgrid_for_second_round_equ(self, precision):
        follower_vals = torch.linspace(self.prior_low, self.prior_high, steps=precision)
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
        # TODO: Adapt to fit Bertrand Competition
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
        return "BertrandCompetition"
