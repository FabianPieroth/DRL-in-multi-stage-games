"""
"""
import time
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

import src.utils.distributions_and_priors as dap_ut
import src.utils.torch_utils as th_ut
from src.envs.equilibria import BertrandCompetitionEquilibrium
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.utils import tensor_norm


class BertrandCompetition(VerifiableEnv, BaseEnvForVec):
    """Bertrand Competition as in https://doi.org/10.1016/j.econlet.2009.03.017
    """

    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_stages = 2
        self.valuation_size = 1
        self.observation_size = config["observation_size"]
        self.action_size = 1
        self.sampler = self._init_sampler(config, device)

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
                low=np.float32([val_low] * self.config.observation_size),
                high=np.float32([val_high] * self.config.observation_size),
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
            dimension consists of the valuation and firm 1's quote (initialized
            at zero).
        """
        states = -torch.ones((n, self.num_agents, 2), device=self.device)
        # keep 2nd entry to -1 as to detect which stage it is (firm 1 must
        # quote a value above 0.)

        # draw valuations
        valuations = self.sampler.draw_profiles(n)[0]
        states[:, :, [0]] = valuations

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
        batch_size = cur_states.shape[0]

        # get current stage
        stage = self._state2stage(cur_states)

        new_states = cur_states.clone()

        # 1. stage: firm 1 quotes
        if stage == 0:
            new_states[:, 1, 1] = actions[0].squeeze()
            new_states[:, 0, 1] = 1  # Set stage info for agent 0
            rewards = {
                0: torch.zeros(batch_size, device=self.device),
                1: torch.zeros(batch_size, device=self.device),
            }
            dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 2. stage: firm 2 quotes
        if stage == 1:
            new_states[:, 0, 1] = actions[1].squeeze()
            rewards = self._compute_rewards(new_states, stage)
            dones = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        observations = self.get_observations(new_states)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played competition."""
        batch_size = states.shape[0]

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

    def custom_evaluation(self, learners, env, writer, iteration: int, config: Dict):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            iteration: current training iteration
        """
        self.plot_strategies_vs_bne(learners, writer, iteration, config)

    def plot_strategies_vs_bne(
        self, strategies, writer, iteration: int, config, num_samples: int = 2 ** 12
    ):
        # TODO: Improve plots
        """Evaluate and log current strategies."""

        plt.style.use("ggplot")
        fig = plt.figure()
        fig.suptitle(f"Iteration {iteration}", fontsize="x-large")
        agent_actions_list = []
        equ_actions_list = []

        states = self.sample_new_states(num_samples)
        for stage in range(self.num_stages):
            observations = self.get_observations(states)
            ma_deterministic_actions = th_ut.get_ma_actions(
                strategies, observations, True
            )
            agent_actions_list.append(ma_deterministic_actions)

            equ_actions = th_ut.get_ma_actions(
                self.equilibrium_strategies, observations
            )
            equ_actions_list.append(equ_actions)

            _, _, _, states = self.compute_step(states, ma_deterministic_actions)

        # Firm 1's one-dim. quote
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Firm 1's Quote")
        ax.plot(
            states[:, 0, 0].cpu().numpy(),
            states[:, 1, 1].cpu().numpy(),
            ".",
            label=f"learned strategy",
        )
        ax.plot(
            states[:, 0, 0].cpu().numpy(),
            equ_actions_list[0][0].cpu().numpy(),
            ".",
            label=f"BNE strategy",
        )
        ax.set_xlabel("Firm 1's Costs")
        ax.set_ylabel("Firm 1's Quote")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.1])
        ax.legend()

        # Firm 2's two-dim. quote
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.set_title("Firm 2's Quote")
        # ax.plot_trisurf(
        ax.scatter(
            states[:, 1, 0].cpu().numpy(),
            states[:, 1, 1].cpu().numpy(),
            states[:, 1, 0].cpu().numpy(),
            alpha=0.7,
            label=f"learned strategy",
        )
        ax.scatter(
            states[:, 1, 0].cpu().numpy(),
            states[:, 1, 1].cpu().numpy(),
            equ_actions_list[1][1].cpu().numpy(),
            alpha=0.7,
            label=f"BNE strategy",
        )
        ax.set_xlabel("Firm 2's Costs")
        ax.set_ylabel("Firm 1's Quote")
        ax.set_zlabel("Firm 2's Quote")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.1])
        ax.set_zlim([0, 1.1])

        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{iteration}.png")
        writer.add_figure("images", fig, iteration)
        plt.close()

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
