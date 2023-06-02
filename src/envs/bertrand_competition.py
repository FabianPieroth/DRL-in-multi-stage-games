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
from src.envs.equilibria import SequentialAuctionEquilibrium
from src.envs.mechanisms import FirstPriceAuction, Mechanism, VickreyAuction
from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.utils import tensor_norm


class BertrandCompetition(VerifiableEnv, BaseEnvForVec):
    """Bertrand Competition as in https://doi.org/10.1016/j.econlet.2009.03.017
    """

    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_rounds_to_play = 2
        self.valuation_size = 1
        self.observation_size = 2
        self.action_size = 1
        self.num_agents = 2

        self.sampler = self._init_sampler(config, device)

        super().__init__(config, device)

    def _get_num_agents(self) -> int:
        return self.num_agents

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[Callable]]:
        return {
            agent_id: self._get_agent_equilibrium_strategy(agent_id)
            for agent_id in range(self.num_agents)
        }
        # return {agent_id: None for agent_id in range(self.num_agents)}

    def _get_agent_equilibrium_strategy(self, agent_id: int):
        if agent_id == 0:
            """See paper eq. (2)"""
            device = self.device
            # F(b) = 0.5(b + b**2)
            # f(b) = 0.5 + b
            # Q(p) = 10 - p
            # q(p) = -p

            # b - phi1(b) = ((10 - b)(1 - 0.5*b - 0.5*b**2)) / ((10 - b)(0.5 + b) + b*(1 - 0.5*b - 0.5*b**2))
            inverse_bid_function = (
                lambda b: -(10 - 6 * b - 4.5 * b ** 2 + 0.5 * b ** 3)
                / (5 + 10.5 * b - b ** 2 - 0.5 * b ** 3)
                + b
            )

            from scipy.interpolate import interp1d

            bids = np.linspace(0, 3, 100)
            inverse_bids = inverse_bid_function(bids)
            bid_function = interp1d(inverse_bids, bids)

            class Strategy:
                def predict(
                    self,
                    observation,
                    state=None,
                    episode_start=None,
                    deterministic=True,
                ):
                    quote = torch.tensor(
                        bid_function(observation[:, 0].cpu().numpy()), device=device
                    )
                    return quote.view(-1, 1), None

        elif agent_id == 1:
            """Given b1, firm 2 has to match that bid to win. It will want to
            do so whenever b1 >= c2, and will thus set b2 = min{b1, pM(c2)},
            where pM(c2) is the monopoly price for unit cost c2. If b1 < c2,
            firm 2 will not match but rather set some price b2 > b1 so as to
            lose.
            """

            class Strategy:
                def predict(
                    self,
                    observation,
                    state=None,
                    episode_start=None,
                    deterministic=True,
                ):
                    c2 = observation[:, 0]
                    b1 = observation[:, 1]

                    # NOTE: This is based on quantity Q(p) = 10 - p and profit
                    # = Q(p) * (p - c)
                    monopoly_price = 5 + c2 / 2

                    # Match if b1 >= c2
                    quote = torch.min(b1, monopoly_price)

                    # If b1 < c2, set b2 > b1 so as to lose
                    quote[b1 < c2] = c2[b1 < c2]

                    return quote.view(-1, 1), None

        return Strategy()

    def _init_sampler(self, config, device):
        return dap_ut.get_sampler(
            self.num_agents,
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

            # observations: valuation, firm 1 quote / placeholder
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
                low=np.float32([val_low] * self.config.action_size),
                high=np.float32([2 * val_high] * self.config.action_size),
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
        device = cur_states.device  # may differ from `self.device`
        batch_size = cur_states.shape[0]
        player_positions_of_actions = set(actions.keys())

        # get current stage
        stage = self._state2stage(cur_states)

        new_states = cur_states.clone()

        # 1. stage: firm 1 quotes
        if stage == 0:
            new_states[:, 1, 1] = actions[0].squeeze()
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
        return {0: states[:, 0, :], 1: states[:, 1, :]}

    def render(self, state):
        return state

    def _state2stage(self, states):
        """Get the current stage from the state."""
        if states.shape[0] == 0:  # empty batch
            return -1
        stage = 0 if states[0, 1, 1] == -1 else 1
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
        from mpl_toolkits.mplot3d import Axes3D

        plt.style.use("ggplot")
        fig = plt.figure()
        fig.suptitle(f"Iteration {iteration}", fontsize="x-large")

        states = self.sample_new_states(num_samples)
        for s in range(2):
            observations = self.get_observations(states)
            ma_deterministic_actions = th_ut.get_ma_actions(
                strategies, observations, True
            )
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
        equ_actions = th_ut.get_ma_actions(self.equilibrium_strategies, observations)
        ax.plot(
            states[:, 0, 0].cpu().numpy(),
            equ_actions[0].cpu().numpy(),
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
            equ_actions[1].cpu().numpy(),
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

    def log_metrics_to_equilibrium(self, strategies, num_samples: int = 4096):
        """Evaluate learned strategies vs BNE."""

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
        return "BertrandCompetition"
