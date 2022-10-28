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

from src.envs.equilibria import equilibrium_fpsb_symmetric_uniform, truthful
from src.envs.mechanisms import FirstPriceAuction, Mechanism, VickreyAuction
from src.envs.torch_vec_env import BaseEnvForVec
from src.learners.utils import batched_index_select, tensor_norm
from src.utils_folder.policy_utils import get_algo_name


class SequentialAuction(BaseEnvForVec):
    """Sequential first price sealed bid auction.

    In each stage there is a single item sold.
    """

    DUMMY_PRICE_KEY = -1

    def __init__(self, config: Dict, device: str = "cpu"):
        self.num_rounds_to_play = config["num_rounds_to_play"]

        # If the opponents are all symmetric, we may just sample from one
        # opponent with the max order statistic corresponding to the same
        # competition as the original opponents
        self.collapse_symmetric_opponents = config["collapse_symmetric_opponents"]
        self.num_opponents = config["num_agents"] - 1

        # `num_actual_agents` may be larger than `num_agents` when we use
        # `collapse_symmetric_opponents`: Then `num_agents` will be lowered to
        # correspond the sole learner.
        self.num_actual_agents = config["num_agents"]
        if self.collapse_symmetric_opponents:
            self.num_opponents = 1

        super().__init__(config, device)

        self.mechanism, self.equilibrium_profile = (
            self._init_mechanism_and_equilibrium_profile()
        )

        # NOTE: unit-demand only atm
        self.valuation_size = self.config["valuation_size"]
        self.action_size = self.config["action_size"]
        self.strategies = self._init_dummy_strategies()

        if self.collapse_symmetric_opponents:
            # Overwrite: external usage of this `BaseEnvForVec` should only
            # interact via a single learner.
            self.num_agents = 1
        self.strategies_bne = self._init_bne_strategies()

    def _init_mechanism_and_equilibrium_profile(self):
        if self.config["mechanism_type"] == "first":
            mechanism: Mechanism = FirstPriceAuction()
            equilibrium_profile = equilibrium_fpsb_symmetric_uniform
        elif self.config["mechanism_type"] in ["second", "vcg", "vickery"]:
            mechanism: Mechanism = VickreyAuction()
            equilibrium_profile = truthful
        else:
            raise NotImplementedError("Payment rule unknown.")
        return mechanism, equilibrium_profile

    def _init_bne_strategies(self):
        return {
            agent_id: self.equilibrium_profile(
                num_agents=self.num_actual_agents,
                num_units=self.num_rounds_to_play,
                player_position=agent_id,
            )
            for agent_id in range(self.num_agents)
        }

    def _init_dummy_strategies(self):
        return [
            lambda obs, deterministic=True: torch.zeros(
                (obs.shape[0], self.config["action_size"]), device=obs.device
            )
            for _ in range(self.num_agents)
        ]

    def _get_num_agents(self) -> int:
        return self.config["num_agents"]

    def _init_observation_spaces(self):
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """
        # unit-demand
        self.valuation_size = 1
        self.action_size = 1

        # set up observation space
        # NOTE: does not support non unit-demand
        self.reduced_observation_space = self.config["reduced_observation_space"]
        if self.reduced_observation_space:
            # observations: valuation, stage, allocation (up to now)
            low = [0.0] * 3
            high = [1.0, self.num_rounds_to_play, 1.0]
        else:
            # observations: valuation, allocation (including in which stage
            # obtained), previous prices (including in which stage payed)
            low = (
                [0.0]
                + [0.0] * self.num_rounds_to_play
                + [-1.0] * (self.num_rounds_to_play * 2)
            )
            high = (
                [1.0]
                + [1.0] * self.num_rounds_to_play
                + [np.inf] * (self.num_rounds_to_play * 2)
            )
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
        """Create new initial states consisting of
            * one valuation per agent
            * num_rounds_to_play * allocation per agent
            * prices of all stages (-1 for future stages TODO?)
                -> implicitly tells agents which the current round is

        :param n: Batch size of how many auction games are played in parallel.
        :return: The new states, in shape=(n, num_agents*2 + num_rounds_to_play),
            where ...
        `current_round` and `num_rounds_to_play`.
        """
        self.valuations_start_index = 0
        self.allocations_start_index = self.valuation_size
        self.payments_start_index = (
            self.valuation_size + self.valuation_size * self.num_rounds_to_play
        )
        # NOTE: We keep track of all (incl. zero) payments for reward calculations
        states = torch.zeros(
            (
                n,
                self.num_opponents + 1,
                self.payments_start_index + self.num_rounds_to_play,
            ),
            device=self.device,
        )

        # ipv symmetric uniform priors
        states[:, :, : self.valuation_size].uniform_(0, 1)

        if self.collapse_symmetric_opponents:
            m = torch.distributions.Beta(
                torch.tensor([self.num_actual_agents - 1], device=self.device),
                torch.tensor([1.0], device=self.device),
            )
            states[:, 1, : self.valuation_size] = m.sample((n,))

        # dummy prices
        states[:, :, self.payments_start_index :] = SequentialAuction.DUMMY_PRICE_KEY

        return states

    def compute_step(
        self, cur_states, actions: torch.Tensor, for_single_learner: bool = True
    ):
        """Compute a step in the game.

        :param cur_states: The current states of the games.
        :param actions: Actions that the active player at
            `self.player_position` is choosing.
        :return observations:
        :return rewards:
        :return episode-done markers:
        :return updated_states:
        """
        player_positions_of_actions = set(actions.keys())

        # create opponent action
        if self.collapse_symmetric_opponents:

            # TODO: The cases should be clear!!!
            # Possibly we do not want to overwrite 1's action?
            assert player_positions_of_actions in [{0, 1}, {0}]

            opponent_obs = self.get_observations(cur_states, for_single_learner=False)[
                1
            ]
            opponent_actions = self.learners[0].policy.forward(opponent_obs)[0]
            actions[1] = opponent_actions
        else:
            assert player_positions_of_actions == set(range(self.num_agents))

        # append opponents' actions
        action_profile = torch.stack(tuple(actions.values()), dim=1)

        # run auction round
        allocations, payments = self.mechanism.run(action_profile)

        # states are for all agents, obs and rewards for `player_position` only
        new_states = cur_states.detach().clone()

        # get current stage
        stage = self._state2stage(cur_states)

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
            # the only thing the current player knows is that the opponent
            # faced in the next round is weaker
            highest_opponent = new_states[:, 1, : self.valuation_size]
            m = torch.distributions.Beta(
                torch.tensor(
                    [max(1, self.num_actual_agents - 2 - stage)], device=self.device
                ),
                torch.tensor([1.0], device=self.device),
            )
            batch_size = cur_states.shape[0]
            new_states[:, 1, : self.valuation_size] = highest_opponent * m.sample(
                (batch_size,)
            )

            # force opponent's allocation to zero again st it competes in next
            # stage
            new_states[
                :,
                1,
                self.allocations_start_index
                + stage * self.valuation_size : self.allocations_start_index
                + (stage + 1) * self.valuation_size,
            ] = 0

        # reached last stage?
        if stage >= self.num_rounds_to_play - 1:
            dones = torch.ones((cur_states.shape[0]), device=cur_states.device).bool()
        else:
            dones = torch.zeros((cur_states.shape[0]), device=cur_states.device).bool()

        observations = self.get_observations(
            new_states, for_single_learner=for_single_learner
        )

        rewards = self._compute_rewards(new_states, stage)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, states: torch.Tensor, stage: int) -> torch.Tensor:
        """Computes the rewards for the played auction games for the player at
        `self.player_position`.

        TODO: do we want intermediate rewards or not?
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

        # set valuation to zero if we already own the unit
        has_won_already = self._has_won_already(states, stage)[agent_id].view_as(
            payments
        )

        # quasi-linear utility
        rewards = (valuations * allocations) * torch.logical_not(
            has_won_already
        ) - payments

        return rewards.view(-1)

    def get_observations(
        self, states: torch.Tensor, for_single_learner: bool = True
    ) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, obs_private_dim
            + obs_public_dim), where the private observations consist of the
            valuation and a vector of allocations and payments (for each stage)
            and the public observation consists of published prices.
        """
        num_agents = self.num_agents if for_single_learner else self.num_opponents + 1

        if self.reduced_observation_space:
            # Observation consists of: own valuation, stage number, and own
            # allocations

            stage = self._state2stage(states)
            won = self._has_won_already(states, stage)
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
                    self.valuations_start_index : self.valuations_start_index
                    + self.valuation_size,
                ]
                observation_dict[agent_id][:, 1] = stage
                observation_dict[agent_id][:, 2] = won[agent_id]

        else:
            # Observation consists of: own valuation, own allocations, own
            # payments and published (here = highest payments)
            obs_public = states[:, :, self.payments_start_index :].max(axis=1).values
            observation_dict = {
                agent_id: torch.concat((states[:, agent_id, :], obs_public), axis=1)
                for agent_id in range(num_agents)
            }
        return observation_dict

    def render(self, state):
        return state

    def _state2stage(self, cur_states):
        """Get the current stage from the state."""
        if cur_states.shape[0] == 0:  # empty batch
            return -1
        try:
            # NOTE: only works for fixed length / each batch at same stage
            stage = (
                cur_states[0, 0, self.payments_start_index :]
                .tolist()
                .index(SequentialAuction.DUMMY_PRICE_KEY)
            )
        except ValueError as _:  # last round
            stage = self.num_rounds_to_play - 1
        return stage

    def _has_won_already(self, state: torch.Tensor, stage: int):
        """Check if the current player already has won in previous stages of the auction."""
        # NOTE: unit-demand hardcoded

        low = self.allocations_start_index
        high = self.allocations_start_index + stage
        return {
            agent_id: state[:, agent_id, low:high].sum(axis=-1) > 0
            for agent_id in range(self.num_opponents + 1)
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
        self.log_metrics_to_equilibrium(learners)

    def get_bne_actions(
        self, valuations: torch.Tensor, stage: int, won: torch.Tensor, agent_id: int
    ) -> torch.Tensor:
        if self.collapse_symmetric_opponents:
            agent_id = 0  # we only consider a single BNE strategy then
        return self.strategies_bne[agent_id](stage, valuations, won)

    @staticmethod
    def get_ma_learner_predictions(
        learners,
        observations,
        deterministic: bool = True,
        for_single_learner: bool = True,
    ):
        if for_single_learner:
            predictions = {
                agent_id: learner.predict(
                    observations[agent_id], deterministic=deterministic
                )[0]
                for agent_id, learner in learners.items()
            }
        else:
            predictions = {
                agent_id: learners[0].predict(obs, deterministic=deterministic)[0]
                for agent_id, obs in observations.items()
            }
        return predictions

    @staticmethod
    def get_ma_learner_stddevs(learners, observations):
        stddevs = {
            agent_id: learners[0].policy.get_stddev(obs)
            for agent_id, obs in observations.items()
        }
        return stddevs

    @staticmethod
    def get_equilibrium_actions(
        stage: int, equilibrium_strategies, observations, has_won_already
    ):
        return {
            agent_id: equilibrium_strategy(
                stage=stage,
                valuation=observations[agent_id][:, 0],
                won=has_won_already[agent_id],
            )
            for agent_id, equilibrium_strategy in equilibrium_strategies.items()
        }

    def plot_strategies_vs_bne(
        self, learners, writer, iteration: int, config, num_samples: int = 2 ** 12
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
            observations = self.get_observations(states, for_single_learner=False)
            ma_deterministic_actions = self.get_ma_learner_predictions(
                learners, observations, True, for_single_learner=False
            )
            ma_stddevs = self.get_ma_learner_stddevs(learners, observations)

            for agent_id in range(len(learners)):
                agent_obs = observations[agent_id]
                increasing_order = agent_obs[:, 0].sort(axis=0)[1]

                # get algorithm type
                learner_name = learners[agent_id].__class__.__name__

                # sort
                agent_obs = agent_obs[increasing_order]

                has_won_already = self._has_won_already(
                    states[increasing_order], stage
                )[agent_id]

                # get actual actions
                deterministic_actions = ma_deterministic_actions[agent_id][
                    increasing_order
                ]
                stddevs = ma_stddevs[agent_id][increasing_order]

                # get BNE actions
                actions_bne = self.get_bne_actions(
                    valuations=agent_obs[:, 0],
                    stage=stage,
                    won=has_won_already,
                    agent_id=agent_id,
                )

                # covert to numpy
                agent_obs = agent_obs[:, 0].detach().cpu().view(-1).numpy()
                actions_array = deterministic_actions.view(-1, 1).detach().cpu().numpy()
                stddevs = stddevs.view(-1).detach().cpu().numpy()
                actions_bne = actions_bne.view(-1, 1).detach().cpu().numpy()
                has_won_already = has_won_already.cpu().numpy()

                algo_name = get_algo_name(agent_id, config)

                # plotting
                drawing, = ax.plot(
                    agent_obs[~has_won_already],
                    actions_array[~has_won_already],
                    linestyle="-",
                    label=f"bidder {agent_id+1} {learner_name}",
                )
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
                        label=f"bidder {agent_id+1} {learner_name} (won)",
                    )
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
            lin = np.linspace(0, 1, 2)
            ax.plot(lin, lin, "--", color="grey", alpha=0.5)
            ax.set_xlabel("valuation $v$")
            if stage == 0:
                ax.set_ylabel("bid $b$")
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.05, 1.05])

            # apply actions to get to next stage
            _, _, _, states = self.compute_step(
                states, ma_deterministic_actions, for_single_learner=False
            )

        handles, labels = ax.get_legend_handles_labels()
        axs[0].legend(handles, labels, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{writer.log_dir}/plot_{iteration}.png")
        writer.add_figure("images", fig, iteration)
        plt.close()

        # reset seed
        self.seed(int(time.time()))

    def log_metrics_to_equilibrium(self, learners, num_samples: int = 4096):
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
        actual_states = self.sample_new_states(num_samples)
        actual_observations = self.get_observations(actual_states)

        equ_states = actual_states.clone()
        equ_observations = self.get_observations(equ_states)

        l2_distances = {i: [None] * self.num_rounds_to_play for i in learners.keys()}
        actual_rewards_total = {i: 0 for i in learners.keys()}
        equ_rewards_total = {i: 0 for i in learners.keys()}

        for stage in range(self.num_rounds_to_play):

            actual_has_won_already = self._has_won_already(actual_states, stage)
            equ_actions_in_actual_play = self.get_equilibrium_actions(
                stage, self.strategies_bne, actual_observations, actual_has_won_already
            )

            equ_has_won_already = self._has_won_already(equ_states, stage)
            equ_actions_in_equ = self.get_equilibrium_actions(
                stage, self.strategies_bne, equ_observations, equ_has_won_already
            )

            # NOTE: Here we need to query `self.learners` (even when policy
            # sharing is turned on), because we need multi-agent actions for
            # all player positions
            actual_actions = self.get_ma_learner_predictions(
                self.learners, actual_observations, True
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
