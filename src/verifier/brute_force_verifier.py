"""Verifier"""
import gc
import traceback
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import torch

import src.utils.evaluation_utils as ev_ut
import src.utils.io_utils as io_ut
import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import VerifiableEnv
from src.learners.base_learner import SABaseAlgorithm
from src.verifier.information_set_tree import InformationSetTree
from src.verifier.mean_utility_tracker import UtilityTracker


class BFVerifier:
    """Verifier that tries out a grid of alternative actions and choses the
    best combination as approximation to a best response.
    Assumptions:
    1. The number of stages is equal for all initial conditions
    2. We assume that the initial conditions are independent between the agents
    3. We only support single dimension actions at the moment
    """

    def __init__(
        self,
        env,
        num_simulations: int,
        obs_discretization: int,
        action_discretization: int,
        batch_size: int,
        device: str,
    ):
        self.env = env
        self.env_is_compatible_with_verifier = isinstance(env.model, VerifiableEnv)
        self.num_agents = self.env.model.num_agents
        self.num_simulations = num_simulations
        self.action_discretization = action_discretization
        self.obs_discretization = obs_discretization
        self.action_dim = env.model.ACTION_DIM
        self.batch_size = batch_size
        self.device = device
        if self.env_is_compatible_with_verifier:
            self.num_stages = self.env.model.num_stages
            self.env.model.verfier_env_info = self.env.model.get_verifier_env_infos(
                obs_discretization=self.obs_discretization
            )

    def verify_br(
        self, strategies: Dict[int, SABaseAlgorithm], agent_ids: List[int] = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Dict[int, Callable]]
    ]:
        """Build information set tree for single agent and calcuate best-reponse over grid.
        Return the estimated metrics plus the br-strategies.

        Args:
            strategies (Dict[int, SABaseAlgorithm]):
            agent_ids (List[int], optional): Defaults to None.

        Returns:
            actual_utilities (torch.Tensor):
            utility_losses (torch.Tensor):
            relative_utility_losses (torch.Tensor):
            best_responses (Dict[int, Dict[int, Callable]]):
        """
        agent_ids = list(range(self.num_agents)) if agent_ids is None else agent_ids
        actual_utilities = {agent_id: 0 for agent_id in agent_ids}
        utility_losses = {agent_id: 0 for agent_id in agent_ids}
        relative_utility_losses = {agent_id: 0 for agent_id in agent_ids}
        best_responses = {agent_id: None for agent_id in agent_ids}

        for agent_id in agent_ids:
            (
                actual_utilities[agent_id],
                utility_losses[agent_id],
                relative_utility_losses[agent_id],
                best_responses[agent_id],
            ) = self._get_agent_br_utility_loss_and_br(strategies, agent_id)

        return actual_utilities, utility_losses, relative_utility_losses, best_responses

    def verify_against_equilibrium(
        self, strategies: Dict[int, SABaseAlgorithm], agent_ids: List[int] = None
    ):
        """Use analytical equilibrium to estimate utility loss (exact ex post,
        still approximates interim and ex ante).
        """
        agent_ids = list(range(self.num_agents)) if agent_ids is None else agent_ids
        equ_strategies = self.env.model.equilibrium_strategies
        equilibrium_utilities = {agent_id: None for agent_id in agent_ids}
        utility_losses = {agent_id: None for agent_id in agent_ids}
        relative_utility_losses = {agent_id: None for agent_id in agent_ids}

        for agent_id in agent_ids:

            # 1. Calculate NE utility of this agent
            equilibrium_utility = self.estimate_agent_average_utility(
                equ_strategies, agent_id
            )

            # 2. Calculate utility under current strategy
            actual_vs_bne_strategies = deepcopy(equ_strategies)
            actual_vs_bne_strategies[agent_id] = strategies[agent_id]
            actual_utility = self.estimate_agent_average_utility(
                actual_vs_bne_strategies, agent_id
            )

            equilibrium_utilities[agent_id] = equilibrium_utility
            utility_losses[agent_id] = equilibrium_utility - actual_utility
            if equilibrium_utility != 0:
                relative_utility_losses[agent_id] = (
                    1 - actual_utility / equilibrium_utility
                )
            else:
                relative_utility_losses[agent_id] = -actual_utility

        return equilibrium_utilities, utility_losses, relative_utility_losses

    def _get_agent_br_utility_loss_and_br(self, learners, agent_id: int) -> Tuple:
        """
        Args:
            learners (_type_): holds agents' strategies
            agent_id (int): 

        Returns:
            Tuple: 
                estimated utility of learner
                estimated utility loss
                estimated relative utility loss
                best response strategies
        """
        information_tree = InformationSetTree(
            agent_id,
            self.env,
            self.env.model.verfier_env_info.discretizations,
            self.action_discretization,
            self.device,
        )
        num_done_sims = 0
        batch_size = min(self.num_simulations, self.batch_size)
        while num_done_sims <= self.num_simulations:
            try:
                print(
                    f"Starting verification of agent {agent_id} with batch size {batch_size}."
                )
                self._add_simulation_results_to_tree(
                    learners, agent_id, batch_size, information_tree
                )
                num_done_sims += batch_size
                io_ut.progress_bar(num_done_sims, self.num_simulations)
            except RuntimeError as e:
                ev_ut.catch_failed_simulation(self.device, batch_size, e)
                batch_size = int(batch_size / 2)
                self.clean_residuals()

        self.clean_residuals()
        utility_loss_estimates = information_tree.get_utility_loss_estimate()
        best_responses = information_tree.get_best_response_estimate()

        return (*utility_loss_estimates, best_responses)

    def clean_residuals(self):
        gc.collect()  # manually call gc to delete residuals
        torch.cuda.empty_cache()

    def get_rollout_utilities_from_states(
        self, states: torch.Tensor, learners: Dict[int, "Strategy"], agent_id: int
    ):
        """Given a strategy profile `learners` and initial states `states`,
        calculate the average utility.
        """
        batch_size, device = states.shape[0], states.device

        actual_utility = torch.zeros((batch_size,), device=device)

        for stage in range(self.num_stages):
            obs = self.env.model.get_observations(states)
            actions = th_ut.get_ma_actions(learners, obs)
            obs, rewards, _, states = self.env.model.compute_step(states, actions)
            actual_utility += rewards[agent_id]

        return actual_utility

    def estimate_agent_average_utility(
        self, learners: Dict[int, "Strategy"], agent_id: int
    ):
        """Given a strategy profile `learners` and an agent id `agent_id`,
        estimate his or her average utility over self.num_simulations rollouts.
        """
        actual_utility_tracker = UtilityTracker(agent_id, self.device)

        num_done_sims = 0
        batch_size = min(self.num_simulations, self.batch_size)
        while num_done_sims <= self.num_simulations:
            try:
                states = self.env.model.sample_new_states(batch_size).to(self.device)
                actual_utility = self.get_rollout_utilities_from_states(
                    states.clone(), learners, agent_id
                )
                actual_utility_tracker.add_utility(actual_utility)
                num_done_sims += batch_size
                io_ut.progress_bar(num_done_sims, self.num_simulations)
            except RuntimeError as e:
                ev_ut.catch_failed_simulation(self.device, batch_size, e)
                batch_size = int(batch_size / 2)
        actual_utility = actual_utility_tracker.get_mean_utility()

        return actual_utility

    def _add_simulation_results_to_tree(
        self,
        learners,
        agent_id: int,
        batch_size: int,
        information_tree: InformationSetTree,
    ):
        """Monte-Carlo approximation for a fixed discretized strategy for the
        given agent. We simulate batch_size games for this given strategy.

        Args:
            learners (_type_): holds agents' strategies
            agent_id (int):
            batch_size (int): 
            information_tree (InformationSetTree): game tree
        """

        # shape = (sim_size, *state_size)
        states = self.env.model.sample_new_states(batch_size).to(self.device)

        # A. Sample actual utility
        actual_utility = self.get_rollout_utilities_from_states(
            states.clone(), learners, agent_id
        )

        # B. Calculate best response utility
        best_response_utilities, sim_batch_indices = self.get_br_utilities_and_indices(
            learners, agent_id, batch_size, states
        )

        information_tree.add_simulation_results(
            best_response_utilities, sim_batch_indices, actual_utility
        )

    def get_br_utilities_and_indices(self, learners, agent_id, batch_size, states):
        best_response_utilities = torch.zeros(
            self._total_utilities_shape_for_batch(batch_size), device=self.device
        ).flatten()
        sim_size = batch_size  # The current size of the simulation

        agent_obs, opp_obs = self._split_obs(
            agent_id, self.env.model.get_observations(states)
        )
        episode_starts = torch.ones((batch_size,), dtype=bool, device=self.device)
        sim_batch_indices = torch.tensor([], device=self.device).long()

        for stage in range(self.num_stages):
            # Repeat states such that we can try out all discrete actions in
            # all states
            # shape = (sim_size, action_discretization, *state_size)
            sim_size_states = self._get_sim_size_states(states)

            # Repeat all discrete actions for all states / observations
            # shape = (sim_size, action_discretization, *action_size)
            sim_size_grid_actions = self._get_agent_grid_actions(agent_id, sim_size)

            # Dict with opp actions where their actions are repeated such that
            # the current player can try out all discrete actions against them
            # shape = (sim_size, action_discretization, *action_size)
            opp_actions = self._get_sim_size_opp_actions(
                learners, agent_id, opp_obs, episode_starts
            )

            # Combine and flatten actions
            combined_actions = self._get_combined_actions(
                agent_id, sim_size_grid_actions, opp_actions
            )

            # Create bins / indices for keeping track of observations & actions
            # shape = (sim_size * sims_to_be_made, 1)
            # where sims_to_be_made = action_discretization ** (num_stages
            #   - stage)
            sim_size_obs_bins = self._get_sim_size_obs_bins(agent_id, agent_obs, stage)
            sim_size_action_bins = self._get_sim_size_action_bins(sim_size, stage)
            sim_batch_indices = torch.cat(
                (sim_batch_indices, sim_size_obs_bins, sim_size_action_bins), dim=-1
            )

            # Flatten all simulations
            # shape = (sim_size * action_discretization, *state_size)
            flattened_states = sim_size_states.flatten(end_dim=1)

            # Simulate environment
            new_obs, rewards, dones, new_states = self.env.model.compute_step(
                flattened_states, combined_actions
            )
            # shape = (sim_size * self.action_discretization * sims_to_be_made)
            # where sims_to_be_made = action_discretization **
            #   (num_stages - (stage + 1))
            repeated_agent_rewards = self._repeat_rewards_and_flatten_to_full_stage_sim_size(
                rewards[agent_id], stage + 1
            )
            best_response_utilities += repeated_agent_rewards

            # Update variables for next game stage
            episode_starts = dones
            agent_obs, opp_obs = self._split_obs(agent_id, new_obs)
            states = new_states

            # sim_size increases by no. of alternative actions we try out
            # (branching factor of game tree)
            sim_size *= self.action_discretization
        assert (
            torch.all(dones).cpu().item()
        ), "All games should have ended after playing all rounds! Check num_stages of env!"
        return best_response_utilities, sim_batch_indices

    def _get_combined_actions(self, agent_id, sim_size_grid_actions, opp_actions):
        combined_actions = {}
        for agent_identifier in range(self.num_agents):
            if agent_identifier == agent_id:
                combined_actions[agent_identifier] = sim_size_grid_actions.flatten(
                    end_dim=1
                )
            else:
                combined_actions[agent_identifier] = opp_actions[
                    agent_identifier
                ].flatten(end_dim=1)

        return combined_actions

    def _get_sim_size_opp_actions(
        self,
        learners,
        agent_id: int,
        opp_obs: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Get opponent actions Dict[agent_id: (sim_size, action_size)] and repeat
        for any possible action taken by agent agent_id:
            Dict[agent_id: (sim_size, action_discretization, action_size)]
        Returns:
            Dict[int, torch.Tensor]:
        """
        opp_actions = self._get_opponent_actions(
            episode_starts, learners, agent_id, opp_obs
        )
        for opp_agent, opp_action in opp_actions.items():
            opp_actions[opp_agent] = th_ut.repeat_tensor_along_new_axis(
                data=opp_action, pos=[1], repeats=[self.action_discretization]
            )
        return opp_actions

    def _get_sim_size_states(self, states: torch.Tensor) -> torch.Tensor:
        """Repeat states for each possible action taken.
        Args:
            states (torch.Tensor): shape: (sim_size, )

        Returns:
            torch.Tensor: shape: (sim_size, action_discretization)
        """
        sim_size_states = th_ut.repeat_tensor_along_new_axis(
            data=states, pos=[1], repeats=[self.action_discretization]
        )

        return sim_size_states

    def _get_sim_size_action_bins(self, sim_size: int, stage: int) -> torch.Tensor:
        """Get repeated action bin indices to track tree path indices.
        Args:
            sim_size (int): 
            stage (int): 

        Returns:
            torch.Tensor: 
        """
        action_bins = torch.arange(self.action_discretization, device=self.device)
        repeated_action_bins = (
            th_ut.repeat_tensor_along_new_axis(
                data=action_bins,
                pos=[0, 2],
                repeats=[
                    sim_size,
                    self.action_discretization ** (self.num_stages - stage - 1),
                ],
            )
            .flatten()
            .unsqueeze(-1)
        )

        return repeated_action_bins

    def _get_sim_size_obs_bins(
        self, agent_id: int, agent_obs: torch.Tensor, stage: int
    ) -> torch.Tensor:
        """Indices of bins for each simulation of current stage. Repeat this, as
        this information is needed for all branching simulations to index the 
        final path followed by each simulation.

        Args:
            agent_id (int):
            agent_obs (torch.Tensor): 
            stage (int):

        Returns:
            torch.Tensor: (sim_size * sims_to_be_made, 1)
        """
        obs_bin_indices = self.env.model.get_obs_bin_indices(agent_id, agent_obs, stage)
        repeated_obs_bin_indices = self._repeat_obs_bins_and_flatten_to_full_stage_sim_size(
            obs_bin_indices, stage
        )

        return repeated_obs_bin_indices

    def _repeat_rewards_and_flatten_to_full_stage_sim_size(
        self, rewards: torch.Tensor, stage: int
    ) -> torch.Tensor:
        """Repeats a tensor to be the full size of the simulation.
        Args:
            data (torch.Tensor): _description_
            stage (int): _description_

        Returns:
            torch.Tensor: shape=(sim_size * sims_to_be_made)
        """
        pos_to_repeat = len(rewards.shape)
        return th_ut.repeat_tensor_along_new_axis(
            data=rewards,
            pos=[pos_to_repeat],
            repeats=[self.action_discretization ** (self.num_stages - stage)],
        ).flatten()

    def _repeat_obs_bins_and_flatten_to_full_stage_sim_size(
        self, obs_bins: torch.LongTensor, stage: int
    ) -> torch.Tensor:
        """Repeats a tensor to be the full size of the simulation.
        Args:
            data (torch.Tensor): _description_
            stage (int): _description_

        Returns:
            torch.Tensor: shape=(sim_size * sims_to_be_made)
        """
        return th_ut.repeat_tensor_along_new_axis(
            data=obs_bins,
            pos=[1],
            repeats=[self.action_discretization ** (self.num_stages - stage)],
        ).flatten(start_dim=0, end_dim=-2)

    def _get_agent_grid_actions(self, agent_id, cur_sim_size):
        """Repeats all grid actions from the environment to a size of
        `cur_sim_size`.
        Args:
            agent_id (int): agent ID
            cur_sim_size (int): simulation size

        Returns:
            torch.Tensor: shape=(cur_sim_size, action_discretization,
                *(env.action_size))
        """
        agent_grid_actions = self.env.get_action_grid(
            agent_id, grid_size=self.action_discretization
        )
        repeated_grid_actions = th_ut.repeat_tensor_along_new_axis(
            data=agent_grid_actions, pos=[0], repeats=[cur_sim_size]
        )

        return repeated_grid_actions

    def _split_obs(self, agent_id: int, obs: Dict):
        agent_obs = obs.pop(agent_id)
        opp_obs = obs
        return agent_obs, opp_obs

    def _get_opponent_actions(
        self, episode_starts: torch.Tensor, learners, agent_id, opp_obs
    ):
        states = {agent_id: None for agent_id in range(self.num_agents)}
        opp_actions = th_ut.get_ma_actions(
            learners, opp_obs, deterministic=True, excluded_agents=[agent_id]
        )

        return opp_actions

    def _total_utilities_shape_for_batch(self, batch_size: int) -> Tuple[int]:
        """Gives the shape of total utilities to be estimated in a single batch rollout.
        Args:
            batch_size (int):
        Returns:
            Tuple[int]: Shape of utility tensor
        """
        return (batch_size,) + tuple([self.action_discretization] * self.num_stages)
