from typing import Callable, Dict, List, Tuple

import torch

import src.utils.spaces_utils as sp_ut
from src.verifier.mean_utility_tracker import UtilityTracker


class InformationSetTree(object):
    def __init__(
        self,
        agent_id: int,
        env,
        obs_discretization: int,
        action_discretization: int,
        device,
    ) -> None:
        self.agent_id = agent_id
        self.env = env
        self.num_agents = self.env.model.num_agents
        self.obs_discretization = obs_discretization
        self.action_discretization = action_discretization
        self.num_rounds_to_play = self.env.model.num_rounds_to_play
        self.action_dim = env.model.ACTION_DIM
        self.device = device

        self.actual_utility_tracker = UtilityTracker(agent_id, device)

        self.stored_br_indices = self._init_stage_to_data_dict()
        self.best_responses = self._init_stage_to_data_dict()

        # Dict of obs shapes for all stages
        self.obs_discretization_shapes = self._get_obs_discretization_shapes()

        # Expand to account for all possible actions
        self.all_nodes_shape = self._get_all_nodes_shape()

        # Initialize all game tree node utilities & visitation counts to zero
        self.nodes_utility_estimates: torch.Tensor = self._init_nodes_utility_estimates()
        self.nodes_counts = self._init_nodes_utility_estimates().long()

        self.max_nodes_index = self.nodes_counts.shape[0]
        assert (
            self.max_nodes_index
            == torch.prod(torch.tensor(list(self.all_nodes_shape))).item()
        )

    def _init_stage_to_data_dict(self):
        return {stage: None for stage in range(self.num_rounds_to_play)}

    def _get_obs_discretization_shapes(self) -> Dict[int, Tuple[int]]:
        return {
            stage: self.env.model.get_obs_discretization_shape(
                self.agent_id, self.obs_discretization, stage
            )
            for stage in range(self.num_rounds_to_play)
        }

    def _get_all_nodes_shape(self) -> int:
        all_nodes_shape = tuple()
        for _, obs_shape in self.obs_discretization_shapes.items():
            all_nodes_shape += obs_shape + (self.action_discretization,)
        return all_nodes_shape

    def _init_nodes_utility_estimates(self) -> torch.Tensor:
        return torch.zeros(self.all_nodes_shape, device=self.device).flatten()

    def _init_max_nodes_index(self) -> List:
        return torch.prod(torch.tensor(list(self.all_nodes_shape))).item()

    def add_simulation_results(
        self,
        sim_utilities: torch.Tensor,
        sim_indices: torch.LongTensor,
        actual_utilities: torch.Tensor,
    ):
        # Ravel the stage-wise observation indices to indexing the whole game tree
        flat_indices = sp_ut.ravel_multi_index(sim_indices, self.all_nodes_shape)

        # Add utilities to cumulative utilities
        self.nodes_utility_estimates.index_add_(0, flat_indices, sim_utilities)

        # Increment visitation counts
        counts = torch.ones(
            flat_indices.shape, device=self.device, dtype=torch.long
        )  # TODO: expand instead creating big tensor
        self.nodes_counts.index_add_(0, flat_indices, counts)

        # store actual utilities to tracker
        self.actual_utility_tracker.add_utility(actual_utilities)

    def get_utility_loss_estimate(self) -> Tuple[float]:
        """Iteratively compute best-response utility estimate.
        Shape of self.nodes_utility_estimates = (N_V^1, N_A, ..., N_V^k, N_A).
        Visitation counts are identical in size. Iterate reversely over stages:
        1. max over last dim (br given previous trajectory)
        2. calculate visitation probabilities over obs
        3. weight utilities of br by visitation probabilities
        Returns:
            Tuple(float): 
                estimated utility of learner
                estimated utility loss
                estimated relative utility loss
        """
        assert (
            self.env.model.ACTION_DIM == 1
        ), "We assume one dimensional action spaces in all rounds!"

        # Utility estimate at last/terminal stage for all simulations
        estimated_br_utilities = self.calc_monte_carlo_utility_estimations()
        nodes_counts = self.nodes_counts.view(self.all_nodes_shape)

        # Backwards traversal of game tree
        for stage in reversed(range(self.num_rounds_to_play)):

            # Select action with highest utility
            estimated_br_utilities, br_indices = torch.max(
                estimated_br_utilities, dim=-1
            )

            self.stored_br_indices[stage] = br_indices.clone()
            # TODO: Do I need the clone here?

            # Weight the utilities by their reach probabilities (sample mean of
            # utility for previous stage following the BR actions)
            visitation_probabilities, nodes_counts = self._calc_visitation_probabilities_and_update_nodes_counts(
                nodes_counts, br_indices, stage
            )
            estimated_br_utilities *= visitation_probabilities

            # Sum over all possible states of this stage (chance node in
            # game tree)
            estimated_br_utilities = estimated_br_utilities.sum(
                dim=self._get_stage_obs_summing_dim(stage)
            )

        estimated_br_utilities = estimated_br_utilities.item()

        # Now we have a scalar estimate of the utility when playing the BR
        self._calculate_best_responses()
        actual_utility_estimate = self.actual_utility_tracker.get_mean_utility()

        # Final ex ante utility loss
        estimated_utility_loss = estimated_br_utilities - actual_utility_estimate

        # Relative ex ante utility loss
        if estimated_br_utilities == 0:  # catch div. by 0
            estimated_relative_util_loss = 1
        else:
            estimated_relative_util_loss = (
                1 - actual_utility_estimate / estimated_br_utilities
            )

        return (
            actual_utility_estimate,
            estimated_utility_loss,
            estimated_relative_util_loss,
        )

    def _calc_visitation_probabilities_and_update_nodes_counts(
        self, nodes_counts: torch.LongTensor, br_indices: torch.LongTensor, stage: int
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        1. Calculate the visitation probabilities for each branch of the current stage (=depth in tree). 
        2. Update the nodes_counts for next iteration (to tree depth - 1)
        # TODO: Separate these two things without redundant operations.
        Args:
            nodes_counts (torch.LongTensor): 
            br_indices (torch.LongTensor): best-response indices of current stage 
            stage (int): depth in information tree

        Returns:
            Tuple[torch.Tensor, torch.LongTensor]: visitation_probabilities, updated_node_counts
        """

        # Subselect the counts for the paths reaching the BR action
        nodes_counts = torch.gather(
            nodes_counts, -1, br_indices.unsqueeze(-1)
        ).squeeze()

        # Sum counts over all states for counts of taking actions iny previous stage
        summing_dim = self._get_stage_obs_summing_dim(stage)
        nodes_counts_obs_sum = nodes_counts.sum(summing_dim, keepdim=True)

        # Calculate visitation probabilities
        # TODO: Possibly cleaner with Einstein notation and no expansion(?)
        expansion_dim = (
            tuple(
                [-1]
                * (
                    len(nodes_counts_obs_sum.shape)
                    - len(self.obs_discretization_shapes[stage])
                )
            )
            + self.obs_discretization_shapes[stage]
        )
        expanded_nodes_counts_obs_sum = nodes_counts_obs_sum.expand(expansion_dim)

        mask = expanded_nodes_counts_obs_sum > 0
        visitation_probabilities = nodes_counts.float()
        visitation_probabilities[mask] /= expanded_nodes_counts_obs_sum[mask]

        return visitation_probabilities, nodes_counts_obs_sum.squeeze()

    def _get_stage_obs_summing_dim(self, stage):
        return tuple(
            [-(k + 1) for k in range(len(self.obs_discretization_shapes[stage]))]
        )

    def calc_monte_carlo_utility_estimations(self):
        averaged_utilities = torch.zeros(
            self.all_nodes_shape, device=self.device
        ).flatten()  # TODO: Can we prevent new allocation/copy?
        mask = self.nodes_counts > 0
        averaged_utilities[mask] = (
            self.nodes_utility_estimates[mask] / self.nodes_counts[mask]
        )
        return averaged_utilities.view(self.all_nodes_shape)

    def _calculate_best_responses(self):
        prev_stage_br_slice = None
        for stage, br_indices in self.stored_br_indices.items():
            self.best_responses[
                stage
            ], prev_stage_br_slice = self._calculate_stage_br_from_indices(
                stage, br_indices, prev_stage_br_slice
            )

    def _calculate_stage_br_from_indices(
        self,
        stage: int,
        br_indices: torch.LongTensor,
        prev_stage_br_slice: torch.LongTensor,
    ) -> Tuple[Callable, torch.LongTensor]:
        if stage == 0:

            def best_response(agent_obs: torch.Tensor):
                # NOTE: This may be highly inefficient as one needs to keep all br_indices in memory!
                obs_bin_indices = self.env.model.get_obs_bin_indices(
                    agent_obs, self.agent_id, stage, self.obs_discretization
                )
                action_bins = br_indices[obs_bin_indices.squeeze()]
                agent_grid_actions = self.env.get_action_grid(
                    self.agent_id, grid_size=self.action_discretization
                )
                return agent_grid_actions[action_bins]

        else:

            def best_response(agent_obs: torch.Tensor):
                pass

        return best_response, prev_stage_br_slice

    def get_best_response_estimate(self):
        return self.best_responses
