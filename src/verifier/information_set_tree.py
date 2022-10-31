from typing import Callable, Dict, List, Tuple

import torch

import src.utils_folder.spaces_utils as sp_ut


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
        self.device = self.env.device

        self.stored_br_indices = self._init_stage_to_data_dict()
        self.best_responses = self._init_stage_to_data_dict()

        self.obs_discretization_shapes = self._get_obs_discretization_shapes()
        self.all_nodes_shape = self._get_all_nodes_shape()
        self.nodes_utility_estimates = self._init_nodes_utility_estimates()
        self.nodes_counts = self._init_nodes_utility_estimates().long()
        self.max_nodes_index = self._init_max_nodes_index()
        self.device = device

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
        self, utilities: torch.Tensor, indices: torch.LongTensor
    ):
        flat_indices = sp_ut.ravel_multi_index(indices, self.all_nodes_shape)
        self.nodes_utility_estimates.index_add_(0, flat_indices, utilities)

        counts = torch.ones(
            flat_indices.shape, device=self.device
        ).long()  # TODO: expand instead creating big tensor
        self.nodes_counts.index_add_(0, flat_indices, counts)

    def get_br_utility_estimate(self):
        """Iteratively compute best-response utility estimate.
        Shape of self.nodes_utility_estimates = (N_V^1, N_A, ..., N_V^k, N_A).
        Visitation counts are identical in size. Iterate reversely over stages:
        1. max over last dim (br given previous trajectory)
        2. calculate visition probabilities over obs
        3. weight utilities of br by visitation probabilities
        """
        # NOTE: We assume one dimensional action spaces in all rounds!
        estimated_utilities = self.calc_monte_carlo_utility_estimations()
        nodes_counts = self.nodes_counts.reshape(self.all_nodes_shape)

        for stage in reversed(range(self.num_rounds_to_play)):

            estimated_utilities, br_indices = torch.max(estimated_utilities, dim=-1)

            self.stored_br_indices[
                stage
            ] = br_indices.clone()  # TODO: Do I need the clone here?

            visitation_probabilities, nodes_counts = self._calc_visitation_probabilities_and_update_nodes_counts(
                nodes_counts, br_indices, stage
            )
            estimated_utilities = estimated_utilities * visitation_probabilities

            estimated_utilities = estimated_utilities.sum(
                dim=self._get_stage_obs_summing_dim(stage)
            )

        self._calculate_best_responses()

        return estimated_utilities.item()

    def _calc_visitation_probabilities_and_update_nodes_counts(
        self, nodes_counts: torch.LongTensor, br_indices: torch.LongTensor, stage: int
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        1. Calculate the visitation probabilities for each branch of the current stage (=depth in tree). 
        2. Update the nodes_counts for next iteration (to tree depth - 1)
        #TODO: Separate these two things without redundant operations.
        Args:
            nodes_counts (torch.LongTensor): 
            br_indices (torch.LongTensor): best-response indices of current stage 
            stage (int): depth in information tree

        Returns:
            Tuple[torch.Tensor, torch.LongTensor]: visitation_probabilities, updated_node_counts
        """
        nodes_counts = torch.gather(
            nodes_counts, -1, br_indices.unsqueeze(-1)
        ).squeeze()
        summing_dim = self._get_stage_obs_summing_dim(stage)
        nodes_counts_obs_sum = nodes_counts.sum(summing_dim, keepdim=True)
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

        visitation_probabilities = nodes_counts.float()

        visitation_probabilities[
            expanded_nodes_counts_obs_sum > 0
        ] /= expanded_nodes_counts_obs_sum[expanded_nodes_counts_obs_sum > 0]

        return visitation_probabilities, nodes_counts_obs_sum.squeeze()

    def _get_stage_obs_summing_dim(self, stage):
        return tuple(
            [-(k + 1) for k in range(len(self.obs_discretization_shapes[stage]))]
        )

    def calc_monte_carlo_utility_estimations(self):
        averaged_utilities = torch.zeros(
            self.all_nodes_shape, device=self.device
        ).flatten()  # TODO: Can we prevent new allocation/copy?
        averaged_utilities[self.nodes_counts > 0] = (
            self.nodes_utility_estimates[self.nodes_counts > 0]
            / self.nodes_counts[self.nodes_counts > 0]
        )
        averaged_utilities.reshape(self.all_nodes_shape)
        return averaged_utilities.reshape(self.all_nodes_shape)

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
