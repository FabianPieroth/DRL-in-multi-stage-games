from typing import Dict, List, Tuple

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

        self.obs_discretization_shapes = self._get_obs_discretization_shapes()
        self.all_nodes_shape = self._get_all_nodes_shape()
        self.nodes_utility_estimates = self._init_nodes_utility_estimates()
        self.nodes_counts = self._init_nodes_utility_estimates().long()
        self.max_nodes_index = self._init_max_nodes_index()
        self.device = device

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
        averaged_utilities = torch.zeros(
            self.all_nodes_shape, device=self.device
        )  # TODO: Can we prevent new allocation/copy?
        averaged_utilities[self.nodes_counts > 0] = (
            self.nodes_utility_estimates[self.nodes_counts > 0]
            / self.nodes_counts[self.nodes_counts > 0]
        )
        reshaped_utilities = averaged_utilities.reshape(self.all_nodes_shape)
        # TODO: determine correct best-response utilities
        pass

    def get_best_response_estimate(self):
        pass
