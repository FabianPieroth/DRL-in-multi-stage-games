from typing import Dict, List, Tuple

import torch

import src.utils_folder.spaces_utils as sp_ut


class DiscreteStrategyEnumerator(object):
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
        self.all_strategies_shape = self._get_all_strategies_shape()
        self.strategy_utility_estimates = self._init_strategy_utility_estimates()
        self.max_strategy_index = self._init_max_strategy_index()
        self.device = device

    def _get_obs_discretization_shapes(self) -> Dict[int, Tuple[int]]:
        return {
            stage: self.env.model.get_obs_discretization_shape(
                self.agent_id, self.obs_discretization, stage
            )
            for stage in range(self.num_rounds_to_play)
        }

    def get_strategy_index_iterator(self) -> List:
        """We can use this method to exclude certain strategies to consider in the search."""
        return range(self.max_strategy_index)

    def _get_all_strategies_shape(self) -> int:
        all_strategies_shape = tuple()
        for _, obs_shape in self.obs_discretization_shapes.items():
            all_strategies_shape += obs_shape + (self.action_discretization,)
        return all_strategies_shape

    def _init_strategy_utility_estimates(self) -> torch.Tensor:
        return torch.zeros(self.all_strategies_shape, device=self.device)

    def _init_max_strategy_index(self) -> List:
        return torch.prod(torch.tensor(list(self.all_strategies_shape))).item()

    def get_actions_for_strategy(
        self, strategy_index, agent_obs: torch.Tensor, agent_id
    ) -> torch.Tensor:
        pass

    def add_strategy_results(self, strategy_index, utilities: torch.Tensor):
        pass

    def get_br_utility_estimate(self):
        pass

    def get_best_response_estimate(self):
        pass
