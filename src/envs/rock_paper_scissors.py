from typing import Any, Dict

import numpy as np
import torch
from gym import spaces

import src.utils_folder.spaces_utils as sp_ut
from src.base_env_for_vec_env import BaseEnvForVec


class RockPaperScissors(BaseEnvForVec):
    """Iterated RockPaperScissors Game as simple env example."""

    def __init__(self, config: Dict, device):
        super().__init__(config, device)
        self.rl_env_config = config["rl_envs"]
        self.num_rounds_to_play = self.rl_env_config["num_rounds_to_play"]
        self.num_agents = self.rl_env_config["num_agents"]
        self.observation_space = spaces.Box(
            0, self.num_rounds_to_play + 1, shape=(2, self.num_agents)
        )

        self.action_space_sizes = tuple([3 for k in range(self.num_agents)])
        self.action_space = spaces.Discrete(np.prod(self.action_space_sizes))

    def to(self, device) -> Any:
        self.device = device
        self.action_space_nvec = self.action_space_nvec.to(device)
        return self

    def sample_new_states(self, n: int) -> Any:
        """Creates states in shape=(num_envs, 2)
        The 2 stands for current_round and num_rounds_to_play
        """
        states = torch.zeros((n, 2))
        states[:, -1] = self.num_rounds_to_play
        return states

    def compute_step(self, cur_states, actions: torch.Tensor):
        """ Returns:
            observations:
            rewards:
            episode-done markers:
            updated_states:"""
        unraveled_actions = sp_ut.unravel_index(
            actions, self.action_space_sizes
        ).squeeze()

        new_states = cur_states.detach().clone()
        new_states[:, 0] += 1

        dones = new_states[:, 0] >= new_states[:, 1]

        observations = self.get_observations(new_states)

        rewards = self._compute_rewards(unraveled_actions)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, unraveled_actions: torch.Tensor) -> torch.Tensor:
        """Computes the rewards for the played games of Rock-Paper-Scissors
        0: Rock
        1: Paper
        2: Scissors
        We have a cycle Rock < Paper < Scissors < Rock.

        Args:
            unraveled_actions (torch.Tensor): shape=(num_envs, num_agents)

        Returns:
            torch.Tensor: shape=(num_envs, num_agents)
            winning: +1
            losing: -1
            draw: 0
            If all three options occur, there is always a draw.
        """
        rewards = torch.zeros(unraveled_actions.shape, device=self.device)
        rock_played = torch.any(unraveled_actions == 0, dim=1)
        paper_played = torch.any(unraveled_actions == 1, dim=1)
        scissors_played = torch.any(unraveled_actions == 2, dim=1)

        # Case 1: Rock vs Paper
        paper_wins = self.first_and_second_not_third(
            rock_played, paper_played, scissors_played
        )
        rewards[
            torch.logical_and(paper_wins.unsqueeze(1), unraveled_actions == 1)
        ] = 1.0

        # Case 2: Paper vs Scissors
        scissors_wins = self.first_and_second_not_third(
            paper_played, scissors_played, rock_played
        )
        rewards[
            torch.logical_and(scissors_wins.unsqueeze(1), unraveled_actions == 2)
        ] = 1.0

        # Case 3: Scissors vs Rock
        rock_wins = self.first_and_second_not_third(
            scissors_played, rock_played, paper_played
        )
        rewards[torch.logical_and(rock_wins.unsqueeze(1), unraveled_actions == 0)] = 1.0

        return rewards

    @staticmethod
    def first_and_second_not_third(
        first: torch.Tensor, second: torch.Tensor, third: torch.Tensor
    ) -> torch.Tensor:
        return torch.logical_and(torch.logical_and(first, second), ~third)

    def get_observations(self, states) -> Any:
        """
        Args:
            states (_type_): shape=(num_samples, state_dim)

        Returns:
            Any: observations.shape=(num_samples, num_agents, state_dim)
        """
        return states.unsqueeze(1).repeat(1, self.num_agents, 1)

    def render(self, state):
        return state
