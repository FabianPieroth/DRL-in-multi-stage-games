from typing import Any, Dict

import numpy as np
import torch
from gym import spaces

import src.utils_folder.spaces_utils as sp_ut
from src.envs.torch_vec_env import BaseEnvForVec


def eval_rps_strategy(env, player_position, eval_strategy):
    """Evaluate a given RPS strategy."""
    env_player_position = env.model.player_position
    env.model.player_position = player_position

    obs = env.reset()
    i = 1
    while True:
        action = eval_strategy(obs)
        obs, reward, done, info = env.step(action)
        print(
            f"action frequencies player {player_position} round {i}/{env.model.num_rounds_to_play}:",
            [round((action == i).sum().item() / action.shape[0], 2) for i in range(3)],
        )
        if done.all():
            break
        i += 1

    # Reset `player_position` of env
    env.model.player_position = env_player_position

    return


class RockPaperScissors(BaseEnvForVec):
    """Iterated RockPaperScissors game as simple env example.

    This environment keeps track of the strategies of the participants and the
    position of the current activa player. From the outside, this env always
    looks like a single-agent env from the perspective of the active player.
    This allows for simulatenous learning of all agents with dynamically
    changing strategies of players via a `MultiAgentCoordinator`.

    NOTE: Also see
    https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/rps/rps.py
    for a peeting zoo (multi-agent) implementation. (Noticed after creating
    this class.)
    """

    def __init__(self, config: Dict, player_position: int = 0, device: str = None):
        super().__init__(config, device)
        self.rl_env_config = config["rl_envs"]
        self.num_rounds_to_play = self.rl_env_config["num_rounds_to_play"]

        # single-agent learning
        self.num_agents = 2  # self.rl_env_config["num_agents"]

        self.state_dim = 2
        self.observation_space = spaces.Box(
            0, self.num_rounds_to_play + 1, shape=(self.state_dim,)
        )

        self.action_space_size = 1
        self.action_space_sizes = (3,)
        self.action_space = spaces.Discrete(np.prod(self.action_space_sizes))

        # positions
        self.player_position = player_position

        # dummy strategies: these can be overwritten by strategies that are
        # learned over time in repeated self-play.
        self.strategies = [
            lambda obs: torch.zeros(
                (obs.shape[0], self.action_space_size), device=obs.device
            )
            for _ in range(self.num_agents)
        ]

    def to(self, device) -> Any:
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states.

        :param n: Batch size of how many games are played in parallel.        
        :return: the new states, in shape=(n, num_agents, 2), where 2 stands
            for `current_round` and `num_rounds_to_play`.
        """
        states = torch.zeros((n, self.num_agents, 2), device=self.device)
        states[:, :, -1] = self.num_rounds_to_play
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
        unraveled_actions = sp_ut.unravel_index(actions, self.action_space_sizes)
        unraveled_actions = unraveled_actions.view(-1, 1, self.action_space_size)

        # Append opponents' actions
        unraveled_actions = unraveled_actions.repeat(1, self.num_agents, 1)
        for opponent_position, opponent_strategy in enumerate(self.strategies):
            if opponent_position != self.player_position:
                opponent_obs = self.get_observations(
                    cur_states, player_position=opponent_position
                )
                unraveled_actions[:, opponent_position, :] = opponent_strategy(
                    opponent_obs
                ).view(-1, self.action_space_size)

        # States are for all agents, obs and rewards for `player_position` only
        new_states = cur_states.detach().clone()
        new_states[:, :, 0] += 1

        # Reached last stage? (Independent from `player_position`)
        dones = new_states[:, 0, 0] >= new_states[:, 0, 1]

        observations = self.get_observations(new_states)

        rewards = self._compute_rewards(unraveled_actions)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, unraveled_actions: torch.Tensor) -> torch.Tensor:
        """Computes the rewards for the played games of Rock-Paper-Scissors for
        the player at `self.player_position`.

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
        rewards = torch.zeros(
            (unraveled_actions.shape[0], self.num_agents),
            device=unraveled_actions.device,
        )
        rock_played = torch.any(unraveled_actions == 0, dim=1)
        paper_played = torch.any(unraveled_actions == 1, dim=1)
        scissors_played = torch.any(unraveled_actions == 2, dim=1)

        # Case 1: Rock vs Paper
        paper_wins = self.first_and_second_not_third(
            rock_played, paper_played, scissors_played
        )
        rewards[torch.logical_and(paper_wins, unraveled_actions.squeeze() == 1)] = 1.0

        # Case 2: Paper vs Scissors
        scissors_wins = self.first_and_second_not_third(
            paper_played, scissors_played, rock_played
        )
        rewards[
            torch.logical_and(scissors_wins, unraveled_actions.squeeze() == 2)
        ] = 1.0

        # Case 3: Scissors vs Rock
        rock_wins = self.first_and_second_not_third(
            scissors_played, rock_played, paper_played
        )
        rewards[torch.logical_and(rock_wins, unraveled_actions.squeeze() == 0)] = 1.0

        # single-agent
        rewards = rewards[:, self.player_position]

        return rewards

    @staticmethod
    def first_and_second_not_third(
        first: torch.Tensor, second: torch.Tensor, third: torch.Tensor
    ) -> torch.Tensor:
        """first and second not third"""
        return torch.logical_and(torch.logical_and(first, second), ~third)

    def get_observations(
        self, states: torch.Tensor, player_position: int = None
    ) -> torch.Tensor:
        """Return the observations at the player at `player_position`.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :param player_position: Needed when called for one of the static
            (non-learning) strategies.
        :returns observations: Observations of shape (num_env, num_agents,
            state_dim).
        """
        if player_position is None:
            player_position = self.player_position

        return states.view(-1, self.num_agents, self.state_dim)[:, player_position, :]

    def render(self, state):
        return state

    def seed(self, seed: int):
        """Set seeds."""
        torch.manual_seed(seed)
        np.random.seed(seed)
