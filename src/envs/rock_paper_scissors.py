from typing import Any, Dict

import torch
from gym import spaces
from gym.spaces import Space

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


ROCK = 0
PAPER = 1
SCISSORS = 2


class RockPaperScissors(BaseEnvForVec):
    """Iterated RockPaperScissors game as simple env example.
    """

    def __init__(self, config: Dict, device: str = None):
        super().__init__(config, device)
        self.rl_env_config = config
        self.num_rounds_to_play = self.rl_env_config["num_rounds_to_play"]

        self.num_agents = self.rl_env_config["num_agents"]
        self.state_shape = (self.num_agents, 2)

        self.observation_spaces = self._init_observation_spaces()
        self.action_spaces = self._init_action_spaces()
        self.action_space_sizes = self._init_action_space_sizes()

    def _init_observation_spaces(self) -> Dict[int, Space]:
        return {
            agent_id: spaces.Box(0, self.num_rounds_to_play, shape=(2,))
            for agent_id in range(self.num_agents)
        }

    def _init_action_spaces(self) -> Dict[int, Space]:
        return {agent_id: spaces.Discrete(3) for agent_id in range(self.num_agents)}

    def _init_action_space_sizes(self) -> Dict[int, int]:
        return {agent_id: 3 for agent_id in range(self.num_agents)}

    def to(self, device) -> Any:
        self.device = device
        return self

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states.

        :param n: Batch size of how many games are played in parallel.        
        :return: the new states, in shape=(n, num_agents, 2), where 2 stands
            for `current_round` and `num_rounds_to_play`.
        """
        shape_to_sample = (n,) + self.state_shape
        states = torch.zeros(shape_to_sample, device=self.device)
        states[:, :, -1] = self.num_rounds_to_play
        return states

    def compute_step(self, cur_states, actions: Dict[int, torch.Tensor]):
        """Compute a step in the game.
        
        :param cur_states: The current states of the games.
        :param actions: Dict[agent_id, actions]
        :return observations:
        :return rewards:
        :return episode-done markers:
        :return updated_states:
        """
        """unraveled_actions = sp_ut.unravel_index(actions, self.action_space_sizes)
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
                ).view(-1, self.action_space_size)"""

        new_states = cur_states.detach().clone()
        # current_round = int(new_states[0, 0, -2].detach().cpu())
        # new_states[:, :, current_round] = actions  # Store played actions
        new_states[:, :, -2] += 1.0  # Add current round

        # Reached last stage? (Independent from agent)
        # TODO: dones cannot handle individual agents finishing before episode ends!
        dones = new_states[:, 0, -2] >= new_states[:, 0, -1]

        observations = self.get_observations(new_states)

        rewards = self._compute_rewards(actions)

        return observations, rewards, dones, new_states

    def _compute_rewards(self, actions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Computes the rewards for the played games of Rock-Paper-Scissors for
        the player at `self.player_position`.

        0: Rock
        1: Paper
        2: Scissors
        We have a cycle Rock < Paper < Scissors < Rock.

        Args:
            actions Dict[int, torch.Tensor]: each tensor of shape (num_envs, )

        Returns:
            torch.Tensor: shape=(num_envs, num_agents)
            winning: +1
            losing: -1
            draw: 0
            If all three options occur, there is always a draw.
        """

        # stack actions to single tensor
        actions = torch.stack([sa_action for sa_action in actions.values()], dim=1)

        rewards = torch.zeros(
            (actions.shape[0], self.num_agents), device=actions.device
        )
        rock_played = torch.any(actions == ROCK, dim=1)
        paper_played = torch.any(actions == PAPER, dim=1)
        scissors_played = torch.any(actions == SCISSORS, dim=1)

        # Case 1: Rock vs Paper
        paper_wins = self.first_and_second_not_third(
            rock_played, paper_played, scissors_played
        )
        rewards[torch.logical_and(paper_wins.unsqueeze(1), actions == PAPER)] = 1.0

        # Case 2: Paper vs Scissors
        scissors_wins = self.first_and_second_not_third(
            paper_played, scissors_played, rock_played
        )
        rewards[
            torch.logical_and(scissors_wins.unsqueeze(1), actions == SCISSORS)
        ] = 1.0

        # Case 3: Scissors vs Rock
        rock_wins = self.first_and_second_not_third(
            scissors_played, rock_played, paper_played
        )
        rewards[torch.logical_and(rock_wins.unsqueeze(1), actions == ROCK)] = 1.0

        return {agent_id: rewards[:, agent_id] for agent_id in range(self.num_agents)}

    @staticmethod
    def first_and_second_not_third(
        first: torch.Tensor, second: torch.Tensor, third: torch.Tensor
    ) -> torch.Tensor:
        """first and second not third"""
        return torch.logical_and(torch.logical_and(first, second), ~third)

    def get_observations(self, states: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Return the observations of the players.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :returns observations: Dict of Observations of shape (num_env, num_agents, 2).
        """

        return {agent_id: states[:, agent_id, :] for agent_id in range(self.num_agents)}

    def render(self, state):
        return state
