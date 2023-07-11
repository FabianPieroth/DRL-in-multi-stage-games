from typing import Any, Dict

import torch
from gym import spaces
from gym.spaces import Space

import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import BaseEnvForVec

ROCK = 0
PAPER = 1
SCISSORS = 2


class RockPaperScissors(BaseEnvForVec):
    """Iterated RockPaperScissors game as simple env example."""

    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = None):
        super().__init__(config, device)
        self.state_shape = (self.num_agents, 2)
        self.action_space_sizes = self._init_action_space_sizes()
        self.variable_num_rounds = self.config["variable_num_rounds"]

        self.num_rounds_to_play = self.config["num_rounds_to_play"]

    def _get_num_rounds_to_play(self, num: int) -> torch.Tensor:
        if self.config["variable_num_rounds"]:
            return torch.randint(
                low=self.config["num_rounds_to_play"] - 1,
                high=self.config["num_rounds_to_play"] + 2,
                size=(num,),
                device=self.device,
            )
        else:
            return (
                torch.ones((num,), device=self.device)
                * self.config["num_rounds_to_play"]
            )

    def _get_num_agents(self) -> int:
        return self.config["num_agents"]

    def _init_observation_spaces(self) -> Dict[int, Space]:
        return {
            agent_id: spaces.Box(0, self.config["num_rounds_to_play"], shape=(2,))
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
        states[:, :, -1] = self._get_num_rounds_to_play(n)[:, None]
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

        new_states = cur_states.detach().clone()
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
        rewards[
            torch.logical_and(paper_wins.unsqueeze(1), actions == PAPER).view_as(
                rewards
            )
        ] = 1.0

        # Case 2: Paper vs Scissors
        scissors_wins = self.first_and_second_not_third(
            paper_played, scissors_played, rock_played
        )
        rewards[
            torch.logical_and(scissors_wins.unsqueeze(1), actions == SCISSORS).view_as(
                rewards
            )
        ] = 1.0

        # Case 3: Scissors vs Rock
        rock_wins = self.first_and_second_not_third(
            scissors_played, rock_played, paper_played
        )
        rewards[
            torch.logical_and(rock_wins.unsqueeze(1), actions == ROCK).view_as(rewards)
        ] = 1.0

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

    def custom_evaluation(
        self, learners, env, writer=None, iteration: int = 0, config: Dict = None
    ):
        deterministic = True
        episode_iter = 0
        observations = env.reset()

        freq_dict = {
            agent_id: {
                "rock_freq": 0,
                "paper_freq": 0,
                "scissors_freq": 0,
                "total_num_actions": 0,
            }
            for agent_id in range(env.model.num_agents)
        }

        while episode_iter < 10:
            actions = th_ut.get_ma_actions(
                learners, observations, deterministic=deterministic
            )
            observations, rewards, dones, infos = env.step(actions)
            for agent_id in range(env.model.num_agents):
                (
                    num_rock,
                    num_paper,
                    num_scissors,
                    num_total,
                ) = RockPaperScissors.get_action_nums(actions[agent_id])
                freq_dict[agent_id]["rock_freq"] += num_rock
                freq_dict[agent_id]["paper_freq"] += num_paper
                freq_dict[agent_id]["scissors_freq"] += num_scissors
                freq_dict[agent_id]["total_num_actions"] += num_total

            episode_starts = dones
            if dones.any().detach().item():
                episode_iter += 1
        for agent_id, learner in learners.items():
            learner.logger.record(
                "eval/rock_freq",
                (
                    freq_dict[agent_id]["rock_freq"]
                    / freq_dict[agent_id]["total_num_actions"]
                )
                .detach()
                .item(),
            )
            learner.logger.record(
                "eval/paper_freq",
                (
                    freq_dict[agent_id]["paper_freq"]
                    / freq_dict[agent_id]["total_num_actions"]
                )
                .detach()
                .item(),
            )
            learner.logger.record(
                "eval/scissors_freq",
                (
                    freq_dict[agent_id]["scissors_freq"]
                    / freq_dict[agent_id]["total_num_actions"]
                )
                .detach()
                .item(),
            )

    @staticmethod
    def get_action_nums(sa_actions):
        num_rock = torch.sum(sa_actions == ROCK)
        num_paper = torch.sum(sa_actions == PAPER)
        num_scissors = torch.sum(sa_actions == SCISSORS)
        num_total = torch.numel(sa_actions)
        return num_rock, num_paper, num_scissors, num_total

    def render(self, state):
        return state

    def __str__(self):
        return "RockPaperScissors"
