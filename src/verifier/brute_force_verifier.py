"""Verifier"""
from typing import Dict

import torch
from tqdm import tqdm

from src.learners.base_learner import SABaseAlgorithm


class BFVerifier:
    """Verifier that tries out a grid of alternative actions and choses the
    best combination as approximation to a best response.
    """

    def __init__(self, env, num_envs: int = 8, num_alternative_actions: int = 4):
        self.env = env
        self.num_agents = self.env.model.num_agents

        self.device = self.env.device

        # TODO: information should be read from env
        self.action_range = [0, 1]

        self.num_own_envs = num_envs
        self.num_opponent_envs = num_envs
        self.num_alternative_actions = num_alternative_actions
        self.num_rounds_to_play = self.env.model.num_rounds_to_play

    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        """Loop over the current player's valuation. This can be done to
        sequentialize some of the computation for reduced memory consumption.
        """
        utility_loss = torch.zeros(self.num_agents)
        for i in tqdm(range(self.num_own_envs)):
            utility_loss += torch.tensor(
                [
                    l / self.num_own_envs
                    for l in self._verify(learners, 1, self.num_opponent_envs)
                ]
            )
        return utility_loss.tolist()

    def _verify(
        self,
        learners: Dict[int, SABaseAlgorithm],
        num_own_envs: int,
        num_opponent_envs: int,
    ):
        """For each player we sample `num_own_envs` times from its prior
        and try out all combination of actions (over all stages) and average
        over `num_opponent_envs` samples from the opponents' priors. In the
        limit, this should converge to the best response (and the corresponding)
        utility against the current opponent strategies. Therefore, the loss
        should be zero in BNE.

        Args:
            learners (dict[SABaseAlgorithm]): Learners whose current strategies
                shall be evaluated against one another.

        Returns:
            list: Utility loss that each agent "left on the table" against the
                current opponents.
        """
        num_total_envs: int = (
            num_own_envs
            * num_opponent_envs
            * (self.num_alternative_actions ** self.num_rounds_to_play + 1)
        )

        utility_loss = [None] * self.num_agents

        # Dimensions (opponent_batch_size, [actual_actions, alternative_actions])
        for player_position, learner in learners.items():

            states = self.env.model.sample_new_states(num_total_envs)

            # TODO: we need the same own valuation for multiple samples of opponents
            # -> should be done in env
            own_states = states[
                : num_own_envs
                * (self.num_alternative_actions ** self.num_rounds_to_play + 1),
                [player_position],
                0,
            ]
            own_states = own_states.repeat([1, num_opponent_envs])
            states[:, player_position, 0] = own_states.flatten()

            observations = self.env.model.get_observations(states)

            actual_rewards_total = torch.zeros(1, device=self.device)
            alternative_rewards_total = torch.zeros(
                self.num_own_envs,
                self.num_alternative_actions ** self.num_rounds_to_play,
                device=self.device,
            )

            for stage in range(self.num_rounds_to_play):

                actions = self.env.model.get_ma_learner_predictions(
                    learners, observations, True
                )

                # Replace our actions with alternative actions
                # Except the first one -> for actual play
                alternative_actions = self._generate_action_grid(stage=stage)
                player_actions = actions[player_position].view(
                    num_own_envs * num_opponent_envs, -1
                )
                player_actions[:, 1:] = alternative_actions.repeat(
                    [num_own_envs * num_opponent_envs, 1]
                )
                actions[player_position] = player_actions.view(
                    -1, self.env.model.action_size
                )

                observations, rewards, _, states = self.env.model.compute_step(
                    states, actions
                )

                # Collect rewards
                player_rewards = (
                    rewards[player_position]
                    .view(num_own_envs, num_opponent_envs, -1)
                    .mean(axis=1)
                )
                actual_rewards_total += player_rewards[:, 0].mean()
                alternative_rewards_total += player_rewards[:, 1:]

            # Compute approximate best response & its utility
            alternative_utility = alternative_rewards_total.max(axis=1).values.mean()
            utility_loss[player_position] = (
                (alternative_utility - actual_rewards_total).relu().item()
            )

        return utility_loss

    def _generate_action_grid(self, stage: int):
        """Generate a grid of alternative actions."""

        # TODO: support for higher action dimensions
        action_dim = 1
        dims = self.num_rounds_to_play * action_dim

        # create equidistant lines along the support in each dimension
        lines = [
            torch.linspace(
                self.action_range[0],
                self.action_range[1],
                self.num_alternative_actions,
                device=self.device,
            )
            for d in range(dims)
        ]
        mesh = torch.meshgrid(lines)
        grid = torch.stack(mesh, dim=-1).view(-1, dims)

        # TODO: only create this stage's actions in the first place
        return grid[:, stage]
