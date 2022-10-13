"""Verifier"""
import traceback
from typing import Dict

import torch
from gym import spaces
from tqdm import tqdm

from src.learners.base_learner import SABaseAlgorithm
from src.verifier.base_verifier import BaseVerifier

_CUDA_OOM_ERR_MSG_START = "CUDA out of memory. Tried to allocate"
ERR_MSG_OOM_SINGLE_BATCH = "Failed for good. Even a batch_size of 1 leads to OOM!"


class BFVerifier(BaseVerifier):
    """Verifier that tries out a grid of alternative actions and choses the
    best combination as approximation to a best response.
    """

    def __init__(self, env, num_envs: int = 1024, num_alternative_actions: int = 32):
        self.env = env
        self.num_agents = self.env.model.num_agents
        self.action_dim = env.model.ACTION_DIM

        self.device = self.env.device

        # Numeric actions
        if all(isinstance(s, spaces.Box) for s in env.model.action_spaces.values()):
            self.action_range = [
                env.model.ACTION_LOWER_BOUND,
                env.model.ACTION_UPPER_BOUND,
            ]
        else:
            raise ValueError("This verifier is for numeric/continuous actions only.")

        self.num_own_envs = num_envs
        self.num_opponent_envs = num_envs
        self.num_alternative_actions = num_alternative_actions
        self.num_rounds_to_play = self.env.model.num_rounds_to_play

    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        """Loop over the current player's valuation. This can be done to
        sequentialize some of the computation for reduced memory consumption.
        """
        utility_loss = torch.zeros(self.num_agents)

        # Lower batch size until tensor fits in (GPU) memory
        calculation_successful = False
        mini_batch_size = self.num_own_envs
        while not calculation_successful:
            try:
                for i in tqdm(range(int(self.num_own_envs / mini_batch_size))):
                    utility_loss += torch.tensor(
                        [
                            l / mini_batch_size
                            for l in self._verify(
                                learners, mini_batch_size, self.num_opponent_envs
                            )
                        ]
                    )
                calculation_successful = True
            except RuntimeError as e:
                if not str(e).startswith(_CUDA_OOM_ERR_MSG_START):
                    raise e
                if mini_batch_size <= 1:
                    traceback.print_exc()
                    raise RuntimeError(ERR_MSG_OOM_SINGLE_BATCH)
                mini_batch_size = int(mini_batch_size / 2)

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
        # TODO: Even when policy sharing is used, there are multiple learners!
        for player_position, learner in learners.items():

            states = self.env.model.sample_new_states(num_total_envs)

            # We need to reduce the number of own valuations to exactly
            # `num_own_envs`: Then, for each valuation and all combinations of
            # actions throughout all stages of the game are averaged over
            # `num_opponent_envs` opponents (and their corresponding valuations
            # and actions).
            own_states = states[
                :num_own_envs, [player_position], : self.env.model.valuation_size
            ].view(num_own_envs, 1, self.env.model.valuation_size)
            own_states = own_states.repeat(
                [
                    num_opponent_envs
                    * (self.num_alternative_actions ** self.num_rounds_to_play + 1),
                    1,
                    1,
                ]
            )
            states[:, [player_position], : self.env.model.valuation_size] = own_states

            # NOTE: Sample size from opponents could be reduced analogously to
            # `num_opponent_envs` to reduce variance

            observations = self.env.model.get_observations(states)

            actual_rewards_total = torch.zeros(1, device=self.device)
            alternative_rewards_total = torch.zeros(
                num_own_envs,
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
                # player_actions[:, 0] now correspond to the actual actions
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

            # NOTE: Currently we cannot get the BR's b/c we don't track actions
            # from previous stages

            # Compute utility of approximate best response
            alternative_utility = alternative_rewards_total.max(axis=1).values.mean()

            utility_loss[player_position] = (
                (alternative_utility - actual_rewards_total).relu().item()
            )

        return utility_loss

    def _generate_action_grid(self, stage: int):
        """Generate a grid of alternative actions."""

        # TODO: support for higher action dimensions
        if self.action_dim != 1:
            raise NotImplementedError()

        dims = self.num_rounds_to_play * self.action_dim

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
