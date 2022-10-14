"""Verifier"""
import traceback
from typing import Dict, List, Tuple

import torch
from gym import spaces
from tqdm import tqdm

import src.utils_folder.logging_utils as log_ut
from src.learners.base_learner import SABaseAlgorithm
from src.verifier.base_verifier import BaseVerifier

_CUDA_OOM_ERR_MSG_START = "CUDA out of memory. Tried to allocate"
ERR_MSG_OOM_SINGLE_BATCH = "Failed for good. Even a batch size of 1 leads to OOM!"


class BFVerifier(BaseVerifier):
    """Verifier that tries out a grid of alternative actions and choses the
    best combination as approximation to a best response.
    Assumptions:
    1. The number of stages is equal for all initial conditions
    2. We assume that the initial conditions are independent between the agents
    3. We only support single dimension actions at the moment
    """

    def __init__(self, env, mc_agent: int, mc_opps: int, action_discretization: int):
        self.env = env
        self.num_agents = self.env.model.num_agents
        self.mc_agent = mc_agent
        self.mc_opps = mc_opps
        self.action_discretization = action_discretization
        self.num_rounds_to_play = (
            self.env.model.num_rounds_to_play
        )  # TODO: Is this in every env?
        self.action_dim = env.model.ACTION_DIM

        self.device = self.env.device

        """if all(isinstance(s, spaces.Box) for s in env.model.action_spaces.values()):
            self.action_range = [
                env.model.ACTION_LOWER_BOUND,
                env.model.ACTION_UPPER_BOUND,
            ]
        else:
            raise ValueError("This verifier is for numeric/continuous actions only.")"""

    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        """Loop over the current player's valuation. This can be done to
        sequentialize some of the computation for reduced memory consumption.
        """
        utility_loss = torch.zeros(self.num_agents, device=self.device)

        for agent_id in range(self.num_agents):
            utility_loss[agent_id] = self._get_agent_utility_loss(learners, agent_id)
        return utility_loss.cpu().detach().tolist()

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

    def _get_agent_utility_loss(self, learners, agent_id: int) -> float:
        """
        Args:
            learners (_type_): holds agents' strategies
            agent_id (int): 

        Returns:
            float: estimated utility loss
            # TODO: could also return best response
        """
        first_stage_batch_sizes = self._get_first_stage_batch_sizes()
        eval_utilities = torch.zeros(len(first_stage_batch_sizes), device=self.device)
        for k, batch_size in enumerate(first_stage_batch_sizes):
            eval_utilities[k] = self._get_batch_utilities(
                learners, agent_id, batch_size
            )
        return self.weighted_average(eval_utilities, first_stage_batch_sizes)

    @staticmethod
    def weighted_average(values: torch.Tensor, weight_list: List[int]) -> float:
        # TODO: Put this into some utils file
        weight_tensor = torch.tensor(weight_list).to(values.device)
        weight_tensor = weight_tensor / torch.sum(weight_tensor)
        return torch.sum(values * weight_tensor)

    def _get_batch_utilities(self, learners, agent_id: int, batch_size: int) -> float:
        """Iterative Monte-Carlo approximation over an action-grid for the
        given agent. We expand the game tree and average over
        the opponent actions' outcomes. We simulate agent grid actions and
        query the strategies.

        Args:
            learners (_type_): holds agents' strategies
            agent_id (int):
            batch_size (int): 
        Returns:
            float: estimated utility loss
            # TODO: could also return best response
        """
        batch_utilities = torch.zeros(
            self._total_utilities_shape_for_batch(batch_size), device=self.device
        )

        agent_initial_states = self.env.model.sample_new_states(batch_size)
        agent_initial_obs = self.env.model.get_observations(agent_initial_states)[
            agent_id
        ]

        agent_grid_actions = self.env.get_action_grid(
            agent_id, grid_size=self.action_discretization
        )

        sim_initial_states = self.env.model.sample_new_states(self.mc_opps)
        sim_initial_obs = self.env.model.get_observations(sim_initial_states)

        states = {agent_id: None for agent_id in range(self.num_agents)}
        episode_starts = torch.ones((self.mc_opps,), dtype=bool, device=self.device)
        opp_actions = log_ut.get_eval_ma_actions(
            learners,
            sim_initial_obs,
            states,
            episode_starts,
            True,
            excluded_agents=[agent_id],
        )

        # ############### REPEAT TENSORS ############### #
        """
        agent_initial_obs: (batch_size, obs_size) -> (batch_size, action_discretization, mc_opp, obs_size)
        sim_initial_obs_agent: (mc_opp, obs_size) -> (batch_size, action_discretization, mc_opp, obs_size)
        agent_grid_actions: (action_discretization, action_size) -> (batch_size, action_discretization, mc_opp, obs_size)
        opp_actions: (mc_opp, action_size) -> (batch_size, action_discretization, mc_opp, obs_size)
        """
        # TODO: Check if torch.expand() instead of torch.repeat() is feasible!
        cur_batch_size = agent_initial_obs.shape[0]

        agent_initial_obs = self._repeat_tensor_along_new_axis(
            data=agent_initial_obs,
            pos=[1, 1],
            repeats=[self.action_discretization, self.mc_opps],
        )

        agent_initial_obs = agent_initial_obs.unsqueeze(1).unsqueeze(1)
        agent_initial_obs = agent_initial_obs.repeat(
            1, self.action_discretization, self.mc_opps, 1
        )
        flatten_agent_initial_obs = torch.flatten(agent_initial_obs, end_dim=2)

        """Repeat and play:
        1. repeat the agent_obs (batch_size, obs_size) along oppo_obs num (mc_opps, ) 
        to get agent_obs of shape (batch_size * mc_opps, obs_size) and repeat oppo_obs by batch_size
        2. get states from new observation dict
        3. repeat with actions as well
        4. make step
        5. reshape rewards (batch_size * mc_opps,...) to (batch_size, mc_opps)
        6. add and broadcast reshaped rewards to batch_utilities
        7. repeat for next stages
        """

    @staticmethod
    def _repeat_tensor_along_new_axis(
        data: torch.Tensor, pos: List[int], repeats: List[int]
    ) -> torch.Tensor:
        """Add additional dimensions as pos and repeat it for repeats along these dimensions.

        Args:
            data (torch.Tensor): tensor to be repeated
            pos (List[int]): non-decreasing order of positions where dimensions should be added
            repeats (List[int]): number of repeats of dimensions

        Returns:
            torch.Tensor: repeated tensor
        """
        assert len(pos) == len(repeats), "Each pos needs a specified repeat!"
        initial_shape = data.shape
        for single_pos in pos:
            data = data.unsqueeze(single_pos)

        data.repeat()  # TODO: make tuple, insert repeats between 1's

    def _total_utilities_shape_for_batch(self, batch_size: int) -> Tuple[int]:
        """Gives the shape of total utilities to be estimated in a single batch rollout.
        Args:
            batch_size (int):
        Returns:
            Tuple[int]: Shape of utility tensor
        """
        return (batch_size,) + tuple(
            [self.action_discretization, self.mc_opps] * self.num_rounds_to_play
        )

    def _get_first_stage_batch_sizes(self) -> List[int]:
        """We check how to distribute the initial draws of
        environments so that they fit on the GPU.

        Returns:
            List[int]: How many envs to create in first stage
        """
        # TODO:
        return [2 for _ in range(int(self.mc_agent / 2))]

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

            actual_rewards_total = torch.zeros(1, device=self.device)
            alternative_rewards_total = torch.zeros(
                num_own_envs,
                self.num_alternative_actions ** self.num_rounds_to_play,
                device=self.device,
            )

            for stage in range(self.num_rounds_to_play):

                observations = self.env.model.get_observations(states)
                actions = self.env.model.get_ma_learner_predictions(
                    learners, observations, True
                )

                # Replace our actions with alternative actions
                # Except the first one -> for actual play
                alternative_actions = self._generate_action_grid(player_position, stage)
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

                _, rewards, _, states = self.env.model.compute_step(states, actions)

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

    def _generate_action_grid(self, player_position: int, stage: int):
        """Generate a grid of alternative actions."""

        # TODO: support for higher action dimensions
        if self.action_dim != 1:
            raise NotImplementedError()

        dims = self.num_rounds_to_play * self.action_dim

        # create equidistant lines along the support in each dimension
        lines = [
            self.env.get_action_grid(
                player_position, self.num_alternative_actions
            ).view(-1)
            for d in range(dims)
        ]
        mesh = torch.meshgrid(lines)
        grid = torch.stack(mesh, dim=-1).view(-1, dims)

        # TODO: only create this stage's actions in the first place
        return grid[:, stage]
