"""Verifier"""
import traceback
from typing import Dict, List, Tuple

import torch
from gym import spaces
from tqdm import tqdm

import src.utils_folder.logging_utils as log_ut
from src.learners.base_learner import SABaseAlgorithm
from src.verifier.base_verifier import BaseVerifier
from src.verifier.information_set_tree import InformationSetTree

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

    def __init__(
        self,
        env,
        num_simulations: int,
        obs_discretization: int,
        action_discretization: int,
    ):
        self.env = env
        self.num_agents = self.env.model.num_agents
        self.num_simulations = num_simulations
        self.action_discretization = action_discretization
        self.obs_discretization = obs_discretization
        self.num_rounds_to_play = (
            self.env.model.num_rounds_to_play
        )  # TODO: Is this in every (feasible) env?
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
        information_tree = InformationSetTree(
            agent_id,
            self.env,
            self.obs_discretization,
            self.action_discretization,
            self.device,
        )
        for batch_size in self._get_first_stage_batch_sizes():
            self._add_simulation_results_to_tree(
                learners, agent_id, batch_size, information_tree
            )
        return information_tree.get_br_utility_estimate()

    def _get_first_stage_batch_sizes(self) -> List[int]:
        """We check how to distribute the initial draws of
        environments so that they fit on the GPU.

        Returns:
            List[int]: How many envs to create in first stage
        """
        # TODO:
        return [2 for _ in range(int(self.num_simulations / 2))]

    @staticmethod
    def weighted_average(values: torch.Tensor, weight_list: List[int]) -> float:
        # TODO: Put this into some utils file or delete
        weight_tensor = torch.tensor(weight_list).to(values.device)
        weight_tensor = weight_tensor / torch.sum(weight_tensor)
        return torch.sum(values * weight_tensor)

    def _add_simulation_results_to_tree(
        self,
        learners,
        agent_id: int,
        batch_size: int,
        information_tree: InformationSetTree,
    ):
        """Monte-Carlo approximation for a fixed discretized strategy for the
        given agent. We simulate batch_size games for this given strategy.

        Args:
            learners (_type_): holds agents' strategies
            agent_id (int):
            batch_size (int): 
            information_tree (InformationSetTree): game tree
        """

        batch_utilities = torch.zeros(
            self._total_utilities_shape_for_batch(batch_size), device=self.device
        ).flatten()
        cur_sim_size = batch_size

        states = self.env.model.sample_new_states(batch_size)
        agent_obs, opp_obs = self._split_obs(
            agent_id, self.env.model.get_observations(states)
        )
        episode_starts = torch.ones((batch_size,), dtype=bool, device=self.device)
        batch_indices = torch.tensor([], device=self.device).long()

        for stage in range(self.num_rounds_to_play):
            # TODO: Write a method with corresponding test for the repeation here. It looks correct atm.
            obs_bin_indices = self.env.model.get_obs_bin_indices(
                agent_obs, agent_id, stage, self.obs_discretization
            )
            repeated_obs_bin_indices = self._repeat_and_flatten_to_full_sim_size(
                obs_bin_indices, stage
            ).unsqueeze(-1)
            action_bins = torch.arange(self.action_discretization, device=self.device)
            repeated_action_bins = (
                self._repeat_tensor_along_new_axis(
                    data=action_bins,
                    pos=[0, 2],
                    repeats=[
                        cur_sim_size,
                        self.action_discretization
                        ** (self.num_rounds_to_play - stage - 1),
                    ],
                )
                .flatten()
                .unsqueeze(-1)
            )
            batch_indices = torch.cat(
                (batch_indices, repeated_obs_bin_indices, repeated_action_bins), dim=-1
            )
            repeated_grid_actions = self._get_agent_grid_actions(agent_id, cur_sim_size)

            cur_sim_size *= self.action_discretization

            opp_actions = self._get_opponent_actions(
                episode_starts, learners, agent_id, opp_obs
            )
            for opp_agent, opp_action in opp_actions.items():
                opp_actions[opp_agent] = self._repeat_tensor_along_new_axis(
                    data=opp_action, pos=[1], repeats=[self.action_discretization]
                )

            repeated_states = self._repeat_tensor_along_new_axis(
                data=states, pos=[1], repeats=[self.action_discretization]
            )

            combined_actions = {}
            for agent_identifier in range(self.num_agents):
                if agent_identifier == agent_id:
                    combined_actions[agent_identifier] = repeated_grid_actions.flatten(
                        end_dim=1
                    )
                else:
                    combined_actions[agent_identifier] = opp_actions[
                        agent_identifier
                    ].flatten(end_dim=1)
            flattened_states = repeated_states.flatten(end_dim=1)

            new_obs, rewards, dones, new_states = self.env.model.compute_step(
                flattened_states, combined_actions
            )
            repeated_agent_rewards = self._repeat_and_flatten_to_full_sim_size(
                rewards[agent_id], stage + 1
            )
            batch_utilities += repeated_agent_rewards
            episode_starts = dones

            agent_obs, opp_obs = self._split_obs(agent_id, new_obs)
            states = new_states

        assert (
            torch.all(dones).cpu().item()
        ), "The game should have ended after playing all rounds! Check num_rounds_to_play of env!"
        information_tree.add_simulation_results(batch_utilities, batch_indices)

    def _repeat_and_flatten_to_full_sim_size(
        self, data: torch.Tensor, stage: int
    ) -> torch.Tensor:
        pos_to_repeat = len(data.shape)
        return self._repeat_tensor_along_new_axis(
            data=data,
            pos=[pos_to_repeat],
            repeats=[self.action_discretization ** (self.num_rounds_to_play - stage)],
        ).flatten()

    def _get_agent_grid_actions(self, agent_id, cur_sim_size):
        agent_grid_actions = self.env.get_action_grid(
            agent_id, grid_size=self.action_discretization
        )
        repeated_grid_actions = self._repeat_tensor_along_new_axis(
            data=agent_grid_actions, pos=[0], repeats=[cur_sim_size]
        )

        return repeated_grid_actions

    def _split_obs(self, agent_id: int, obs: Dict):
        agent_obs = obs.pop(agent_id)
        opp_obs = obs
        return agent_obs, opp_obs

    def _get_opponent_actions(
        self, episode_starts: torch.Tensor, learners, agent_id, opp_obs
    ):
        states = {agent_id: None for agent_id in range(self.num_agents)}
        opp_actions = log_ut.get_eval_ma_actions(
            learners, opp_obs, states, episode_starts, True, excluded_agents=[agent_id]
        )

        return opp_actions

    @staticmethod
    def _repeat_tensor_along_new_axis(
        data: torch.Tensor, pos: List[int], repeats: List[int]
    ) -> torch.Tensor:
        """Add additional dimensions as pos and repeat it for repeats along these dimensions.

        Args:
            data (torch.Tensor): tensor to be repeated
            pos (List[int]): strictly increasing order of positions where repeats should be in out-tensor
            repeats (List[int]): number of repeats of dimensions

        Returns:
            torch.Tensor: repeated tensor
        Example:
        data.shape = (2, 3)
        pos = [1, 2]
        repeats = [11, 7]
        out.shape = (2, 11, 7, 3)
        """
        # TODO: Check if torch.expand() instead of torch.repeat() is feasible!
        assert len(pos) == len(repeats), "Each pos needs a specified repeat!"
        for single_pos in pos:
            data = data.unsqueeze(single_pos)
        dims_to_be_repeated = [1 for i in range(len(data.shape))]
        for k, repeat in enumerate(repeats):
            dims_to_be_repeated[pos[k]] = repeat
        return data.repeat(tuple(dims_to_be_repeated))

    def _total_utilities_shape_for_batch(self, batch_size: int) -> Tuple[int]:
        """Gives the shape of total utilities to be estimated in a single batch rollout.
        Args:
            batch_size (int):
        Returns:
            Tuple[int]: Shape of utility tensor
        """
        return (batch_size,) + tuple(
            [self.action_discretization] * self.num_rounds_to_play
        )

    def _get_batch_sizes_for_sim(self) -> List[int]:
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
