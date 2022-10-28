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
        # TODO: Lower the batch sizes if needed. Maybe change outer loop instead of filling in the correct sizes
        return [4 for _ in range(int(self.num_simulations / 4))]

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
