import itertools
import random
from typing import Any, Dict, List, Tuple

import gizeh
import moviepy.editor as mpy
import numpy as np
import torch
from gym import spaces
from gym.spaces import Space
from tensordict import TensorDict

import src.utils.evaluation_utils as ev_ut
import src.utils.logging_write_utils as log_ut
import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import BaseEnvForVec

"""Treat 2d-grid as (x, y) coordinates,
where x can be imagined as horizontal movement and
y as vertical movement.
Grid length is maximal x
Grid width is maximal y
Lower-left corner is (0, 0)
move RIGHT --> (1, 0)
move UP --> (1, 1)
move LEFT --> (0, 1)
move DOWN --> (0, 0)"""
UP = (0, 1)
DOWN = (0, -1)
LEFT = (-1, 0)
RIGHT = (1, 0)
NOOP = (0, 0)


class CoinGame(BaseEnvForVec):
    """Multi-Agent Coin game. Standard setting is a two person social dilemma. Each agent has special coins for itself.
    However, an agent gets a penalty if an opponent agent picks up one of its coins.
    This is an extension to multiple agents including several extensions to study dependency structures."""

    ACTION_DIM = 1

    def __init__(self, config: Dict, device: str = None):
        self.num_actions = 5
        self.grid_length = config.grid_length
        self.grid_width = config.grid_width
        self.max_episode_length = config.max_episode_length
        self.num_max_coins_per_agent = config.num_max_coins_per_agent
        self.action_index_tensor = self._init_action_index_tensor(device)
        self.display_scale = 20.0
        super().__init__(config, device)

        self.agent_color_dict = self._init_agent_color_dict()
        self.reward_structure = self._init_reward_structure()
        self.penalty_structure = self._init_penalty_structure()
        assert self.num_max_coins_per_agent >= 1, "The agents need at least one coin!"

    def _get_num_agents(self) -> int:
        return self.config["num_agents"]

    def _init_observation_spaces(self) -> Dict[int, Space]:
        return {
            agent_id: self._get_observation_space(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _init_action_index_tensor(self, device) -> torch.Tensor:
        """Creates a lookup tensor for the actions.
        The mapping is:
            0: UP
            1: DOWN
            2: LEFT
            3: RIGHT
            4: NOOP
        Returns:
            torch.Tensor: _description_
        """
        action_index = [UP, DOWN, LEFT, RIGHT, NOOP]
        action_index = [list(a) for a in action_index]
        return torch.concat(
            [
                torch.tensor(
                    action_index, device=device, dtype=torch.float, requires_grad=False
                )
            ]
        )

    def _get_observation_space(self, agent_id: int) -> spaces.Space:
        return spaces.Box(
            low=-1.0,
            high=np.inf,
            shape=(self.num_agents, self._get_agent_state_info_size()),
        )

    def _init_action_spaces(self) -> Dict[int, Space]:
        return {
            agent_id: spaces.Discrete(self.num_actions)
            for agent_id in range(self.num_agents)
        }

    def _init_action_space_sizes(self) -> Dict[int, int]:
        return {agent_id: self.num_actions for agent_id in range(self.num_agents)}

    def to(self, device) -> Any:
        self.device = device
        return self

    def _init_agent_color_dict(self) -> Dict[int, Tuple[int]]:
        colors = [
            (0 / 255.0, 150 / 255.0, 196 / 255.0),
            (248 / 255.0, 118 / 255.0, 109 / 255.0),
            (150 / 255.0, 120 / 255.0, 170 / 255.0),
            (255 / 255.0, 215 / 255.0, 130 / 255.0),
        ]
        color_dict = {}
        for agent_id in range(self.num_agents):
            if agent_id > 3:
                color_dict[agent_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            else:
                color_dict[agent_id] = colors[agent_id]
        return color_dict

    def _init_reward_structure(self) -> torch.Tensor:
        """Decides how much reward is given to each agent for
        collecting a coin of a specific agent.
        Returns:
            torch.Tensor: shape=(num_agents, num_agents)
                    entry (i, j): i receives this reward for collecting coin of j
        """
        if self.config.reward_structure == "unitary":
            reward_structure = torch.ones(
                (self.num_agents, self.num_agents), device=self.device
            )
        else:
            raise ValueError(
                "No valid reward_structure selected! Check "
                + str(self.config.reward_structure)
            )
        reward_structure.requires_grad = False
        return reward_structure

    def _init_penalty_structure(self) -> torch.Tensor:
        """Decides how much penalty is given to each agent for
        collecting a coin of a specific agent.
        Returns:
            torch.Tensor: shape=(num_agents, num_agents)
                    entry (i, j): i receives this penalty for collecting coin of j
        """
        if self.config.penalty_structure == "no_penalty":
            penalty_structure = torch.zeros(
                (self.num_agents, self.num_agents), device=self.device
            )
        elif self.config.penalty_structure == "all_others":
            penalty_amount = 2.0
            diag_no_penalty = torch.diag(
                torch.ones(self.num_agents, device=self.device)
            )
            penalty_structure = torch.ones(
                (self.num_agents, self.num_agents), device=self.device
            )
            penalty_structure -= diag_no_penalty
            penalty_structure *= penalty_amount
        else:
            raise ValueError(
                "No valid penalty_structure selected! Check "
                + str(self.config.penalty_structure)
            )
        penalty_structure.requires_grad = False
        return penalty_structure

    def _get_state_info_sizes_dict(self) -> Dict[str, int]:
        """Info contains:
            (x, y) coordinate of agent
            (grid_length, grid_width)
            (current_step, max_episode_length)
            (coin_exists, ) + (x_i, y_i) coordinates of coins for 1 <= i <= num_max_coins_per_agent"""
        return {
            "agent_xy": 2,
            "grid_lw": 2,
            "progress_info": 2,
            "coins_xy": 2 * self.num_max_coins_per_agent,
        }

    def _get_agent_state_info_size(self) -> int:
        return sum(self._get_state_info_sizes_dict().values())

    def _get_grid_lw_tensor(self, n: int) -> torch.Tensor:
        grid_lw = torch.tensor(
            [self.grid_length, self.grid_width], dtype=torch.float, device=self.device
        )
        return grid_lw.unsqueeze(0).unsqueeze(0).repeat(n, self.num_agents, 1)

    def _sample_random_xy(self, num_states: int, num_agents: int) -> torch.Tensor:
        x_coord = torch.randint(
            low=0,
            high=self.grid_length,
            size=(num_states, num_agents),
            dtype=torch.float,
            device=self.device,
        )
        y_coord = torch.randint(
            low=0,
            high=self.grid_width,
            size=(num_states, num_agents),
            dtype=torch.float,
            device=self.device,
        )
        return torch.stack((x_coord, y_coord), dim=-1)

    def _sample_random_disjoint_xy(
        self, num_states: int, num_agents: int
    ) -> torch.Tensor:
        disjoint_x = self._sample_disjoint_along_1d(
            self.grid_length, num_states, num_agents
        )
        disjoint_y = self._sample_disjoint_along_1d(
            self.grid_width, num_states, num_agents
        )
        return torch.stack((disjoint_x, disjoint_y), dim=-1)

    def _sample_disjoint_along_1d(
        self, max_value: int, num_states: int, num_agents: int
    ) -> torch.Tensor:
        permutations = self._get_random_permutations(max_value, num_agents)
        idx_sample = torch.randint(
            low=0,
            high=permutations.shape[0],
            size=(num_states,),
            device=self.device,
            dtype=torch.long,
        )
        return torch.index_select(permutations, 0, idx_sample)

    def _get_random_permutations(
        self, max_value: int, length_perm: int
    ) -> torch.Tensor:
        assert (
            length_perm <= max_value
        ), "Can not draw a longer permutation than number of elements to permute!"
        if length_perm > 2 and max_value > 6:
            permutations = torch.stack(
                [
                    torch.randperm(max_value, device=self.device)[:length_perm]
                    for _ in range(100)
                ]
            )
        else:
            permutations = [
                list(perm)[:length_perm]
                for perm in itertools.permutations(list(range(max_value)))
            ]
        permutations = torch.tensor(permutations, device=self.device, dtype=torch.float)
        return permutations

    def _get_disjoint_xy(self, num_states: int, num_agents: int) -> torch.Tensor:
        return self._sample_random_disjoint_xy(num_states, num_agents)

    def _get_progress_info_tensor(self, n: int) -> torch.Tensor:
        progress_info = torch.tensor([0.0, self.max_episode_length], device=self.device)
        return progress_info.unsqueeze(0).unsqueeze(0).repeat(n, self.num_agents, 1)

    def sample_new_states(self, n: int) -> Any:
        """Create new initial states.

        :param n: Batch size of how many games are played in parallel.        
        :return: the new states, in shape=(n, num_agents, size_info).
        """
        states = TensorDict(
            {
                "grid_lw": self._get_grid_lw_tensor(n),
                "agents_xy": self._get_disjoint_xy(n, self.num_agents),
                "coins_xy_0": self._get_disjoint_xy(n, self.num_agents),
                "progress_info": self._get_progress_info_tensor(n),
            },
            batch_size=n,
        )
        for add_coins_index in range(1, self.num_max_coins_per_agent):
            states["coins_xy_" + str(add_coins_index)] = -1.0 * torch.ones(
                (n, self.num_agents, 2), device=self.device
            )
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
        new_states["progress_info"][:, :, 0] += 1.0  # Increment round
        new_states = self.move_agents(new_states, actions)
        new_states, collecting_info = self.collect_coins(new_states)
        new_states = self.coin_spawning(new_states)

        rewards = self._compute_rewards(collecting_info)

        dones = new_states["progress_info"][:, 0, 0] >= self.config.max_episode_length

        observations = self.get_observations(new_states)

        return observations, rewards, dones.clone(), new_states

    def _compute_rewards(
        self, collecting_info: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the rewards depending on which coins were collected.
        """
        reward_dict = {}
        for agent_id in range(self.num_agents):
            agent_num_collected_coins_per_agent = collecting_info[agent_id].sum(dim=2)
            sa_rewards = agent_num_collected_coins_per_agent * self.reward_structure[
                agent_id, :
            ].unsqueeze(0)
            sa_penalties = agent_num_collected_coins_per_agent * self.penalty_structure[
                agent_id, :
            ].unsqueeze(0)
            reward_dict[agent_id] = (sa_rewards - sa_penalties).sum(dim=1)

        return reward_dict

    def get_observations(self, states: TensorDict) -> Dict[int, torch.Tensor]:
        """Return the observations of the players.

        :param states: The current states of shape (num_env, num_agents,
            state_dim).
        :returns observations: Dict of Observations of shape (num_env, num_agents, state_dim).
        """
        obs_dict = {}
        observations = torch.cat(list(states.values()), dim=2)
        for agent_id in range(self.num_agents):
            rolling_index = 0
            if self.config.make_obs_invariant_to_agent_order:
                rolling_index = self.num_agents - agent_id
            obs_dict[agent_id] = (
                torch.roll(observations, rolling_index, dims=1).detach().clone()
            )
        return obs_dict

    def move_agents(
        self, states: torch.Tensor, actions: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        agent_move_order = self._get_agent_move_order(list(actions.keys()))
        for agent_id in agent_move_order:
            states["agents_xy"][:, agent_id, :] = self._get_updated_agent_xy(
                states, actions[agent_id], agent_id
            )
        return states

    def _get_agent_move_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_index_tensor.index_select(0, action)

    def _get_updated_agent_xy(
        self, states: torch.Tensor, agent_actions: torch.Tensor, agent_id: int
    ) -> torch.Tensor:
        agent_xy = states["agents_xy"][:, agent_id, :]
        new_agent_xy = agent_xy + self._get_agent_move_tensor(agent_actions)
        new_agent_xy = self._update_xy_by_grid_boundary_rule(new_agent_xy)

        opponent_xys = states["agents_xy"][:, self._get_opponent_dims([agent_id]), :]
        new_agent_xy_allowed = self._check_overlap_by_stacking_rule(
            xy_to_check=new_agent_xy,
            comp_xys=opponent_xys,
            allow_stacking=self.config.agents_can_stack,
        )
        # reset position for occupied positions
        new_agent_xy[new_agent_xy_allowed] = agent_xy[new_agent_xy_allowed]

        return new_agent_xy

    def _get_opponent_dims(self, excluded_ids: List[int]) -> Tuple[int]:
        return tuple(
            [
                agent_id
                for agent_id in range(self.num_agents)
                if agent_id not in excluded_ids
            ]
        )

    def _check_overlap_by_stacking_rule(
        self,
        xy_to_check: torch.Tensor,
        comp_xys: torch.Tensor,
        allow_stacking: bool = False,
        collapse_to_single_dim: bool = True,
    ) -> torch.Tensor:
        """If we do not allow stacking, we check whether the xy positions overlaps with any of comp_xys

        Args:
            xy_to_check (torch.Tensor): xy position to check
            comp_xys (torch.Tensor): all occupied positions
            allow_stacking (bool): whether we care about stacking or not

        Returns:
            torch.Tensor: is the new position allowed by the stacking rule
        """
        xy_overlaps = torch.ones(
            xy_to_check.shape[0], dtype=torch.bool, device=xy_to_check.device
        )
        if not allow_stacking:
            x_to_check, y_to_check = torch.split(xy_to_check, 1, dim=1)
            comp_xs, comp_ys = torch.split(comp_xys, 1, dim=2)
            comp_xs, comp_ys = comp_xs.squeeze(-1), comp_ys.squeeze(-1)
            xy_overlaps = torch.logical_and(
                comp_xs == x_to_check, comp_ys == y_to_check
            )
            if collapse_to_single_dim:
                xy_overlaps = torch.any(xy_overlaps, dim=1)
        return xy_overlaps

    def _update_xy_by_grid_boundary_rule(self, new_agent_xy):
        new_x, new_y = torch.split(new_agent_xy, 1, dim=1)
        if self.config.boundary_treatment_type == "loop":
            # agent re-enters on the other field-side if moving out of grid
            new_x = torch.remainder(new_x, self.grid_length)
            new_y = torch.remainder(new_y, self.grid_width)
        elif self.config.boundary_treatment_type == "closed":
            # agent cannot move outside of grid
            new_x = torch.clamp(new_x, min=0, max=self.grid_length - 1)
            new_y = torch.clamp(new_y, min=0, max=self.grid_width - 1)
        else:
            raise ValueError(
                "No valid boundary_treatment_type selected, check: "
                + str(self.config.boundary_treatment_type)
            )

        return torch.cat([new_x, new_y], dim=-1)

    def _get_agent_move_order(self, agent_ids: List[int]) -> List[int]:
        """Which agent moves first can have a big effect depending on other properties.
        For example, agents can (by default) not occupy the same field. The move order gives
        earlier agents an advantage.

        Args:
            agent_ids (List[int]): Agents to move

        Returns:
            List[int]: Order to move by
        """
        move_type = self.config.move_order_type
        move_order = None
        if move_type == "static":
            move_order = agent_ids
        elif move_type == "random":
            random.shuffle(agent_ids)
            move_order = agent_ids
        else:
            raise ValueError(
                "No valid move_order_type selected, check: " + str(move_type)
            )
        return move_order

    def collect_coins(
        self, states: TensorDict
    ) -> Tuple[TensorDict, Dict[int, torch.Tensor]]:
        collection_info = self._get_collection_info(states)
        self._reset_xy_for_collected_coins(states, collection_info)
        return states, collection_info

    def _reset_xy_for_collected_coins(
        self, states: TensorDict, collection_info: Dict[int, torch.Tensor]
    ):
        coins_collected = torch.any(
            torch.stack(list(collection_info.values()), dim=-1), dim=-1
        )
        coins_xy_to_reset = coins_collected.unsqueeze(-1).expand(-1, -1, -1, 2)
        for coin_index in range(self.num_max_coins_per_agent):
            states["coins_xy_" + str(coin_index)][
                coins_xy_to_reset[:, :, coin_index, :]
            ] = -1.0

    def _get_collection_info(self, states: TensorDict) -> Dict[int, torch.Tensor]:
        """Returns a dict[agent_id, coins_collected], where
        coins_collected is a boolean tensor with shape=[batch_size, num_agents, num_coins]

        Args:
            states (TensorDict): _description_

        Returns:
            Dict[int, torch.Tensor]: _description_
        """
        collection_info = {}
        for agent_id in range(self.num_agents):
            coins_xy_to_collect = self._get_relevant_coins_to_collect(states, agent_id)
            agent_coins_collected = self._check_overlap_by_stacking_rule(
                states["agents_xy"][:, agent_id, :],
                coins_xy_to_collect,
                allow_stacking=False,
                collapse_to_single_dim=False,
            )
            collection_info[agent_id] = agent_coins_collected.reshape(
                states.batch_size + (self.num_agents, self.num_max_coins_per_agent)
            )

        return collection_info

    def _get_relevant_coins_to_collect(
        self, states: TensorDict, agent_id: int
    ) -> torch.Tensor:
        """_summary_

        Args:
            states (TensorDict): _description_
            agent_id (int): _description_

        Returns:
            torch.Tensor: Shape=(batch_size, num_agents * num_coins, 2)
        """
        coins_xy_to_collect = torch.stack(
            [
                states["coins_xy_" + str(coin_index)]
                for coin_index in range(self.num_max_coins_per_agent)
            ],
            dim=2,
        )  # Stack and reshape later, so that entries are at correct position.
        if self.config.only_pick_up_own_coin:
            agents_to_exclude = tuple(
                [
                    agent_idx
                    for agent_idx in range(self.num_agents)
                    if agent_idx != agent_id
                ]
            )
            coins_xy_to_collect[:, agents_to_exclude, :, :] = -2.0
        return coins_xy_to_collect.reshape(
            states.batch_size + (self.num_agents * self.num_max_coins_per_agent, 2)
        )

    def _coins_exist(self, coins_xy: TensorDict) -> torch.Tensor:
        return ~torch.all((coins_xy == -1.0), dim=2)

    def coin_spawning(self, states: TensorDict) -> TensorDict:
        prob_boundary = self.config["spawn_rate"] / self.num_max_coins_per_agent
        for coin_index in range(self.num_max_coins_per_agent):
            coins_to_spawn = (
                torch.rand(states.batch_size + (self.num_agents,), device=self.device)
                > 1 - prob_boundary
            )
            coins_exist = self._coins_exist(states["coins_xy_" + str(coin_index)])
            coins_to_spawn = torch.logical_and(coins_to_spawn, ~coins_exist)

            new_coins_xy = self._sample_random_xy(states.batch_size[0], self.num_agents)

            for agent_id in range(self.num_agents):
                new_agent_coin_xy_allowed = ~self._check_overlap_by_stacking_rule(
                    new_coins_xy[:, agent_id, :],
                    self._get_all_other_coins_xy(
                        states, excluded_coin=coin_index, excluded_agent=agent_id
                    ),
                    allow_stacking=self.config["coins_can_stack"],
                )
                newly_spawned_coins_bool = torch.logical_and(
                    new_agent_coin_xy_allowed, coins_to_spawn[:, agent_id]
                )
                states["coins_xy_" + str(coin_index)][:, agent_id, :][
                    newly_spawned_coins_bool
                ] = new_coins_xy[:, agent_id, :][newly_spawned_coins_bool]
        return states

    def _get_all_other_coins_xy(
        self, states: TensorDict, excluded_coin: int, excluded_agent: int
    ) -> torch.Tensor:
        other_coins_xy = torch.cat(
            [
                states["coins_xy_" + str(coin_index)]
                for coin_index in range(self.num_max_coins_per_agent)
            ],
            dim=1,
        )
        index_to_drop = excluded_coin * self.num_agents + excluded_agent
        indices_to_keep = tuple(
            [
                coin_agent_index
                for coin_agent_index in range(
                    self.num_agents * self.num_max_coins_per_agent
                )
                if coin_agent_index != index_to_drop
            ]
        )
        return other_coins_xy[:, indices_to_keep, :]

    def custom_evaluation(
        self,
        learners,
        env,
        writer=None,
        iteration: int = 0,
        config: Dict = None,
        num_samples=2 ** 16,
    ):
        self._store_eval_videos(learners, env, iteration, config, num_videos=3)

        states_list, observations_list, actions_list, rewards_list = ev_ut.run_algorithms(
            self, learners, num_samples, self.max_episode_length, deterministic=True
        )
        self._eval_collected_coins_by_agents(learners, states_list, actions_list)

    def _eval_collected_coins_by_agents(
        self,
        learners,
        states_list: List[TensorDict],
        actions_list: List[Dict[int, torch.Tensor]],
    ):
        rollout_collected_coins_dict = self._get_number_of_coins_collected_in_rollout(
            states_list, actions_list
        )
        own_coin_collection_dict, remaining_coins_collection_dict = self._extract_coins_collected_to_log(
            rollout_collected_coins_dict
        )
        # Store data
        log_ut.log_data_dict_to_learner_loggers(
            learners, own_coin_collection_dict, f"eval/own_coins_collected"
        )
        for (
            target_agent_id,
            targets_collected_by_source,
        ) in remaining_coins_collection_dict.items():
            log_ut.log_data_dict_to_learner_loggers(
                learners,
                targets_collected_by_source,
                f"eval/coins_collected_of_agent_{target_agent_id}",
            )

    def _extract_coins_collected_to_log(
        self, rollout_collected_coins_dict: Dict[int, torch.Tensor]
    ):
        own_coin_collection_dict = {}
        remaining_coins_collection_dict = {
            agent_id: {} for agent_id in range(self.num_agents)
        }
        for source_agent in range(self.num_agents):
            agent_collected_coins = (
                rollout_collected_coins_dict[source_agent].float().mean(axis=0)
            )
            for target_agent in range(self.num_agents):
                coins_collected_of_target_by_source = (
                    agent_collected_coins[target_agent].cpu().item()
                )
                if source_agent == target_agent:
                    own_coin_collection_dict[
                        target_agent
                    ] = coins_collected_of_target_by_source
                else:
                    remaining_coins_collection_dict[target_agent][
                        source_agent
                    ] = coins_collected_of_target_by_source
        return own_coin_collection_dict, remaining_coins_collection_dict

    def _get_number_of_coins_collected_in_rollout(
        self, states_list: List[TensorDict], actions_list: List[Dict[int, torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        rollout_collected_coins_dict = {
            agent_id: torch.zeros(
                states_list[0].batch_size + (self.num_agents,),
                device=self.device,
                dtype=torch.long,
            )
            for agent_id in range(self.num_agents)
        }
        for stage, actions in enumerate(actions_list):
            updated_states = self.move_agents(states_list[stage], actions)
            collection_info = self._get_collection_info(updated_states)
            for agent_id in range(self.num_agents):
                rollout_collected_coins_dict[agent_id] += collection_info[agent_id].sum(
                    axis=2
                )
        return rollout_collected_coins_dict

    def _store_eval_videos(self, learners, env, iteration, config, num_videos):
        filepath = config["experiment_log_path"]  # TODO: handle ind dir
        for video_iter in range(num_videos):
            clip = self._get_single_rendered_video(learners, env)
            filename = (
                "iteration_" + str(iteration) + "_clip_num_" + str(video_iter) + ".mp4"
            )
            clip.write_videofile(filepath + filename)

    def _get_single_rendered_video(self, learners, env):
        images = []
        states = self.sample_new_states(1)
        images.append(env.model.render(states[0]))

        dones = torch.zeros((1,), dtype=bool)

        while not dones.detach().item():
            observations = self.get_observations(states)
            actions = th_ut.get_ma_actions(learners, observations, deterministic=True)
            translated_actions = env.translate_joint_actions(actions)
            observations, rewards, dones, states = env.model.compute_step(
                states, translated_actions
            )
            episode_starts = dones
            images.append(env.model.render(states[0]))

        clip = mpy.ImageSequenceClip(images, fps=5)
        return clip

    def render(self, state):
        surface = gizeh.Surface(
            width=self.grid_to_pixel_1d(self.grid_length),
            height=self.grid_to_pixel_1d(self.grid_width),
            bg_color=(1.0, 1.0, 1.0),
        )
        self._draw_grid(surface)
        self._draw_coins(state, surface)
        self._draw_agents(state, surface)

        return surface.get_npimage(transparent=False, y_origin="bottom")

    def _draw_grid(self, surface):
        line_thickness = 2
        x_mid_coord = self.grid_length / 2
        y_mid_coord = self.grid_width / 2
        for x_grid_index in range(self.grid_length):
            line_length = self.grid_width * self.display_scale
            vertical_line = gizeh.rectangle(
                lx=line_thickness,
                ly=line_length,
                xy=self.grid_xy_to_pixel([x_grid_index, y_mid_coord]),
                fill=(0.0, 0.0, 0.0),
            )
            vertical_line.draw(surface)
        for y_grid_index in range(self.grid_width):
            line_length = self.grid_length * self.display_scale
            vertical_line = gizeh.rectangle(
                lx=line_length,
                ly=line_thickness,
                xy=self.grid_xy_to_pixel([x_mid_coord, y_grid_index]),
                fill=(0.0, 0.0, 0.0),
            )
            vertical_line.draw(surface)

    def _draw_agents(self, state, surface):
        for agent_id in range(self.num_agents):
            agent_xy = state["agents_xy"][agent_id, :]
            agent_x, agent_y = agent_xy[0].cpu().item(), agent_xy[1].cpu().item()
            agent = gizeh.square(
                l=self.display_scale,
                xy=self.grid_xy_to_pixel(
                    [int(agent_x), int(agent_y)], shift_center=True
                ),
                fill=self.agent_color_dict[agent_id],
            )
            agent.draw(surface)

    def _draw_coins(self, state, surface):
        for coins_index in range(self.num_max_coins_per_agent):
            coins_xy = state["coins_xy_" + str(coins_index)]
            coins_xs, coins_ys = torch.split(coins_xy, 1, dim=1)
            coins_bool = self._coins_exist(coins_xy.unsqueeze(0)).squeeze(0)
            for coin_index in range(coins_bool.shape[0]):
                if bool(coins_bool[coin_index].cpu().item()):
                    coin = gizeh.circle(
                        r=self.display_scale / 2,
                        xy=self.grid_xy_to_pixel(
                            [
                                int(coins_xs[coin_index, 0].cpu().item()),
                                int(coins_ys[coin_index, 0].cpu().item()),
                            ],
                            shift_center=True,
                        ),
                        fill=self.agent_color_dict[coin_index],
                    )
                    coin.draw(surface)

    def grid_to_pixel_1d(self, coord: float, shift_center_bool=False) -> int:
        pixel_coord = coord * self.display_scale
        if shift_center_bool:
            pixel_coord += 0.5 * self.display_scale
        return int(round(pixel_coord))

    def grid_xy_to_pixel(
        self, xy_coord: List[int], shift_center: bool = False
    ) -> List[int]:
        assert len(xy_coord) == 2, "Exactly two coordinates expected!"
        x_pixel = self.grid_to_pixel_1d(xy_coord[0], shift_center)
        y_pixel = self.grid_to_pixel_1d(xy_coord[1], shift_center)
        return [x_pixel, y_pixel]

    def __str__(self):
        return "CoinGame"
