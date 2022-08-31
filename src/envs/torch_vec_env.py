"""Base classes for vectorized Gym environments"""
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.envs.space_translators import (
    BaseSpaceTranslator,
    IdentitySpaceTranslator,
    MultiDiscreteToDiscreteSpaceTranslator,
)

VecEnvIndices = Union[None, int, Iterable[int]]
TorchVecEnvStepReturn = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
]


class BaseEnvForVec(ABC):
    """Base Environment used for GPU based environment inference.
    Inherit from this class when writing a new env. Use MATorchVecEnv
    to wrap it."""

    def __init__(self, config: Dict, device):
        self.device = device
        self.config = config
        self.num_agents = self._get_num_agents()
        self.observation_spaces = self._init_observation_spaces()
        self.action_spaces = self._init_action_spaces()
        self.observation_space = None
        self.action_space = None

    @abstractmethod
    def _get_num_agents(self) -> int:
        """
        Returns:
            int: number of agents in env
        """

    @abstractmethod
    def _init_observation_spaces(self) -> Dict[int, Space]:
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """

    @abstractmethod
    def _init_action_spaces(self) -> Dict[int, Space]:
        """Returns dict with agent - action space pairs.
        Returns:
            Dict[int, Space]: agent_id: action space
        """

    @abstractmethod
    def to(self, device) -> Any:
        """Takes an available GPU device and shifts all tensors to the newly specified device.

        Args:
            device (int or string): The device to send the data to.
        Returns:
            Any: returns self of class
        """
        return self

    @abstractmethod
    def sample_new_states(self, n: int) -> torch.Tensor:
        """
        Args:
            n (int): Number of states to sample.

        Returns:
            States: Returns n states.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step(self, cur_states, actions: torch.Tensor):
        """
        Given a state batch and an action batch makes a step of env.
        Args:
            cur_states ():
            actions (torch.Tensor):
        Returns:
            observations:
            rewards:
            episode-done markers:
            updated_states:
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self, states) -> torch.Tensor:
        """Takes a number of states and returns the corresponding observations for the agents.

        Args:
            states (_type_): _description_

        Returns:
            observations: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state):
        """Takes a state and returns a tensor to render.

        Args:
            state (_type_):

        Returns:
            image:
        """

    def custom_evaluation(
        self,
        learners: Dict[int, BaseAlgorithm],
        env,
        writer,
        iteration: int,
        config: Dict,
    ):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            config: Dict of additional data
        """
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment-specific seeding is not used at the moment.
        In the underlying environment, random numbers are generated through
        calls to methods like torch.randn and can be seeded with
        `torch.manual_seed`.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


class MATorchVecEnv(VecEnv):
    """Vectorized Gym environment base class"""

    metadata = {"render.modes": ["console"]}

    def __init__(self, model, num_envs, device, render_num_envs=16):
        """
        VecEnv which takes care of parallelization itself natively on GPU
        instead of vectorizing a single non-batch environment.
        Based on VecEnv interface by stable-baselines3.

        :param model:
        :param num_envs:
        :param device:
        :param render_n_envs:
        """
        self.device = device
        self.model = model
        self.num_envs = num_envs
        self.actions = None
        self.current_states = self.model.sample_new_states(num_envs)
        self.ep_stats = {
            "returns": torch.zeros((num_envs,), device=device),
            "lengths": torch.zeros((num_envs,), device=device),
        }
        self.sa_ep_stats = {
            agent_id: {
                "returns": torch.zeros((num_envs,), device=device),
                "lengths": torch.zeros((num_envs,), device=device),
            }
            for agent_id in range(self.model.num_agents)
        }
        # self.render_n_envs = render_num_envs

        self.observation_spaces = model.observation_spaces
        self.action_spaces = model.action_spaces
        self.joint_spaces = {
            "action_space": self.action_spaces,
            "observation_space": self.observation_spaces,
        }
        self.observation_space = None
        self.action_space = None
        self.learner_type = {
            agent_id: None for agent_id in range(self.model.num_agents)
        }
        self.agent_translators = {
            "action_space": {
                agent_id: None for agent_id in range(self.model.num_agents)
            },
            "observation_space": {
                agent_id: None for agent_id in range(self.model.num_agents)
            },
        }
        self.available_translations = {MultiDiscrete: Discrete}
        self.non_valid_spaces_for_algos = {
            "action_space": {"dqn": [Box, MultiDiscrete, MultiBinary]},
            "observation_space": {},
        }

    def set_env_for_current_agent(self, agent_id: int, learner_type: str):
        """Sets the environment to the view of provided agent. 
        Needed to initialize leaners.
        """
        self.action_space = self._set_translators_for_learners(
            agent_id, learner_type, "action_space"
        )
        self.observation_space = self._set_translators_for_learners(
            agent_id, learner_type, "observation_space"
        )

    def _set_translators_for_learners(
        self, agent_id: int, learner_type: str, space_type: str
    ) -> Space:
        space_translator_class, translator_config = self._get_translator_type(
            agent_id, learner_type, space_type
        )
        agent_translator = space_translator_class(
            domain_space=self.joint_spaces[space_type][agent_id],
            config=translator_config,
        )
        self.agent_translators[space_type][agent_id] = agent_translator
        return agent_translator.image_space

    def _get_translator_type(
        self, agent_id: int, learner_type: str, space_type: str
    ) -> Tuple[BaseSpaceTranslator, Dict]:
        if self._non_identity_translation_needed(agent_id, learner_type, space_type):
            space_translator_class, translator_config = self._choose_non_identity_translator_class(
                agent_id, learner_type, space_type
            )
        else:
            space_translator_class, translator_config = IdentitySpaceTranslator, None
        return space_translator_class, translator_config

    def _choose_non_identity_translator_class(
        self, agent_id: int, learner_type: str, space_type: str
    ) -> Tuple[BaseSpaceTranslator, Dict]:
        agent_space = self.joint_spaces[space_type][agent_id]
        if (
            isinstance(agent_space, MultiDiscrete)
            and self.available_translations.get(type(agent_space)) == Discrete
        ):
            translator_class = MultiDiscreteToDiscreteSpaceTranslator
            translator_config = {"multi_space_shape": tuple(agent_space.nvec)}
        else:
            raise ValueError(
                "No valid non-identity translation available. You should not get here!"
            )
        return translator_class, translator_config

    def _non_identity_translation_needed(
        self, agent_id: int, learner_type: str, space_type: str
    ) -> Tuple[BaseSpaceTranslator, Dict]:
        non_valid_spaces_for_algo = self.non_valid_spaces_for_algos[space_type].get(
            learner_type
        )
        agent_space_type = type(self.joint_spaces[space_type][agent_id])
        if not non_valid_spaces_for_algo is not None:
            return False
        if not agent_space_type in non_valid_spaces_for_algo:
            return False
        if not self.available_translations.get(agent_space_type) is not None:
            return False
        if (
            not self.available_translations.get(agent_space_type)
            not in non_valid_spaces_for_algo
        ):
            return False
        return True

    def step_async(self, actions: Dict[int, torch.Tensor]) -> None:
        self.actions = self.translate_joint_actions(actions)

    def translate_joint_actions(
        self, actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        translated_actions = {}
        for agent_id, agent_actions in actions.items():
            translated_actions[agent_id] = self.agent_translators["action_space"][
                agent_id
            ].inv_translate(agent_actions)
        return translated_actions

    def step_wait(self) -> TorchVecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        with torch.no_grad():
            current_states = self.current_states

            if isinstance(self.actions, np.ndarray) or isinstance(self.actions, list):
                actions = torch.tensor(self.actions, device=self.device)
            else:
                # We are assuming that actions are on the correct device.
                actions = self.actions

            obses, rewards, dones, next_states = self.model.compute_step(
                current_states, actions
            )
        self.current_states = next_states

        n_dones = dones.sum()
        self.current_states[dones] = self.model.sample_new_states(n_dones)

        self.ep_stats["returns"] += torch.sum(
            torch.stack(tuple(rewards.values())), axis=0
        )  # global rewards
        self.ep_stats["lengths"] += torch.ones((self.num_envs,), device=self.device)
        episode_returns = self.ep_stats["returns"][dones]
        episode_lengths = self.ep_stats["lengths"][dones]
        self.ep_stats["returns"][dones] = 0
        self.ep_stats["lengths"][dones] = 0

        infos = dict(episode_returns=episode_returns, episode_lengths=episode_lengths)

        for agent_id in range(self.model.num_agents):
            self.sa_ep_stats[agent_id]["returns"] += rewards[agent_id]
            self.sa_ep_stats[agent_id]["lengths"] += torch.ones(
                (self.num_envs,), device=self.device
            )
            sa_episode_returns = self.sa_ep_stats[agent_id]["returns"][dones]
            sa_episode_lengths = self.sa_ep_stats[agent_id]["lengths"][dones]
            self.sa_ep_stats[agent_id]["returns"][dones] = 0
            self.sa_ep_stats[agent_id]["lengths"][dones] = 0
            infos[agent_id] = dict(
                sa_episode_returns=sa_episode_returns,
                sa_episode_lengths=sa_episode_lengths,
            )

        if dones.any():
            infos["terminal_observation"] = self.model.get_observations(
                self.current_states[dones]
            )

            # Override observations for resetted environments after using them to
            # set "terminal_observation"
            newly_sampled_obses = self.model.get_observations(
                self.current_states[dones]
            )
            for agent_id in range(self.model.num_agents):
                obses[agent_id][dones] = newly_sampled_obses[agent_id]
        # TODO: observation translator not in place yet!
        return (
            self.clone_tensor_dict(obses),
            self.clone_tensor_dict(rewards),
            dones.clone(),
            infos,
        )

    @staticmethod
    def clone_tensor_dict(
        dict_to_clone: Dict[Any, torch.tensor]
    ) -> Dict[Any, torch.tensor]:
        return {key: value.clone() for key, value in dict_to_clone.items()}

    def step(self, actions: np.ndarray):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observations, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment-specific seeding is not used at the moment.
        In the underlying environment, random numbers are generated through
        calls to methods like torch.randn and can be seeded with
        `torch.manual_seed`.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        self.current_states = self.model.sample_new_states(self.num_envs)
        obses = self.model.get_observations(self.current_states)
        return obses

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def render(self, mode: str) -> Optional[np.ndarray]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.model.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return self.model.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return self.model.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs

    def close(self) -> None:
        return self.model.close()
