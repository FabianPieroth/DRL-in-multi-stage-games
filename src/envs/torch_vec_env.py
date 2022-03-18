"""Base classes for vectorized Gym environments"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

VecEnvIndices = Union[None, int, Iterable[int]]
TorchVecEnvStepReturn = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
]


class BaseEnvForVec(ABC):
    """TODO"""

    def __init__(self, config: Dict, device):
        self.device = device
        self.config = config
        self.num_agents = None

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
    def sample_new_states(self, n: int) -> Any:
        """?

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
    def get_observations(self, states) -> Any:
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


class TorchVecEnv(VecEnv):
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
        # self.render_n_envs = render_num_envs

        self.observation_space = model.observation_space
        self.action_space = model.action_space

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

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

        self.ep_stats["returns"] += rewards
        self.ep_stats["lengths"] += torch.ones((self.num_envs,), device=self.device)

        self.current_states = next_states

        n_dones = dones.sum()
        self.current_states[dones] = self.model.sample_new_states(n_dones)

        # episode_returns = self.ep_stats["returns"][dones]
        # episode_lengths = self.ep_stats["lengths"][dones]
        # infos = (episode_returns, episode_lengths)

        # TODO only support for constant length env.
        # otherwise we are slow AF on CPU
        infos = {}
        if dones.all():
            infos["terminal_observation"] = self.model.get_observations(
                self.current_states[dones]
            )  # TODO check

        self.ep_stats["returns"][dones] = 0
        self.ep_stats["lengths"][dones] = 0

        # Override observations for resetted environments after using them to
        # set "terminal_observation"
        obses[dones] = self.model.get_observations(self.current_states[dones])

        # TODO `collect_rollouts` needs arrays!?
        # if isinstance(self.actions, np.ndarray):
        #     return (
        #         np.array(obses.cpu()),
        #         np.array(rewards.cpu()),
        #         np.array(dones.cpu()),
        #         infos,
        #     )
        # else:
        return obses.clone(), rewards.clone(), dones.clone(), infos

    def step(self, actions: np.ndarray):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
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
        raise self.model.seed(seed)

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
