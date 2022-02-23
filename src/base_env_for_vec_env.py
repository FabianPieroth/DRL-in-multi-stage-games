from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseEnvForVec(ABC):
    def __init__(self, config: Dict, device):
        self.device = device
        self.config = config

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
