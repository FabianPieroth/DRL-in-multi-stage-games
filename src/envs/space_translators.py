from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from gym.spaces import Box, Discrete, MultiDiscrete, Space

import src.utils_folder.spaces_utils as sp_ut


class BaseSpaceTranslator(ABC):
    """ Handles a mapping from one gym space to another. 
    These might be lossfull mappings.
    """

    def __init__(self, domain_space: Space, config: Optional[Dict] = None) -> None:
        self.config = config
        self.domain_space = domain_space
        self.image_space = self._set_image_space()

    @abstractmethod
    def _set_image_space(self) -> Space:
        return NotImplemented

    @abstractmethod
    def translate(self, data: torch.Tensor) -> torch.Tensor:
        return NotImplemented

    @abstractmethod
    def inv_translate(self, data: torch.Tensor) -> torch.Tensor:
        return NotImplemented


class IdentitySpaceTranslator(BaseSpaceTranslator):
    def _set_image_space(self) -> Space:
        return self.domain_space

    def translate(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inv_translate(self, data: torch.Tensor) -> torch.Tensor:
        return data


class MultiDiscreteToDiscreteSpaceTranslator(BaseSpaceTranslator):
    def __init__(self, domain_space: Space, config: Dict) -> None:
        if not isinstance(domain_space, MultiDiscrete):
            raise ValueError(
                "The domain space must be MultiDiscrete for this translator!"
            )
        super().__init__(domain_space, config)

    def _set_image_space(self) -> Space:
        prod_space_size = torch.prod(
            torch.tensor(self.config["multi_space_shape"])
        ).item()
        return Discrete(prod_space_size)

    def translate(self, data: torch.Tensor) -> torch.Tensor:
        return sp_ut.multidiscrete_to_discrete(data, self.config["multi_space_shape"])

    def inv_translate(self, data: torch.Tensor) -> torch.Tensor:
        return sp_ut.discrete_to_multidiscrete(data, self.config["multi_space_shape"])
