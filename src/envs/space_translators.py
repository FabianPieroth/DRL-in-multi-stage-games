import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

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


class BoxToDiscreteSpaceTranslator(BaseSpaceTranslator):
    def __init__(self, domain_space: Space, config: Dict) -> None:
        if not isinstance(domain_space, Box):
            raise ValueError("The domain space must be Box for this translator!")
        self.granularity = config["granularity"]
        super().__init__(domain_space, config)
        self.lower_bound, self.upper_bound = self._get_bounds_for_domain_space()
        if self.config["granularity"] < 2:
            raise ValueError("The discretization cannot be lower than 2!")

    def _set_image_space(self) -> Space:
        return Discrete(self.granularity)

    def _get_bounds_for_domain_space(self) -> Tuple[float]:
        if self.domain_space.shape != (1,):
            raise ValueError(
                "The translator is only implemented for a single dimension!"
            )
        if not self.domain_space.bounded_below and not self.domain_space.bounded_above:
            warnings.warn(
                "The box domain space is unbounded. We set the center to 0.0 with a maximum width of "
                + str(self.config["maximum_width"])
            )
            lower_bound = -self.config["maximum_width"] / 2.0
            upper_bound = self.config["maximum_width"] / 2.0
        elif not self.domain_space.bounded_below:
            upper_bound = self.domain_space.high[0]
            lower_bound = upper_bound - self.config["maximum_width"]
            warnings.warn(
                "The box domain space is not bounded from below. We set the lower bound to "
                + str(lower_bound)
                + "while keeping the upper bound at "
                + str(upper_bound)
            )
        elif not self.domain_space.bounded_above:
            lower_bound = self.domain_space.low[0]
            upper_bound = lower_bound + self.config["maximum_width"]
            warnings.warn(
                "The box domain space is not bounded from above. We set the upper bound to "
                + str(upper_bound)
                + "while keeping the lower bound at "
                + str(lower_bound)
            )
        else:
            lower_bound = self.domain_space.low[0]
            upper_bound = self.domain_space.high[0]
        return lower_bound, upper_bound

    def translate(self, data: torch.Tensor) -> torch.Tensor:
        half_bin_width = (self.upper_bound - self.lower_bound) / (
            2.0 * (self.granularity - 1)
        )
        bins = torch.linspace(
            self.lower_bound + half_bin_width,
            self.upper_bound - half_bin_width,
            steps=self.granularity - 1,
        ).to(data.device)
        return torch.bucketize(data, bins).squeeze()

    def inv_translate(self, data: torch.Tensor) -> torch.Tensor:
        data_with_add_dim = data[:, None]
        return (
            data_with_add_dim
            / (self.granularity - 1)
            * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )
