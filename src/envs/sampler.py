"""This class provides primitives to implement samplers that support drawing of
   valuation and observation profiles for a set of players."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from torch.cuda import _device_t as Device
from torch.distributions import Distribution


class ValuationObservationSampler(ABC):
    """Provides functionality to draw valuation and observation profiles."""

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        self.num_agents: int = (
            num_agents  # The number of players in the valuation profile
        )
        self.valuation_size: int = (
            valuation_size  # The dimensionality / length of a single valuation vector
        )
        self.observation_size: int = observation_size  # The dimensionality / length of a single observation vector
        self.default_batch_size: int = default_batch_size  # a default batch size
        self.default_device: Device = (
            (default_device or "cuda") if torch.cuda.is_available() else "cpu"
        )
        self.sampler_config = sampler_config  # The sampler specific config
        self.support_bounds = self._init_support_bounds()
        assert self.support_bounds.size() == torch.Size(
            [num_agents, valuation_size, 2]
        ), "invalid support bounds."
        self.support_bounds: torch.FloatTensor = self.support_bounds.to(
            self.default_device
        )

    def _parse_batch_sizes_arg(
        self, batch_sizes_argument: Union[int, List[int], None]
    ) -> List[int]:
        """Parses an integer batch_size_argument into a list. If none given,
        defaults to the list containing the default_batch_size of the instance.
        """
        if batch_sizes_argument is not None:
            batch_sizes = batch_sizes_argument
        else:
            batch_sizes = self.default_batch_size
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        return batch_sizes

    @abstractmethod
    def _init_support_bounds(self) -> torch.FloatTensor:
        """We need the valuations and observations to come from bounded intervals.
        For each agent and valuation dim, a lower and upper bound needs to be provided.

        Return (torch.FloatTensor): shape=[num_agents, valuation_size, 2]
        """

    @abstractmethod
    def draw_profiles(
        self, batch_sizes: Union[int, List[int]] = None, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs:
            batch_sizes (optional): List[int], the batch_sizes to draw. If none provided,
            `[self.default_batch_size]` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (*batch_sizes x num_agents x valuation_size): a valuation profile
            observations: torch.Tensor (*batch_sizes x num_agents x observation_size): an observation profile
        """


class PVSampler(ValuationObservationSampler, ABC):
    """A sampler for Private Value settings, i.e. when observations and
    valuations are identical.
    """

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        super().__init__(
            num_agents,
            valuation_size,
            observation_size,
            sampler_config,
            default_batch_size,
            default_device,
        )

    @abstractmethod
    def _sample(
        self, batch_sizes: Union[int, List[int]], device: Device
    ) -> torch.Tensor:
        """Returns a batch of profiles (which are both valuations and observations)"""

    def draw_profiles(
        self, batch_sizes: Union[int, List[int]] = None, device: Device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device
        # In the PV setting, valuations and observations are identical.
        profile = self._sample(batch_sizes, device)
        return profile, profile


class CompositeValuationObservationSampler(ValuationObservationSampler):
    """A class representing composite prior distributions that are
    made up of several groups of bidders, each of which can be represented by
    an atomic ValuationObservationSampler, and which are independent between-group
    (but not necessarily within-group).

    Limitation: The current implementation requires that all players nevertheless
    have the same valuation_size.
    """

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        subgroup_samplers: List[ValuationObservationSampler],
        default_batch_size=1,
        default_device=None,
    ):
        self.n_groups = len(subgroup_samplers)
        self.group_sizes = [sampler.num_agents for sampler in subgroup_samplers]
        assert (
            sum(self.group_sizes) == num_agents
        ), "number of players in subgroup don't match total n_players."
        for sampler in subgroup_samplers:
            assert (
                sampler.valuation_size == valuation_size
            ), "incorrect valuation size in subgroup sampler."
            assert (
                sampler.observation_size == observation_size
            ), "incorrect observation size in subgroup sampler"

        self.group_samplers = subgroup_samplers
        self.group_indices: List[torch.IntTensor] = [
            torch.tensor(
                range(sum(self.group_sizes[:i]), sum(self.group_sizes[: i + 1]))
            )
            for i in range(self.n_groups)
        ]

        # concatenate bounds in player dimension
        support_bounds = torch.vstack([s.support_bounds for s in self.group_samplers])

        super().__init__(
            num_agents,
            valuation_size,
            observation_size,
            support_bounds,
            default_batch_size,
            default_device,
        )

    def draw_profiles(
        self, batch_sizes: Union[int, List[int]] = None, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs:
            batch_sizes (optional): List[int], the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (*batch_sizes x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (*batch_sizes x n_players x observation_size): an observation profile
        """
        device = device or self.default_device
        batch_sizes: List[int] = self._parse_batch_sizes_arg(batch_sizes)

        v = torch.empty(
            [*batch_sizes, self.num_agents, self.valuation_size], device=device
        )
        o = torch.empty(
            [*batch_sizes, self.num_agents, self.observation_size], device=device
        )

        # Draw independently for each group.

        for g in range(self.n_groups):
            # player indices in the group
            players = self.group_indices[g]
            v[..., players, :], o[..., players, :] = self.group_samplers[
                g
            ].draw_profiles(batch_sizes, device)

        return v, o

    def _init_support_bounds(self) -> torch.FloatTensor:
        return torch.vstack([s.support_bounds for s in self.group_samplers])

    def generate_valuation_grid(self, **kwargs) -> torch.Tensor:
        """Possibly need to call specific sampling"""
        for g in range(self.n_groups):  # iterate over groups
            player_positions = self.group_indices[g]  # player_positions within group
            for pos in player_positions:
                if kwargs["player_position"] == pos:
                    kwargs["player_position"] = pos - sum(
                        self.group_sizes[:g]
                    )  # i's relative position in subgroup
                    return self.group_samplers[g].generate_valuation_grid(**kwargs)

    def generate_reduced_grid(self, **kwargs) -> torch.Tensor:
        """Possibly need to call specific sampling"""
        for g in range(self.n_groups):  # iterate over groups
            player_positions = self.group_indices[g]  # player_positions within group
            for pos in player_positions:
                if kwargs["player_position"] == pos:
                    kwargs["player_position"] = pos - sum(
                        self.group_sizes[:g]
                    )  # i's relative position in subgroup
                    return self.group_samplers[g].generate_reduced_grid(**kwargs)

    def generate_action_grid(self, **kwargs) -> torch.Tensor:
        """Possibly need to call specific sampling"""
        for g in range(self.n_groups):  # iterate over groups
            player_positions = self.group_indices[g]  # player_positions within group
            for pos in player_positions:
                if kwargs["player_position"] == pos:
                    kwargs["player_position"] = pos - sum(
                        self.group_sizes[:g]
                    )  # i's relative position in subgroup
                    return self.group_samplers[g].generate_action_grid(**kwargs)

    def generate_cell_partition(self, **kwargs) -> torch.Tensor:
        """Possibly need to call specific sampling"""
        for g in range(self.n_groups):  # iterate over groups
            player_positions = self.group_indices[g]  # player_positions within group
            for pos in player_positions:
                if kwargs["player_position"] == pos:
                    kwargs["player_position"] = pos - sum(
                        self.group_sizes[:g]
                    )  # i's relative position in subgroup
                    return self.group_samplers[g].generate_cell_partition(**kwargs)


class SymmetricIPVSampler(PVSampler):
    """A Valuation Oracle that draws valuations independently and symmetrically
    for all players and each entry of their valuation vector according to a specified
    distribution.

    This base class works with all torch.distributions but requires sampling on
    cpu then moving to the device. When using cuda, use the faster,
    distribution-specific subclasses instead where provided.

    """

    UPPER_BOUND_QUARTILE_IF_UNBOUNDED = 0.999

    def __init__(
        self,
        distribution: Distribution,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        """
        Args:
            distribution: a single-dimensional torch.distributions.Distribution.
            num_agents: the number of players
            valuation_size: the length of each valuation vector
            default_batch_size: the default batch size for sampling from this instance
            default_device: the default device to draw valuations. If none given,
                uses 'cuda' if available, 'cpu' otherwise
        """
        self.base_distribution = distribution
        self.distribution = self.base_distribution.expand([num_agents, valuation_size])

        super().__init__(
            num_agents,
            valuation_size,
            observation_size,
            sampler_config,
            default_batch_size,
            default_device,
        )

    def _init_support_bounds(self) -> torch.FloatTensor:
        # bounds: use real support, unless unbounded:
        support = self.base_distribution.support
        if isinstance(support, torch.distributions.constraints._Real):
            upper_bound = self.base_distribution.icdf(
                torch.tensor(self.UPPER_BOUND_QUARTILE_IF_UNBOUNDED)
            )
            lower_bound = torch.tensor(0, device=upper_bound.device)
        else:
            lower_bound = torch.tensor(support.lower_bound).relu()
            upper_bound = torch.tensor(support.upper_bound)

        assert upper_bound >= lower_bound

        # repeat support bounds across all players and valuation dimensions
        return torch.stack([lower_bound, upper_bound]).repeat(
            [self.num_agents, self.valuation_size, 1]
        )

    def _sample(self, batch_sizes: int or List[int], device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in PV)"""
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        return self.distribution.sample(batch_sizes).to(device)


class UniformSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with symmetric Uniform priors."""

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        distribution = torch.distributions.uniform.Uniform(
            low=sampler_config.prior_low, high=sampler_config.prior_high
        )
        super().__init__(
            distribution,
            num_agents,
            valuation_size,
            observation_size,
            sampler_config,
            default_batch_size,
            default_device,
        )

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # create an empty tensor on the output device, then sample in-place
        return torch.empty(
            [*batch_sizes, self.num_agents, self.valuation_size], device=device
        ).uniform_(self.base_distribution.low, self.base_distribution.high)


class GaussianSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with symmetric Gaussian priors."""

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        """Constructs a Gaussian sampler."""
        distribution = torch.distributions.normal.Normal(
            loc=sampler_config.mean, scale=sampler_config.stddev
        )
        super().__init__(
            distribution,
            num_agents,
            valuation_size,
            observation_size,
            sampler_config,
            default_batch_size,
            default_device,
        )

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        # create empty tensor, sample in-place, clip
        return (
            torch.empty(
                [*batch_sizes, self.num_agents, self.valuation_size], device=device
            )
            .normal_(self.base_distribution.loc, self.base_distribution.scale)
            .relu_()
        )


class BertrandSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with a CFD of $F(c) = 0.5(c + c^2)$."""

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        """Constructs a Bertrand sampler."""
        distribution = torch.distributions.uniform.Uniform(
            low=sampler_config.prior_low, high=sampler_config.prior_high
        )
        super().__init__(
            distribution,
            num_agents,
            1,
            2,
            sampler_config,
            default_batch_size,
            default_device,
        )

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # create an empty tensor on the output device, then sample in-place
        uniform = torch.empty(
            [*batch_sizes, self.num_agents, self.valuation_size], device=device
        ).uniform_(self.base_distribution.low, self.base_distribution.high)

        # do inverse CDF transform
        icdf = lambda x: -0.5 + 0.5 * torch.sqrt(8 * x + 1)
        return icdf(uniform)


class MineralRightsValuationObservationSampler(ValuationObservationSampler):
    """The 'Mineral Rights' model is a common value model:
    There is a uniformly distributed common value of the item(s),
    each agent's  observation is then uniformly drawn from U[0,2v].
    See Kishna (2009), Example 6.1
    """

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        """
        Args:
            num_agents
            valuation_size
            observaion_size
            sampler_config.common_value_low: lower bound for uniform common value
            sampler_config.common_value_high: upper bound for uniform common value
            default_batch_size
            default_device
        """

        assert sampler_config.common_value_low >= 0, "valuations must be nonnegative"
        assert (
            sampler_config.common_value_high >= sampler_config.common_value_low
        ), "upper bound must larger than lower bound"

        self._common_value_low = sampler_config.common_value_low
        self._common_value_high = sampler_config.common_value_high

        super().__init__(
            num_agents,
            valuation_size,
            observation_size,
            sampler_config,
            default_batch_size,
            default_device,
        )

    def _init_support_bounds(self) -> torch.FloatTensor:
        # NOTE: This differs from the bnelearn implementation as we give the bounds for the signal instead of valuation!
        return torch.tensor(
            [self._common_value_low, 2 * self._common_value_high]
        ).repeat([self.num_agents, self.valuation_size, 1])

    def draw_profiles(
        self, batch_sizes: List[int] = None, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        common_value = (self._common_value_high - self._common_value_low) * torch.empty(
            [*batch_sizes, 1, self.valuation_size], device=device
        ).uniform_() + self._common_value_low

        valuations = common_value.repeat(
            [*([1] * len(batch_sizes)), self.num_agents, 1]
        )

        individual_factors = torch.empty_like(valuations).uniform_()
        observations = 2 * individual_factors * common_value

        return valuations, observations

    def generate_valuation_grid(
        self,
        player_position: int,
        minimum_number_of_points: int,
        dtype=torch.float,
        device=None,
        support_bounds: torch.Tensor = None,
    ) -> torch.Tensor:
        """This setting needs larger bounds for the grid."""
        return 2 * super().generate_valuation_grid(
            player_position=player_position,
            minimum_number_of_points=minimum_number_of_points,
            dtype=dtype,
            device=device,
            support_bounds=support_bounds,
        )


class AffiliatedValuationObservationSampler(ValuationObservationSampler):
    """The 'Affiliated Values Model' model. (Krishna 2009, Example 6.2).
    This is a private values model.

    Two bidders have signals

     .. math::
     o_i = z_i + s

     and valuations
     .. math::
     v_i = s + (z_1+z_2)/2 = mean_i(o_i)

     where z_i and s are i.i.d. standard uniform.
    """

    def __init__(
        self,
        num_agents: int,
        valuation_size: int,
        observation_size: int,
        sampler_config,
        default_batch_size=1,
        default_device=None,
    ):
        """
        Args:
            num_agents
            valuation_size
            observation_size
            sampler_config.u_low: lower bound for uniform distribution of z_i and s
            sampler_config.u_high: upper bound for uniform distribtuion of z_i and s
            default_batch_size
            default_device
        """

        assert sampler_config.u_low >= 0, "valuations must be nonnegative"
        assert (
            sampler_config.u_high > sampler_config.u_low
        ), "upper bound must larger than lower bound"

        self._u_low = sampler_config.u_low
        self._u_high = sampler_config.u_high

        super().__init__(
            num_agents,
            valuation_size,
            observation_size,
            sampler_config=sampler_config,
            default_batch_size=default_batch_size,
            default_device=default_device,
        )

    def _init_support_bounds(self) -> torch.FloatTensor:
        return torch.tensor([self._u_low, 2 * self._u_high]).repeat(
            [self.num_agents, self.valuation_size, 1]
        )

    def draw_profiles(
        self, batch_sizes: List[int] = None, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        z_and_s = torch.empty(
            [*batch_sizes, self.num_agents + 1, self.valuation_size], device=device
        ).uniform_(self._u_low, self._u_high)

        weights_v = torch.column_stack(
            [
                torch.ones([self.num_agents] * 2, device=device) / self.num_agents,
                torch.ones([self.num_agents, 1], device=device),
            ]
        )

        weights_o = torch.column_stack(
            [
                torch.eye(self.num_agents, device=device),
                torch.ones([self.num_agents, 1], device=device),
            ]
        )

        # dim u represents the n+1 uniform vectors
        valuations = torch.einsum("buv,nu->bnv", z_and_s, weights_v)
        observations = torch.einsum("buv,nu->bnv", z_and_s, weights_o)

        return valuations, observations
