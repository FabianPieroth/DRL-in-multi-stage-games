from typing import Tuple

import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution

from src.envs.sampler import (
    AffiliatedValuationObservationSampler,
    GaussianSymmetricIPVSampler,
    MineralRightsValuationObservationSampler,
    UniformSymmetricIPVSampler,
    ValuationObservationSampler,
)


class DiagGaussianDistributionWithVariableStd(DiagGaussianDistribution):
    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Linear(latent_dim, self.action_dim)
        return mean_actions, log_std

    # TODO @Fabian: Do we have to worry about `log_prob` and `entropy` being different here?


def get_sampler(
    num_agents: int,
    valuation_size: int,
    observation_size: int,
    sampler_config,
    default_batch_size=1,
    default_device=None,
) -> ValuationObservationSampler:
    if sampler_config.name == "symmetric_uniform":
        sampler = UniformSymmetricIPVSampler
    elif sampler_config.name == "symmetric_gaussian":
        sampler = GaussianSymmetricIPVSampler
    elif sampler_config.name == "mineral_rights_common_value":
        sampler = MineralRightsValuationObservationSampler
    elif sampler_config.name == "affiliated_uniform":
        sampler = AffiliatedValuationObservationSampler
    else:
        raise ValueError(
            "No valid prior specified! Please check: " + sampler_config.name
        )
    return sampler(
        num_agents,
        valuation_size,
        observation_size,
        sampler_config,
        default_batch_size,
        default_device,
    )
