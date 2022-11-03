from abc import ABC, abstractmethod
from typing import Dict

from src.envs.torch_vec_env import BaseEnvForVec, VerifiableEnv
from src.learners.base_learner import SABaseAlgorithm


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    @abstractmethod
    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        pass

    @staticmethod
    def _check_if_env_is_compatible(env: BaseEnvForVec) -> bool:
        return isinstance(env.model, VerifiableEnv)
