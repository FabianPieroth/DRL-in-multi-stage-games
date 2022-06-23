from abc import ABC, abstractmethod
from typing import Dict

from src.learners.base_learner import SABaseAlgorithm


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    @abstractmethod
    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        pass
