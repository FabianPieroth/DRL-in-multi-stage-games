import warnings
from typing import Dict

from gym import spaces

from src.learners.base_learner import SABaseAlgorithm
from src.verifier.base_verifier import BaseVerifier


class DiscreteVerifier(BaseVerifier):
    """Verifier for games with discrete action spaces."""

    def __init__(self, env, num_envs: int = 128):
        self.env = env
        self.env_is_compatible_with_verifier = self._check_if_env_is_compatible(env)
        self.num_agents = self.env.model.num_agents
        self.action_dim = env.model.ACTION_DIM

        self.device = self.env.device

        # Discrete actions
        if all(
            isinstance(s, spaces.Discrete) for s in env.model.action_spaces.values()
        ):
            pass
        else:
            raise ValueError("This verifier is for discrete actions only.")

        self.num_own_envs = num_envs
        self.num_opponent_envs = num_envs
        self.num_rounds_to_play = self.env.model.num_rounds_to_play

    def verify(self, learners: Dict[int, SABaseAlgorithm]):
        warnings.warn("Discrete verifier not implemented yet.")
        return [-1] * len(learners)
