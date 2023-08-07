from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

import src.utils.policy_utils as pl_ut
from src.learners.base_learner import MABaseAlgorithm


class RandomLearner(MABaseAlgorithm):
    def __init__(
        self,
        agent_id: int,
        config: Dict,
        policy: Type[BasePolicy] = None,
        env: Union[GymEnv, str, None] = None,
        policy_base: Type[BasePolicy] = ...,
        learning_rate: Union[float, Schedule] = 0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
        support_multi_env: bool = True,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super().__init__(
            agent_id,
            config,
            policy,
            env,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    def get_actions_with_data(
        self, sa_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Draws random actions for the given action spaces.
        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        actions = pl_ut.sample_random_actions(
            self.action_space, sa_obs.shape[0], self.device
        )
        actions_for_env = actions
        additional_actions_data = ()
        return actions_for_env, actions, additional_actions_data

    def predict(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        actions = pl_ut.sample_random_actions(
            self.action_space, observation.shape[0], self.device
        )
        return actions, state

    def ingest_data_to_learner(
        self,
        sa_last_obs,
        last_episode_starts,
        sa_actions,
        sa_rewards,
        sa_additional_actions_data,
        dones,
        infos,
        new_obs,
        agent_id: int,
        policy_sharing: bool,
        callback,
    ):
        self.num_timesteps += self.env.num_envs
