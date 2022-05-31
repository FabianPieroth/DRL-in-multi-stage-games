from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.learners.base_learner import MABaseAlgorithm


class RPSDummyLearner(MABaseAlgorithm):
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
        self.dummy_action = config["action"]
        self.action_dict = {"ROCK": 0, "PAPER": 1, "SCISSORS": 2}

    def get_actions_with_data(
        self, agent_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Computes actions for the current state for env.step()

        Args:
            agent_id (int): determines which agents_actions will be returned

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        action_to_play = self.action_dict[self.dummy_action]
        num_envs = self._last_obs[agent_id].shape[0]
        actions = torch.ones([num_envs], dtype=int, device=self.device) * action_to_play
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
        action_to_play = self.action_dict[self.dummy_action]
        num_envs = observation.shape[0]
        actions = torch.ones([num_envs], dtype=int, device=self.device) * action_to_play
        return actions, state

    def ingest_data_to_learner(
        self,
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
