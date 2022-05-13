from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class MABaseAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        agent_id: int,
        config: Dict,
        policy: Type[BasePolicy] = None,
        env: Union[GymEnv, str, None] = None,
        policy_base: Type[BasePolicy] = "MABasePolicy",
        learning_rate: Union[float, Schedule] = 0.0,
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
        self.agent_id = agent_id
        self.config = config
        if policy is None:
            policy = MABasePolicy()
        super().__init__(
            policy,
            env,
            policy_base,
            learning_rate,
            policy_kwargs,
            tensorboard_log,
            verbose,
            device,
            support_multi_env,
            create_eval_env,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            supported_action_spaces,
        )
        self.n_steps = None

    def prepare_rollout(self, env, callback):
        callback.on_rollout_start()

    def prepare_step(self, n_steps, env):
        pass

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
        raise NotImplementedError()

    def prepare_actions_for_buffer(self, actions):
        return actions

    def handle_dones(self, dones, infos, rewards, agent_id: int) -> torch.Tensor:
        return rewards

    def add_data_to_replay_buffer(
        self, actions, rewards, additional_actions_data, agent_id: int
    ):
        pass

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        if dones is None:
            dones = torch.tensor([False])
        if dones.all().detach().item():
            self.ep_info_buffer.extend([infos])

    def update_internal_state_after_step(self, new_obs, dones):
        pass

    def postprocess_rollout(self, sa_new_obs, dones):
        pass

    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ):
        pass

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int) -> "BaseAlgorithm":
        pass

    def train(self):
        pass


class MABasePolicy(BasePolicy):
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        pass
        # super().__init__(*args, squash_output=squash_output, **kwargs)

    def _predict(self):
        pass

    def forward(self):
        pass
