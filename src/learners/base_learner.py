import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class SABaseAlgorithm(PPO, ABC):
    """
    Base class that extends Stable Baselines 3 PPO and a reduced version for
    the vanilla Reinforce algorithm to vectorized learning.
    """

    def __init__(self, action_dependent_std: bool = False, **kwargs):

        # We want to start off with a much lower variance
        self.log_std_init = -3.0  # default was 1 in SB3

        # Or have an action dependent std
        self.action_dependent_std = action_dependent_std

        super(SABaseAlgorithm, self).__init__(**kwargs)
        self._change_space_attributes_to_tensors()

    def _change_space_attributes_to_tensors(self):
        # convert boundaries to tensors if necessary
        if isinstance(self.action_space, gym.spaces.Box):
            if not isinstance(self.action_space.low, th.Tensor):
                self.action_space.low = th.tensor(self.action_space.low)
            if not isinstance(self.action_space.high, th.Tensor):
                self.action_space.high = th.tensor(self.action_space.high)

            # move boundaries to right device
            self.action_space.low = self.action_space.low.to(device=self.device)
            self.action_space.high = self.action_space.high.to(device=self.device)

    def prepare_next_rollout(self, env, callback):
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        self.rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

    def prepare_step(self, n_steps, env):
        if (
            self.use_sde
            and self.sde_sample_freq > 0
            and n_steps % self.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)

    @abstractmethod
    def get_actions_with_data(self, obs_tensor: th.Tensor):
        """Computes actions for the current state for env.step()

        Args:
            obs_tensor (th.Tensor): Observations the agent makes in the env.

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        pass

    def prepare_actions_for_buffer(self, actions):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
        return actions

    @abstractmethod
    def handle_dones(self, dones, infos, sa_rewards, agent_id: int):
        pass

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
        """This interface is used by the `MultiAgentCoordinator` to be the
        single point of information sharing with this learner.
        """
        # Only do once for all agents that share the same policy
        if not (policy_sharing and agent_id > 0):
            self.num_timesteps += self.env.num_envs
            self._update_info_buffer(infos, dones)

        # Add all data to buffer even when it comes from multiple agents
        # sharing the same policy -> more efficient usage of game data
        sa_actions = self.prepare_actions_for_buffer(sa_actions)
        sa_rewards = self.handle_dones(dones, infos, sa_rewards, agent_id)
        self.add_data_to_replay_buffer(
            sa_last_obs,
            last_episode_starts,
            sa_actions,
            sa_rewards,
            sa_additional_actions_data,
            agent_id,
        )

        if self.rollout_buffer.full:
            sa_new_obs = new_obs[agent_id]
            if policy_sharing:
                assert (
                    agent_id + 1
                ) == self.env.model.num_agents, (
                    "Rollout-buffer is assumed to be equally filled by all agents!"
                )
                sa_new_obs = new_obs
            self.postprocess_rollout(sa_new_obs, dones, policy_sharing)
            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )
            self.train()
            self.prepare_next_rollout(self.env, callback)

    def add_data_to_replay_buffer(
        self,
        sa_last_obs,
        last_episode_starts,
        sa_actions,
        sa_rewards,
        sa_additional_actions_data,
        agent_id: int,
    ):
        sa_values, sa_log_probs = sa_additional_actions_data
        self.rollout_buffer.add(
            sa_last_obs,
            sa_actions,
            sa_rewards,
            last_episode_starts,
            sa_values,
            sa_log_probs,
            th.ones(1, dtype=int) * agent_id,
        )

    @abstractmethod
    def postprocess_rollout(self, sa_new_obs, dones, policy_sharing: bool):
        pass

    @abstractmethod
    def _setup_model(self) -> None:
        pass

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :return:
        """
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose,
                self.tensorboard_log,
                self.__class__.__name__,
                reset_num_timesteps,
            )

        # Create eval callback if needed
        callback = self._init_callback(
            callback, eval_env, eval_freq, n_eval_episodes, log_path
        )

        self.prepare_next_rollout(self.env, callback)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones=None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        NOTE: Only supports fixed length games.
        """
        if dones is None:
            dones = th.tensor([False])
        if dones.any().detach().item():
            self.ep_info_buffer.extend([infos])

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[th.Tensor, ...]] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        self.policy.set_training_mode(False)

        with th.no_grad():
            actions = self.policy._predict(observation, deterministic=deterministic)

        if isinstance(self.policy.action_space, gym.spaces.Box):
            if self.policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = th.clip(
                    actions, self.policy.action_space.low, self.policy.action_space.high
                )

        return actions, state

    def __str__(self):
        name = self.__class__.__name__
        name = name[3:] if name.startswith("Vec") else name
        return name


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
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = True,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        """Most basic multi-agent base algorithm that is used, e.g., for non
        neural network based policies.
        """
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

    def predict(
        self,
        observation: th.Tensor,
        state: Optional[Tuple[th.Tensor, ...]] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Optional[Tuple[th.Tensor, ...]]]:
        raise NotImplementedError

    def get_actions_with_data(
        self, sa_obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Tuple]:
        """Computes actions for the current state for env.step()

        Args:
            agent_id (int): determines which agents_actions will be returned

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        raise NotImplementedError()

    def learn(self):
        pass

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        if dones is None:
            dones = th.tensor([False])
        if dones.all().detach().item():
            self.ep_info_buffer.extend([infos])

    def _setup_model(self) -> None:
        pass


class MABasePolicy(BasePolicy):
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        pass
        # super().__init__(*args, squash_output=squash_output, **kwargs)

    def _predict(self):
        pass

    def forward(self):
        pass
