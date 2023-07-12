import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_linear_fn,
    is_vectorized_observation,
    polyak_update,
)
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F

from src.learners.off_policy_replay_buffer import ReplayBuffer
from src.learners.utils import (
    ActionNoise,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: a list of 2 or 3 elements. action_noise[0] is the type of action noise, action[1]
        is the related parameters, action_noise[2] if provided is the schedule.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(TD3, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.action_noise = action_noise

        self.q_net, self.q_net_target = (
            None,
            None,
        )  # Not really necessary (initialized in _create_aliases)
        self.num_collected_steps = 0
        self.num_timesteps = 0  # Not really necessary, define in BaseAlgorithm

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # TODO: Auto read action noise
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        self._create_aliases()

        # Get action noise and schedule
        self.action_noise_schedule = (
            self.action_noise[2]
            if (self.action_noise and len(self.action_noise) == 3)
            else None
        )

        if self.action_noise and self.action_noise[0] in [
            "NormalActionNoise",
            "OrnsteinUhlenbeckActionNoise",
        ]:
            if self.action_noise[0] == "OrnsteinUhlenbeckActionNoise":
                self.action_noise = OrnsteinUhlenbeckActionNoise(**self.action_noise[1])
            elif self.action_noise[0] == "NormalActionNoise":
                self.action_noise = NormalActionNoise(**self.action_noise[1])
            self.action_noise._mu = th.full(
                (self.n_envs, 1), self.action_noise._mu, device=self.device
            )
            self.action_noise._sigma = th.full(
                (self.n_envs, 1), self.action_noise._sigma, device=self.device
            )

        if self.action_noise_schedule and self.action_noise_schedule[
            "schedule_type"
        ] in ["Linear"]:
            # Support only Linear schedule
            self.action_noise_schedule = (
                get_linear_fn(**self.action_noise_schedule["params"]),
                self.action_noise._sigma,
            )

    ### end checking

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # Update action_noise
        self._update_action_noise()

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):  # check gradient_steps
            self._n_updates += 1

            # # Sample replay buffer (check if still work)
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            if th.isnan(replay_data.observations).any().item():
                obs_non_missing_data = th.any(~th.isnan(replay_data.observations), -1)
                replay_data = replay_data._replace(
                    observations=replay_data.observations[obs_non_missing_data],
                    actions=replay_data.actions[obs_non_missing_data],
                    next_observations=replay_data.next_observations[
                        obs_non_missing_data
                    ],
                    dones=replay_data.dones[obs_non_missing_data],
                    rewards=replay_data.rewards[obs_non_missing_data],
                )
            ### end checking

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                [
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                ]
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def predict(
        self,
        observation: th.Tensor,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        observation = observation.cpu().numpy()
        action, state = self.policy.predict(
            observation, state, episode_start, deterministic
        )
        action = th.from_numpy(action).to(self.device)

        if not deterministic:
            scaled_action = self.scale_action(action)
            assert -1.0 <= scaled_action.all() <= 1.0

            # Add noise to the action (improve exploration)
            if self.action_noise is not None:
                noise = self.action_noise()
                scaled_action = th.clamp(scaled_action + self.action_noise(), -1, 1)
            assert -1.0 <= scaled_action.all() <= 1.0
            action = self.unscale_action(scaled_action)

        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        return super(TD3, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(TD3, self)._excluded_save_params() + [
            "actor",
            "critic",
            "actor_target",
            "critic_target",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    # TODO: implement ingest_data
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
        # Only do once for all agents that share the same policy
        if not (policy_sharing and agent_id > 0):
            self.num_timesteps += self.env.num_envs
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            self.num_collected_steps += 1

        # Store data in replay buffer (normalized action and unnormalized observation)
        self._store_transition(
            self.replay_buffer,
            sa_last_obs,
            last_episode_starts,
            sa_actions,
            new_obs,
            sa_rewards,
            dones,
            infos,
            agent_id,
        )

        self._update_current_progress_remaining(
            self.num_timesteps, self._total_timesteps
        )

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        # self._on_step()

        if self.num_collected_steps > 0 and self.num_timesteps > self.learning_starts:
            if self.num_collected_steps % self.train_freq.frequency == 0:
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else int(
                        self.train_freq.frequency
                        * self.n_envs
                        / self.batch_size
                        * 3
                        / 2
                    )
                )
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
        return self

    # TODO: implement store transition to replay buffer
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        sa_last_obs: th.Tensor,
        last_episode_starts: th.Tensor,
        buffer_action: th.Tensor,
        new_obs: th.Tensor,
        reward: th.Tensor,
        dones: th.Tensor,
        infos: List[Dict[str, Any]],
        agent_id: int,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Avoid changing the original ones
        new_obs_, reward_ = new_obs, reward

        # Avoid modification by reference
        sa_next_obs = deepcopy(new_obs_[agent_id])
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if dones.any():
            sa_next_obs[dones] = infos.get("terminal_observation")[agent_id]

        sa_obs_array = sa_last_obs.detach().cpu().numpy()
        sa_next_obs = sa_next_obs.detach().cpu().numpy()
        buffer_action = buffer_action.detach().cpu().numpy()
        buffer_reward_ = reward_.detach().cpu().numpy()
        buffer_dones = dones.detach().cpu().numpy()
        replay_buffer.add(
            sa_obs_array,
            sa_next_obs,
            buffer_action,
            buffer_reward_,
            buffer_dones,
            infos,
        )

    # TODO: implement store get action
    def get_actions_with_data(
        self, sa_obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Tuple]:
        """
        Computes actions for the current state for env.step()

        Args:
            agent_id (int): determines which agents_actions will be returned

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        low, high = (
            th.Tensor(self.action_space.low).to(self.device),
            th.tensor(self.action_space.high).to(self.device),
        )

        unscaled_action, _ = self.predict(sa_obs, deterministic=False)

        assert low <= unscaled_action.all() <= high

        scaled_action = self.scale_action(unscaled_action)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        actions_for_env = unscaled_action
        assert low <= actions_for_env.all() <= high
        assert -1.0 <= buffer_action.all() <= 1.0

        additional_actions_data = ()
        return (actions_for_env, buffer_action, additional_actions_data)

    def scale_action(self, action: th.Tensor) -> th.Tensor:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = (
            th.Tensor(self.action_space.low).to(self.device),
            th.tensor(self.action_space.high).to(self.device),
        )
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: th.Tensor) -> th.Tensor:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = (
            th.Tensor(self.action_space.low).to(self.device),
            th.tensor(self.action_space.high).to(self.device),
        )
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        if dones is None:
            dones = th.tensor([False])
        if dones.all().detach().item():
            self.ep_info_buffer.extend([infos])

    def _update_action_noise(self) -> None:
        """
        Update the action noise sigma rate if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.action_noise and self.action_noise_schedule:
            # currently support only Normal action noise
            sigma_action_noise_rate = self.action_noise_schedule[0](
                self._current_progress_remaining
            )
            self.action_noise._sigma = (
                sigma_action_noise_rate * self.action_noise_schedule[1]
            )

    def __str__(self):
        name = self.__class__.__name__
        return name
