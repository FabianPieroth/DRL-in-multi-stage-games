"""Customized PPO learner and corresponding buffer"""
import time
from collections import deque
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from torch.nn import functional as F

from src.learners.utils import explained_variance


class VecPPO(PPO):
    """
    Extends Stable Baselines 3 PPO to vectorized learning.
    """

    def __init__(self, **kwargs):

        # We want to start off with a much lower variance
        self.log_std_init = -3.0  # default: 1
        # TODO: possibly try out pretraining

        super(VecPPO, self).__init__(**kwargs)
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

    def prepare_rollout(self, env, callback):
        assert self._last_obs is not None, "No previous observation was provided"
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

    def get_actions_with_data(self, agent_id: int):
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = self._last_obs[agent_id]
            actions, values, log_probs = self.policy.forward(obs_tensor)

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = th.clip(
                actions, self.action_space.low, self.action_space.high
            )
        return clipped_actions, actions, (values, log_probs)

    def prepare_actions_for_buffer(self, actions):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
        return actions

    def handle_dones(self, dones, infos, sa_rewards, agent_id: int):
        # change for vectorized capability
        # TODO: limitation: only constant length games
        if dones.any():
            terminal_obs = infos["terminal_observation"][agent_id]
            with th.no_grad():
                terminal_value = self.policy.predict_values(terminal_obs)[:, 0]
            sa_rewards[dones] += self.gamma * terminal_value
        return sa_rewards

    def add_data_to_replay_buffer(
        self, sa_actions, sa_rewards, sa_additional_actions_data, agent_id: int
    ):
        sa_values, sa_log_probs = sa_additional_actions_data
        self.rollout_buffer.add(
            self._last_obs[agent_id],
            sa_actions,
            sa_rewards,
            self._last_episode_starts,
            sa_values,
            sa_log_probs,
            th.ones(1, dtype=int) * agent_id,
        )

    def update_internal_state_after_step(self, new_obs, dones):
        self._last_obs = new_obs
        self._last_episode_starts = dones

    def postprocess_rollout(self, sa_new_obs, dones, policy_sharing: bool):
        with th.no_grad():
            if policy_sharing:
                values = {
                    agent_id: self.policy.predict_values(sa_obs).squeeze()
                    for agent_id, sa_obs in sa_new_obs.items()
                }
            else:
                values = self.policy.predict_values(sa_new_obs).squeeze()

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones, policy_sharing=policy_sharing
        )

    def _setup_model(self) -> None:
        # 1. `_setup_model` from `BaseLearner` PPO
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Here we need the customized `VecRolloutBuffer`
        # buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
        buffer_cls = VecRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # 2. `_setup_model` from `PPO`
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
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
        :param tb_log_name: the name of the run for tensorboard log
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

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = (
                self.env.reset()
            )  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = th.ones(
                (self.env.num_envs,), dtype=bool, device=self.device
            )
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        # Create eval callback if needed
        callback = self._init_callback(
            callback, eval_env, eval_freq, n_eval_episodes, log_path
        )

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
        if dones.all().detach().item():
            self.ep_info_buffer.extend([infos])

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
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
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
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


class VecRolloutBuffer(RolloutBuffer):
    """
    Extends Stable Baselines 3 RolloutBuffer to vectorzied learning.
    """

    def reset(self) -> None:

        self.observations = th.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=th.float32,
            device=self.device,
        )
        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )
        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.returns = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.episode_starts = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.values = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.log_probs = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.advantages = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.agent_ids = th.zeros((self.buffer_size,), dtype=th.int, device=self.device)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
        agent_ids: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # TODO: do we need `copy()` here? (as in original version) @Nils: could they be changed later on?
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.agent_ids[self.pos] = agent_ids
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: th.Tensor, policy_sharing: bool = False
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        last_gae_lam = 0
        if policy_sharing:  # TODO: Policy sharing needs to be handled in a cleaner way!
            last_values = {
                agent_id: sa_last_values.clone()
                for agent_id, sa_last_values in last_values.items()
            }
        else:
            last_values = last_values.clone()

        for step in reversed(range(self.buffer_size)):
            if policy_sharing:
                sa_last_values = last_values[self.agent_ids[step].detach().item()]
            else:
                sa_last_values = last_values
            if step == self.buffer_size - 1 and not policy_sharing:
                next_non_terminal = th.logical_not(dones)
                next_values = sa_last_values
            elif step >= self.buffer_size - len(last_values) and policy_sharing:
                next_non_terminal = th.logical_not(dones)
                next_values = sa_last_values
            else:
                step_size_to_data = self._find_step_size_to_agent_data(step)
                next_non_terminal = 1.0 - self.episode_starts[step + step_size_to_data]
                next_values = self.values[step + step_size_to_data]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def _find_step_size_to_agent_data(self, step):
        step_size = 1
        target_agent_id = self.agent_ids[step].detach().item()
        while target_agent_id != self.agent_ids[step + step_size].detach().item():
            step_size += 1
        return step_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(data))
