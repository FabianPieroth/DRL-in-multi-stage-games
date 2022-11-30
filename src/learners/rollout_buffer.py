"""Rollout Buffer"""
from abc import ABC, abstractmethod
from typing import Dict, Generator, NamedTuple, Optional

import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SimpleRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    returns: th.Tensor


class VecBaseBuffer(RolloutBuffer, ABC):
    """
    Base class for vectorized buffers.
    """

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _get_samples(
        self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None
    ) -> SimpleRolloutBufferSamples:
        pass

    def _find_step_size_to_agent_data(self, step):
        """When `policy_sharing` is turned on, the buffer receives inputs for
        all agents, thus we need to keep track of how many inputs relate to one
        time step. Should default to 1 if buffer is only responsible for a
        single agent.
        """
        step_size = 1
        target_agent_id = self.agent_ids[step].detach().item()
        while target_agent_id != self.agent_ids[step + step_size].detach().item():
            step_size += 1
        return step_size


class SimpleVecRolloutBuffer(VecBaseBuffer):
    """
    Extends Stable Baselines 3 RolloutBuffer to vectorized learning.

    Does not use the lambda-return and GAE(lambda) advantage.
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
            (self.buffer_size, self.n_envs), dtype=th.bool, device=self.device
        )
        self.log_probs = th.zeros(
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
        self.log_probs[self.pos] = log_prob
        self.agent_ids[self.pos] = agent_ids
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns(
        self,
        last_values: Dict[int, th.Tensor],
        dones: th.Tensor,
        policy_sharing: bool = False,
    ) -> None:
        """
        Post-processing step: compute the returns.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        self.returns = self.rewards
        for step in reversed(range(self.buffer_size)):
            if step >= self.buffer_size - len(last_values):
                next_non_terminal = th.logical_not(dones)
            else:
                step_size_to_data = self._find_step_size_to_agent_data(step)
                next_non_terminal = th.logical_not(
                    self.episode_starts[step + step_size_to_data]
                )
                self.returns[step][next_non_terminal] += (
                    self.gamma
                    * self.returns[step + step_size_to_data][next_non_terminal]
                )

    def _get_samples(
        self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None
    ) -> SimpleRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.log_probs[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return SimpleRolloutBufferSamples(*tuple(data))

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = th.randperm(self.buffer_size * self.n_envs, device=self.device)

        # Prepare the data
        if not self.generator_ready:

            _tensor_names = ["observations", "actions", "log_probs", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


class VecRolloutBuffer(VecBaseBuffer):
    """
    Extends Stable Baselines 3 RolloutBuffer to vectorized learning.

    Uses the lambda-return and GAE(lambda) advantage.
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
            (self.buffer_size, self.n_envs), dtype=th.bool, device=self.device
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
        agent_id = 0

        last_values = {
            agent_idx: sa_last_values.clone()
            for agent_idx, sa_last_values in last_values.items()
        }
        last_gae_lam = {agent_idx: 0 for agent_idx in last_values.keys()}

        for step in reversed(range(self.buffer_size)):
            if policy_sharing:
                agent_id = self.agent_ids[step].detach().item()
            sa_last_values = last_values[agent_id]
            if step >= self.buffer_size - len(last_values):
                next_non_terminal = th.logical_not(dones)
                next_values = sa_last_values
            else:
                step_size_to_data = self._find_step_size_to_agent_data(step)
                next_non_terminal = th.logical_not(
                    self.episode_starts[step + step_size_to_data]
                )
                next_values = self.values[step + step_size_to_data]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam[agent_id] = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam[agent_id]
            )
            self.advantages[step] = last_gae_lam[agent_id]
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def _get_samples(
        self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None
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
