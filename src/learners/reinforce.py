"""Customized PPO learner and corresponding buffer"""
import time
from collections import deque
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from gym import spaces
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

from src.learners.base_learner import SABaseAlgorithm
from src.learners.rollout_buffer import SimpleVecRolloutBuffer
from src.learners.utils import explained_variance


class Reinforce(SABaseAlgorithm):
    """
    Reinforce algorithm.
    """

    def get_actions_with_data(self, agent_id: int):
        self.prepare_step(self.rollout_buffer.pos, self.env)
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = self._last_obs[agent_id]
            actions, _, log_probs = self.policy.forward(obs_tensor)
            # NOTE: `value` predictions disregarded

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = th.clip(
                actions, self.action_space.low, self.action_space.high
            )
        return clipped_actions, actions, (None, log_probs)

    def handle_dones(self, dones, infos, sa_rewards, agent_id: int):
        """Fake method to use common base class with PPO."""
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

    def postprocess_rollout(self, sa_new_obs, dones, policy_sharing: bool):
        # Placebo values that have the length needed for `compute_returns`
        if policy_sharing:
            values = {agent_id: None for agent_id in sa_new_obs.keys()}
        else:
            values = {0: None}
        self.rollout_buffer.compute_returns(
            last_values=values, dones=dones, policy_sharing=policy_sharing
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = SimpleVecRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # NOTE: We stick to using SB3's `ActorCriticPolicy` for simplicity and
        # just disregard the `value` outputs
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        pg_losses = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                _, log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                # PG loss
                loss = -(rollout_data.returns * log_prob).mean()

                # Logging
                pg_losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
