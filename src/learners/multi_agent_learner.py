"""Multi agent learning for SB3"""
import time
from typing import Dict, List

import torch
from stable_baselines3.common.type_aliases import MaybeCallback
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

import src.utils_folder.logging_utils as log_ut
import src.utils_folder.policy_utils as pl_ut
from src.learners.utils import tensor_norm


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, config: Dict, env):
        self.config = config
        self.env = env

        self.learners = pl_ut.get_policies(config, env)
        self.writer = SummaryWriter(log_dir=config["experiment_log_path"])

        # Keep track of running length as Mertikopoulos suggests to detect
        # oscillations
        self.running_length = [0] * len(self.learners)
        self.current_parameters = {
            k: parameters_to_vector([_ for _ in l.policy.parameters()])
            for k, l in self.learners.items()
        }

    def get_ma_action(self):
        actions_for_env = {}
        actions = {}
        additional_actions_data = {}
        for agent_id, learner in self.learners.items():
            sa_actions_for_env, sa_actions, sa_additional_actions_data = learner.get_actions_with_data(
                agent_id
            )
            actions_for_env[agent_id] = sa_actions_for_env
            actions[agent_id] = sa_actions
            additional_actions_data[agent_id] = sa_additional_actions_data
        return actions_for_env, actions, additional_actions_data

    def prepare_ma_actions_for_buffer(self, actions: Dict[int, torch.Tensor]):
        adapted_actions_dict = {}
        for (agent_id, sa_actions), learner in zip(
            actions.items(), self.learners.values()
        ):
            adapted_actions_dict[agent_id] = learner.prepare_actions_for_buffer(
                sa_actions
            )
        return adapted_actions_dict

    def collect_ma_rollouts(self, callbacks, n_rollout_steps):
        """
        Collect experiences using the current policies and fill each agents ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callbacks: Callbacks that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffers: Buffers to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        n_steps = 0
        self.prepare_ma_rollout(callbacks)

        while n_steps < n_rollout_steps:
            self.prepare_ma_step(n_steps)

            actions_for_env, actions, additional_actions_data = self.get_ma_action()

            new_obs, rewards, dones, infos = self.env.step(
                actions_for_env
            )  # TODO: dones for every single agent

            self.update_learner_timesteps()

            # Give access to local variables
            for callback in callbacks:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

            self.update_ma_info_buffer(infos, dones)

            n_steps += 1

            actions = self.prepare_ma_actions_for_buffer(actions)

            rewards = self.handle_ma_dones(rewards, dones, infos)

            self.add_ma_data_to_replay_buffers(
                actions, additional_actions_data, rewards
            )

            self.update_ma_external_state_after_step(new_obs, dones)

        self.postprocess_ma_rollout(new_obs, dones)

        for callback in callbacks:
            callback.on_rollout_end()

        return True

    def _do_break_for_policy_sharing(self, agent_id: int):
        return self.config["policy_sharing"] and agent_id > 0

    def update_ma_external_state_after_step(self, new_obs, dones):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner.update_internal_state_after_step(new_obs, dones)

    def update_learner_timesteps(self):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner.num_timesteps += self.env.num_envs

    def postprocess_ma_rollout(self, new_obs, dones):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            sa_new_obs = new_obs[agent_id]
            postprocess_dones = dones
            if self.config["policy_sharing"]:
                sa_new_obs = new_obs
            learner.postprocess_rollout(
                sa_new_obs, postprocess_dones, self.config["policy_sharing"]
            )

    def add_ma_data_to_replay_buffers(self, actions, additional_actions_data, rewards):
        for agent_id, learner in self.learners.items():
            learner.add_data_to_replay_buffer(
                actions[agent_id],
                rewards[agent_id],
                additional_actions_data[agent_id],
                agent_id,
            )

    def handle_ma_dones(self, rewards, dones, infos):
        for agent_id, learner in self.learners.items():
            rewards[agent_id] = learner.handle_dones(
                dones, infos, rewards[agent_id], agent_id
            )
        return rewards

    def update_ma_info_buffer(self, infos, dones):
        # TODO check if we need this; the info coming out of VecEnv is not agent specific
        # change for vectorized capability: ignore infos
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner._update_info_buffer(infos, dones)

    def prepare_ma_step(self, n_steps):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner.prepare_step(n_steps, self.env)

    def prepare_ma_rollout(self, callbacks):
        for (agent_id, learner), callback in zip(self.learners.items(), callbacks):
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner.prepare_rollout(self.env, callback)

    def _update_remaining_progress(self, total_timesteps):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner._update_current_progress_remaining(
                learner.num_timesteps, total_timesteps
            )

    def _display_and_log_training_progress(self, iteration, log_interval):
        if log_interval is not None and iteration % log_interval == 0:
            for agent_id, learner in self.learners.items():
                if self._do_break_for_policy_sharing(agent_id):
                    break
                fps = int(
                    (learner.num_timesteps - learner._num_timesteps_at_start)
                    / (time.time() - learner.start_time)
                )
                if (
                    len(learner.ep_info_buffer) > 0
                    and len(learner.ep_info_buffer[0]) > 0
                ):
                    learner.logger.record(
                        "rollout/ep_rew_mean",
                        torch.mean(
                            torch.concat(
                                [
                                    ep_info[agent_id]["sa_episode_returns"]
                                    for ep_info in learner.ep_info_buffer
                                ]
                            )
                        )
                        .detach()
                        .item(),
                    )
                    learner.logger.record(
                        "rollout/ep_len_mean",
                        torch.mean(
                            torch.concat(
                                [
                                    ep_info[agent_id]["sa_episode_lengths"]
                                    for ep_info in learner.ep_info_buffer
                                ]
                            )
                        )
                        .detach()
                        .item(),
                    )
                learner.logger.record("time/fps", fps)
                learner.logger.record(
                    "time/time_elapsed", int(time.time() - learner.start_time)
                )
                learner.logger.record("time/total_timesteps", learner.num_timesteps)
                learner.logger.dump(step=learner.num_timesteps)

    def _evaluate_policies(
        self, iteration: int, eval_freq: int, n_eval_episodes: int, callbacks: None
    ) -> None:
        if (iteration + 1) % eval_freq == 0:
            log_ut.evaluate_policies(
                self.learners,
                self.env,
                callbacks=callbacks,
                device=self.config["device"],
                n_eval_episodes=n_eval_episodes,
            )
            self.env.model.custom_evaluation(
                self.learners, self.env, self.writer, iteration + 1, self.config
            )

        # Log running length
        prev_parameters = self.current_parameters
        self.current_parameters = {
            k: parameters_to_vector([_ for _ in l.policy.parameters()])
            for k, l in self.learners.items()
        }
        for i, learner in self.learners.items():
            self.running_length[i] += tensor_norm(
                self.current_parameters[i], prev_parameters[i]
            )
            learner.logger.record("train/running_length", self.running_length[i])

    def train_policies(self):
        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                break
            learner.train()

    def learn(
        self,
        total_timesteps: int,
        n_rollout_steps: int,
        callbacks: List[MaybeCallback] or MaybeCallback = None,
        log_interval: int = 1,
        eval_freq: int = 20,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MultiAgent",
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        """Adapted from Stable-Baselines 3 `OnPolicyAlgorithm`"""

        iteration = 0

        total_timesteps, callbacks = self._setup_learners_for_training(
            total_timesteps,
            callbacks,
            eval_freq,
            n_eval_episodes,
            tb_log_name,
            reset_num_timesteps,
        )

        # Training loop
        while (
            min(learner.num_timesteps for learner in self.learners.values())
            < total_timesteps
        ):
            print(f"Iteration {iteration} starts.")

            continue_training = self.collect_ma_rollouts(
                callbacks, n_rollout_steps=n_rollout_steps
            )

            if continue_training is False:
                break

            self._update_remaining_progress(total_timesteps)

            self._display_and_log_training_progress(iteration, log_interval)

            self._evaluate_policies(iteration, eval_freq, n_eval_episodes, callbacks)

            self.train_policies()

            iteration += 1

        for callback in callbacks:
            callback.on_training_end()

        return self

    def _setup_learners_for_training(
        self,
        total_timesteps,
        callbacks,
        eval_freq,
        n_eval_episodes,
        tb_log_name,
        reset_num_timesteps,
    ):
        # Each learner needs a callback
        if callbacks is None:
            callbacks = [None] * len(self.learners)

        for agent_id, learner in self.learners.items():
            if self._do_break_for_policy_sharing(agent_id):
                callbacks = [callbacks[0] for i in range(len(callbacks))]
                break
            timesteps, callbacks[agent_id] = learner._setup_learn(
                total_timesteps=total_timesteps,
                eval_env=None,
                callback=callbacks[agent_id],
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                log_path=self.config["experiment_log_path"],
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=tb_log_name,
            )
            callbacks[agent_id].on_training_start(locals(), globals())
            total_timesteps = max(timesteps, total_timesteps)
        return total_timesteps, callbacks
