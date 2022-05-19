"""Multi agent learning for SB3"""
import time
from typing import Dict, List

import torch
from stable_baselines3.common.type_aliases import MaybeCallback
from torch.utils.tensorboard import SummaryWriter

import src.utils_folder.logging_utils as log_ut
import src.utils_folder.policy_utils as pl_ut


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, config: Dict, env):
        self.config = config
        self.env = env
        self.policy_sharing = config["policy_sharing"]

        self.learners = pl_ut.get_policies(config, env)
        self.writer = SummaryWriter(log_dir=config["experiment_log_path"])

        # TODO: @Nils: still needed? If used for sequential auctions only, move into env
        # check for possible symmetries
        """if self.policy_sharing:
            self.num_learners = len(set(self.env.model.policy_symmetries))
        else:
            self.num_learners = self.env.model.num_agents

        #for learner in self.learners:
         #   assert isinstance(learner, VecPPO), "Learners must be of appropriate type."

        # update strategy pointers
        def _get_strategy(player_position: int):
            policy_position = self.env.model.policy_symmetries[player_position]
            learner = self.learners[policy_position]
            return lambda obs, deterministic=False: learner.policy.forward(
                obs, deterministic=deterministic
            )[0]

        self.env.model.strategies = [
            _get_strategy(player_position)
            for player_position in range(self.env.model.num_agents)
        ]"""

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
            )  # TODO: done for every single agent

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

    def update_ma_external_state_after_step(self, new_obs, dones):
        for _, learner in self.learners.items():
            learner.update_internal_state_after_step(new_obs, dones)

    def update_learner_timesteps(self):
        for learner in self.learners.values():
            learner.num_timesteps += self.env.num_envs

    def postprocess_ma_rollout(self, new_obs, dones):
        for agent_id, learner in self.learners.items():
            learner.postprocess_rollout(new_obs[agent_id], dones)

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
        for _, learner in self.learners.items():
            learner._update_info_buffer(infos, dones)

    def prepare_ma_step(self, n_steps):
        for learner in self.learners.values():
            learner.prepare_step(n_steps, self.env)

    def prepare_ma_rollout(self, callbacks):
        for learner, callback in zip(self.learners.values(), callbacks):
            learner.prepare_rollout(self.env, callback)

    def _update_remaining_progress(self, total_timesteps):
        for learner in self.learners.values():
            learner._update_current_progress_remaining(
                learner.num_timesteps, total_timesteps
            )

    def _display_and_log_training_progress(self, iteration, log_interval):
        # Display training infos
        # TODO: Check if policy sharing needs to be handled differently!
        if log_interval is not None and iteration % log_interval == 0:
            for agent_id, learner in self.learners.items():
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
                            torch.stack(
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
                            torch.stack(
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

    def train_policies(self):
        # TODO: check if policy sharing needs to be handled differently - takes longer than independent policies right now!
        for learner in self.learners.values():
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
