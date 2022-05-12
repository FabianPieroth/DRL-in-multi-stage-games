"""Multi agent learning for SB3"""
import time
from typing import Dict, List, Optional

import torch
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from torch.utils.tensorboard import SummaryWriter

import src.utils_folder.policy_utils as pl_ut
from src.envs.rock_paper_scissors import eval_rps_strategy
from src.learners.ppo import VecPPO


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, config: Dict, env):
        self.config = config
        self.env = env
        self.policy_sharing = config["policy_sharing"]

        self.learners = pl_ut.get_policies(config, env)
        # TODO: make this adaptable to different policy types
        self.writer = SummaryWriter(log_dir=self.learners[0].tensorboard_log)

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
        pass

    def prepare_ma_actions_for_buffer(self):
        pass

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

            new_obs, rewards, dones, infos = self.env.step(actions_for_env)

            self.num_timesteps += self.env.num_envs

            # Give access to local variables
            for callback in callbacks:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

            self.update_ma_info_buffer(infos)

            n_steps += 1

            actions = self.prepare_ma_actions_for_buffer()

            rewards = self.handle_ma_dones(rewards, dones, infos)

            self.add_ma_data_to_replay_buffers(
                actions, additional_actions_data, rewards
            )

            for agent_id, learner in self.learners.items():
                learner.update_internal_state_after_step(
                    new_obs[agent_id], dones[agent_id]
                )

        self.postprocess_ma_rollout(new_obs, dones)

        for callback in callbacks:
            callback.on_rollout_end()

        return True

    def postprocess_ma_rollout(self, new_obs, dones):
        for agent_id, learner in self.learners.items():
            learner.postprocess_rollout(new_obs[agent_id], dones[agent_id])

    def add_ma_data_to_replay_buffers(self, actions, additional_actions_data, rewards):
        for agent_id, learner in self.learners.items():
            learner.add_data_to_replay_buffer(
                actions[agent_id], rewards[agent_id], additional_actions_data[agent_id]
            )

    def handle_ma_dones(self, rewards, dones, infos):
        for agent_id, learner in self.learners.items():
            rewards[agent_id] = learner.handle_dones(
                dones[agent_id], infos[agent_id], rewards[agent_id]
            )
        return rewards

    def update_ma_info_buffer(self, infos):
        # TODO check if we need this
        # change for vectorized capability: ignore infos
        for agent_id, learner in self.learners.items():
            learner._update_info_buffer(infos[agent_id])

    def prepare_ma_step(self, n_steps):
        for learner in self.learners.values():
            learner.prepare_step(n_steps, self.env)

    def prepare_ma_rollout(self, callbacks):
        for learner, callback in zip(self.learners.values(), callbacks):
            learner.prepare_rollout(self.env, callback)

    def _get_n_rollout_steps_from_learners(self) -> int:
        print("give a warning if this is not identical!")
        return 512

    def _update_remaining_progress(self, total_timesteps):
        for learner in self.learners.values():
            learner._update_current_progress_remaining(
                learner.num_timesteps, total_timesteps
            )

    def _display_and_log_training_progress(self, iteration, log_interval, eval_freq):
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
                        safe_mean([ep_info["r"] for ep_info in learner.ep_info_buffer]),
                    )
                    learner.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in learner.ep_info_buffer]),
                    )
                learner.logger.record("time/fps", fps)
                learner.logger.record(
                    "time/time_elapsed", int(time.time() - learner.start_time)
                )
                learner.logger.record("time/total_timesteps", learner.num_timesteps)
                learner.logger.dump(step=learner.num_timesteps)

            # Custom evaluation
            if iteration % eval_freq == 0:
                self.env.model.log_plotting(writer=self.writer, step=iteration)
                self.env.model.log_vs_bne(logger=learner.logger)
                # # RPS eval
                # eval_strategy = lambda obs: learner.policy(obs)[0]
                # eval_rps_strategy(learner.env, player_position, eval_strategy)

    def train_policies(self):
        # TODO: check if policy sharing needs to be handled differently
        for learner in self.learners.values():
            learner.train()

    def learn(
        self,
        total_timesteps: int,
        callbacks: List[MaybeCallback] or MaybeCallback = None,
        log_interval: int = 1,
        eval_freq: int = 20,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MultiAgentPPO",
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

        n_steps = self._get_n_rollout_steps_from_learners()

        # Training loop
        while (
            min(learner.num_timesteps for learner in self.learners.values())
            < total_timesteps
        ):
            print(f"Iteration {iteration} starts.")

            continue_training = self.collect_ma_rollouts(
                callbacks, n_rollout_steps=n_steps
            )

            if continue_training is False:
                break

            self._update_remaining_progress(total_timesteps)

            self._display_and_log_training_progress()

            self.train_policies()

            iteration += 1

        for callback in callbacks:
            callback.on_training_end()

        return self
        """for (agent_id, learner), callback in zip(self.learners.items(), callbacks):

            # # Testing: Dummy non-learner
            # if player_position == 1:
            #     continue

            # Set actively learning player
            learner.env.model.player_position = agent_id
            # TODO should be equivalent to self.env.player_position

            continue_training = learner.collect_rollouts(
                learner.env,
                callback,
                learner.rollout_buffer,
                n_rollout_steps=learner.n_steps,
            )

            if continue_training is False:
                break

            learner._update_current_progress_remaining(
                learner.num_timesteps, total_timesteps
            )

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
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
                        safe_mean(
                            [ep_info["r"] for ep_info in learner.ep_info_buffer]
                        ),
                    )
                    learner.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean(
                            [ep_info["l"] for ep_info in learner.ep_info_buffer]
                        ),
                    )
                learner.logger.record("time/fps", fps)
                learner.logger.record(
                    "time/time_elapsed", int(time.time() - learner.start_time)
                )
                learner.logger.record("time/total_timesteps", learner.num_timesteps)
                learner.logger.dump(step=learner.num_timesteps)

            # Custom evaluation
            if iteration % eval_freq == 0:
                self.env.model.log_plotting(writer=self.writer, step=iteration)
                self.env.model.log_vs_bne(logger=learner.logger)
                # # RPS eval
                # eval_strategy = lambda obs: learner.policy(obs)[0]
                # eval_rps_strategy(learner.env, player_position, eval_strategy)

            learner.train()

        iteration += 1

    for callback in callbacks:
        callback.on_training_end()

    return self"""

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
                total_timesteps,
                None,
                callbacks[agent_id],
                eval_freq,
                n_eval_episodes,
                None,
                reset_num_timesteps,
                tb_log_name,
            )
            callbacks[agent_id].on_training_start(locals(), globals())
            total_timesteps = max(timesteps, total_timesteps)
            # TODO: is that what we want here?
        return total_timesteps, callbacks
