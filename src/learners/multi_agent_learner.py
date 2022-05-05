"""Multi agent learning for SB3"""
import time
from typing import List, Optional

import torch
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

from src.envs.rock_paper_scissors import eval_rps_strategy
from src.learners.ppo import VecPPO
from src.learners.utils import tensor_norm


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(
        self,
        env,
        learner_class: VecPPO,
        learner_kwargs: dict = None,
        policy_sharing: bool = True,
        backward_induction_learning: bool = False,
    ):
        self.env = env
        self.policy_sharing = policy_sharing

        # TODO: WIP
        self.backward_induction_learning = backward_induction_learning

        # check for possible symmetries
        if self.policy_sharing:
            self.num_learners = len(set(self.env.model.policy_symmetries))
        else:
            self.num_learners = self.env.model.num_agents

        self.learners = [
            learner_class(env=self.env, **learner_kwargs)
            for _ in range(self.num_learners)
        ]

        for learner in self.learners:
            assert isinstance(learner, VecPPO), "Learners must be of appropriate type."

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
        ]

        self.writer = SummaryWriter(log_dir=self.learners[0].tensorboard_log)

        # Keep track of running length as Mertikopoulos suggests to detect
        # oscillations
        self.running_length = [0] * self.num_learners
        self.current_parameters = [
            parameters_to_vector([_ for _ in l.policy.parameters()])
            for l in self.learners
        ]

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

        # Each learner needs a callback
        if callbacks is None:
            callbacks = [None] * len(self.learners)

        iteration = 0

        for player_position, learner in enumerate(self.learners):
            timesteps, callbacks[player_position] = learner._setup_learn(
                total_timesteps,
                None,
                callbacks[player_position],
                eval_freq,
                n_eval_episodes,
                None,
                reset_num_timesteps,
                tb_log_name,
            )
            callbacks[player_position].on_training_start(locals(), globals())
            total_timesteps = max(
                timesteps, total_timesteps
            )  # TODO: is that what we want here?

        # # Testing: Dummy non-learner
        # def dumb(obs):
        #     actions = torch.zeros((obs.shape[0], self.learners[1].env.model.action_space_size), device=obs.device)
        #     actions[obs[:, 0] < 5] = 0  # rock
        #     actions[obs[:, 0] >= 5] = 1  #
        #     return actions
        # self.env.strategies[1] = dumb

        # Training loop
        while min(learner.num_timesteps for learner in self.learners) < total_timesteps:
            print(f"Iteration {iteration} starts.")

            for player_position, (learner, callback) in enumerate(
                zip(self.learners, callbacks)
            ):

                # # Testing: Dummy non-learner
                # if player_position == 1:
                #     continue

                # Set actively learning player
                learner.env.model.player_position = player_position
                # TODO should be equivalent to self.env.player_position

                # Backward induction learning
                if self.backward_induction_learning:
                    sub_n = total_timesteps // learner.env.model.num_rounds_to_play
                    stage = (
                        learner.env.model.num_rounds_to_play
                        - 1
                        - learner.num_timesteps // sub_n
                    )
                    # stage = int(torch.randint(stage, learner.env.model.num_rounds_to_play, (1,)))
                    learner.env.model.earliest_stage = stage
                    print("stage", stage)

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

            # Log running length
            prev_parameters = self.current_parameters
            self.current_parameters = [
                parameters_to_vector([_ for _ in learner.policy.parameters()])
                for learner in self.learners
            ]
            for i, learner in enumerate(self.learners):
                self.running_length[i] += tensor_norm(
                    self.current_parameters[i], prev_parameters[i]
                )
                learner.logger.record("train/running_length", self.running_length[i])

            iteration += 1

        for callback in callbacks:
            callback.on_training_end()

        return self
