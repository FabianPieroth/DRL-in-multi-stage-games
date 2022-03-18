"""Multi agent learning for SB3"""
import time
from typing import List, Optional

import torch
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from src.envs.rock_paper_scissors import eval_rps_strategy
from src.learners.ppo import VecPPO


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, learners: List[VecPPO], learner_names: List[str] = None):
        self.learners = learners

        if learner_names is None:
            self.learner_names = [f"learner{i}" for i in range(len(learners))]
        else:
            self.learner_names = learner_names

        for l in learners:
            assert isinstance(l, VecPPO), "Learners must be of appropriate type."

        assert (
            len(set(id(l.env.model) for l in self.learners)) == 1
        ), "All learners must point to the same base env."
        self.env = self.learners[0].env.model

        # Update strategy pointers
        def get_strategy_from_learner(learner):
            return lambda obs: learner.policy.forward(obs)[0]

        self.env.strategies = [
            get_strategy_from_learner(learner) for learner in self.learners
        ]

    def learn(
        self,
        total_timesteps: int,
        callbacks: List[MaybeCallback] or MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MultiAgentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        """Adapted from SB3 `OnPolicyAlgorithm`"""

        # Each learner needs a callback
        if callbacks is None:
            callbacks = [None] * len(self.learners)

        iteration = 0

        for player_position, learner in enumerate(self.learners):
            timesteps, callbacks[player_position] = learner._setup_learn(
                total_timesteps,
                eval_env,
                callbacks[player_position],
                eval_freq,
                n_eval_episodes,
                eval_log_path,
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

            for player_position, (learner, callback) in enumerate(
                zip(self.learners, callbacks)
            ):

                # # Testing: Dummy non-learner
                # if player_position == 1:
                #     continue

                # Set actively learning player
                learner.env.model.player_position = player_position

                continue_training = learner.collect_rollouts(
                    learner.env,
                    callback,
                    learner.rollout_buffer,
                    n_rollout_steps=learner.n_steps,
                )

                if continue_training is False:
                    break

                iteration += 1
                learner._update_current_progress_remaining(
                    learner.num_timesteps, total_timesteps
                )

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    fps = int(
                        (learner.num_timesteps - learner._num_timesteps_at_start)
                        / (time.time() - learner.start_time)
                    )
                    learner.logger.record(
                        "time/iterations", iteration, exclude="tensorboard"
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
                        "time/time_elapsed",
                        int(time.time() - learner.start_time),
                        exclude="tensorboard",
                    )
                    learner.logger.record(
                        "time/total_timesteps",
                        learner.num_timesteps,
                        exclude="tensorboard",
                    )
                    learner.logger.dump(step=learner.num_timesteps)

                learner.train()

                # Manuel evaluation
                eval_strategy = lambda obs: learner.policy(obs)[0]
                eval_rps_strategy(learner.env, player_position, eval_strategy)
                print()

        for callback in callbacks:
            callback.on_training_end()

        return self
