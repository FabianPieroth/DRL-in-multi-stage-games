"""Multi agent learning for SB3"""
import time
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import torch
from gym import spaces
from stable_baselines3.common.type_aliases import MaybeCallback
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

import src.utils.logging_utils as log_ut
import src.utils.policy_utils as pl_ut
from src.learners.utils import tensor_norm
from src.verifier import BFVerifier


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, config: Dict, env):
        self.config = config
        self.env = env
        self.n_step = 0

        self.learners = pl_ut.get_policies(config, env)
        self.writer = SummaryWriter(log_dir=config["experiment_log_path"])

        # TODO Nils: This is needed, unfortunately, for `collapse_symmetric_opponents`
        self.env.model.learners = self.learners

        # Keep track of running length as Mertikopoulos suggests to detect
        # oscillations
        self.running_length = [0] * len(self.learners)
        self.current_parameters = self._get_policy_parameters(self.learners)

        # Setup verifier
        self.verifier = self._setup_verifier()

    def _get_policy_parameters(self, learners):
        """Collect all current neural network parameters of the policy."""
        param_dict = {}
        for agent_id, learner in learners.items():
            if learner.policy is not None:
                param_dict[agent_id] = parameters_to_vector(
                    [_ for _ in learner.policy.parameters()]
                )
            else:
                param_dict[agent_id] = torch.zeros(1, device=self.config["device"])
        return param_dict

    def _setup_verifier(self):
        """Use appropriate verifier."""
        if all(
            isinstance(s, spaces.Box) for s in self.env.model.action_spaces.values()
        ):
            verifier = BFVerifier(
                num_simulations=self.config["verifier"]["num_simulations"],
                obs_discretization=self.config["verifier"]["obs_discretization"],
                action_discretization=self.config["verifier"]["action_discretization"],
                env=self.env,
            )
        elif all(
            isinstance(s, spaces.Discrete)
            for s in self.env.model.action_spaces.values()
        ):
            verifier = DiscreteVerifier(env=self.env)
        else:
            raise NotImplementedError("Could not infer verifier for this game.")
        return verifier

    def get_ma_action(self, obs: Dict[int, torch.Tensor]):
        actions_for_env = {}
        actions = {}
        additional_actions_data = {}
        for agent_id, learner in self.learners.items():
            sa_actions_for_env, sa_actions, sa_additional_actions_data = learner.get_actions_with_data(
                obs[agent_id]
            )
            actions_for_env[agent_id] = sa_actions_for_env
            actions[agent_id] = sa_actions
            additional_actions_data[agent_id] = sa_additional_actions_data
        return actions_for_env, actions, additional_actions_data

    def ingest_ma_data_into_learners(
        self,
        last_obs,
        last_episode_starts,
        actions,
        rewards,
        additional_actions_data,
        dones,
        infos,
        new_obs,
        policy_sharing,
        callbacks,
    ):
        """All required information is shared with the individual learners."""
        for agent_id, learner in self.learners.items():
            learner.ingest_data_to_learner(
                last_obs[agent_id],
                last_episode_starts,
                actions[agent_id],
                rewards[agent_id],
                additional_actions_data[agent_id],
                dones,
                infos,
                new_obs,
                agent_id,
                policy_sharing,
                callbacks[agent_id],
            )

    def _do_break_for_policy_sharing(self, agent_id: int):
        """If all agents share the same policy, we only train the one at 
        index 0.
        """
        return self.config["policy_sharing"] and agent_id > 0

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
        """Evaluate current training progress."""
        if iteration == 0 or (iteration + 1) % eval_freq == 0:
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
        self._log_change_in_parameter_space()

    def _verify_policies(self, iteration: int, eval_freq: int) -> None:
        if (
            (iteration == 0 or (iteration + 1) % eval_freq == 0)
            and self.verifier.env_is_compatible_with_verifier
            and self.config["verify_br"]
        ):
            utility_loss, best_responses = self.verifier.verify(self.learners)
            log_ut.log_data_dict_to_learner_loggers(
                self.learners, utility_loss, "eval/utility_loss"
            )
            br_plot = self.env.model.plot_br_strategy(best_responses)
            log_ut.log_figure_to_writer(
                self.writer, br_plot, iteration, "estimated_br_strategies"
            )
            plt.savefig(f"{self.writer.log_dir}/br_plot_{iteration}.png")

    def verify_in_BNE(self) -> None:
        if not self.verifier.env_is_compatible_with_verifier:
            return None

        num_agents = self.env.model.num_agents
        strategies = {agent_id: None for agent_id in range(num_agents)}
        utility_losses = {agent_id: 0 for agent_id in range(num_agents)}

        for agent_id in range(num_agents):
            for opp_id in range(num_agents):
                if opp_id == agent_id:
                    # TODO: should not be needed
                    strategies[opp_id] = self.learners[opp_id]
                else:
                    # Create a wrapper such that the BNE strategies provide
                    # the same interface as the learners
                    class EquilibriumStrategy:
                        @staticmethod
                        def predict(
                            observations, states, episode_start, deterministic=True
                        ):
                            strategy = self.env.model.get_ma_equilibrium_actions
                            return strategy({opp_id: observations})[opp_id], None

                    strategies[opp_id] = EquilibriumStrategy()

            utility_loss, best_response = self.verifier.verify(
                strategies, agent_ids=[agent_id]
            )
            utility_losses[agent_id] = utility_loss[agent_id]

            # Debug plotting
            self.env.model.plot_br_strategy(best_response)
            plt.savefig(f"./logs/{self.env.model}_{agent_id}_br.png")

        return utility_losses

    def _log_change_in_parameter_space(self):
        prev_parameters = self.current_parameters
        self.current_parameters = self._get_policy_parameters(self.learners)
        for i, learner in self.learners.items():
            self.running_length[i] += tensor_norm(
                self.current_parameters[i], prev_parameters[i]
            )
            learner.logger.record("train/running_length", self.running_length[i])

    def learn(
        self,
        total_timesteps: int,
        n_steps_per_iteration: int,
        callbacks: List[MaybeCallback] or MaybeCallback = None,
        log_interval: int = 1,
        eval_freq: int = 20,
        n_eval_episodes: int = 5,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        """Main training loop for multi-agent learning: Here, (1) the agents
        are asked to submit actions, (2) these are passed to the environment,
        and (3) the utilities are passed back to the agents for learning.        
        
        Adapted from Stable-Baselines 3 `OnPolicyAlgorithm`.
        """
        iteration = 0

        total_timesteps, callbacks = self._setup_learners_for_training(
            total_timesteps, callbacks, eval_freq, n_eval_episodes, reset_num_timesteps
        )

        last_obs = self.env.reset()
        last_episode_starts = torch.ones(
            (self.env.num_envs,), dtype=bool, device=self.env.device
        )

        while (
            min(learner.num_timesteps for learner in self.learners.values())
            < total_timesteps
        ):
            if self._iteration_finished(n_steps_per_iteration):
                print(f"Iteration {iteration} starts.")

                # Evaluate & log
                with torch.no_grad():  # TODO: Is this necessary? Should we call this here?
                    self._display_and_log_training_progress(iteration, log_interval)
                    self._evaluate_policies(
                        iteration, eval_freq, n_eval_episodes, callbacks
                    )
                    self._verify_policies(iteration, eval_freq)

            actions_for_env, actions, additional_actions_data = self.get_ma_action(
                last_obs
            )

            new_obs, rewards, dones, infos = self.env.step(
                actions_for_env
            )  # TODO: dones for every single agent

            self.n_step += 1

            self.ingest_ma_data_into_learners(
                last_obs,
                last_episode_starts,
                actions,
                rewards,
                additional_actions_data,
                dones,
                infos,
                new_obs,
                self.config["policy_sharing"],
                callbacks,
            )

            last_obs = {k: v.detach().clone() for k, v in new_obs.items()}
            last_episode_starts = dones.detach().clone()
            # Give access to local variables
            for callback in callbacks:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

            if self._iteration_finished(n_steps_per_iteration):
                iteration += 1

        for callback in callbacks:
            callback.on_training_end()

        return self

    def _iteration_finished(self, n_steps_per_iteration: int):
        return self.n_step % n_steps_per_iteration == 0

    def _setup_learners_for_training(
        self,
        total_timesteps,
        callbacks,
        eval_freq,
        n_eval_episodes,
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
            )
            callbacks[agent_id].on_training_start(locals(), globals())
            total_timesteps = max(timesteps, total_timesteps)
        return total_timesteps, callbacks
