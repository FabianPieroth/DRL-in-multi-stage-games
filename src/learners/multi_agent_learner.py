"""Multi agent learning for SB3"""
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import src.utils.evaluation_utils as ev_ut
import src.utils.io_utils as io_ut
import src.utils.logging_write_utils as log_ut
import src.utils.policy_utils as pl_ut
from src.learners.policies.register_policies import register_policies
from src.verifier import BFVerifier


class MultiAgentCoordinator:
    """Coordinates simultaneous learning of multiple learners."""

    def __init__(self, config: Dict, env):
        # Register all custom policies
        register_policies()

        self.config = config
        self.env = env
        self.n_step = 0

        self.learners = pl_ut.get_policies(config, env)
        self.writer = SummaryWriter(log_dir=config.experiment_log_path)

        # TODO Nils: This is needed, unfortunately, for `collapse_symmetric_opponents`
        self.env.model.learners = self.learners

        # Keep track of running length as Mertikopoulos suggests to detect
        # oscillations
        self.running_length = [0] * len(self.learners)
        self.current_parameters = log_ut.get_policy_parameters(self.learners)

        # Setup verifier
        self.verifier = self._setup_verifier()

    def _setup_verifier(self):
        """Use appropriate verifier."""
        verifier = BFVerifier(
            num_simulations=self.config.verifier.num_simulations,
            obs_discretization=self.config.verifier.obs_discretization,
            action_discretization=self.config.verifier.action_discretization,
            env=self.env,
            batch_size=self.config.verifier.batch_size,
            device=self.config.device
            if self.config.verifier.device is None
            else self.config.verifier.device,
        )
        return verifier

    def get_ma_action(self, obs: Dict[int, torch.Tensor]):
        actions_for_env = {}
        actions = {}
        additional_data = {}
        for agent_id, learner in self.learners.items():
            (
                sa_actions_for_env,
                sa_actions,
                sa_additional_data,
            ) = learner.get_actions_with_data(obs[agent_id])
            actions_for_env[agent_id] = sa_actions_for_env
            actions[agent_id] = sa_actions
            additional_data[agent_id] = sa_additional_data
        return actions_for_env, actions, additional_data

    def ingest_ma_data_into_learners(
        self,
        last_obs,
        last_episode_starts,
        actions,
        rewards,
        additional_data,
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
                additional_data[agent_id],
                dones,
                infos,
                new_obs,
                agent_id,
                policy_sharing,
                callbacks[agent_id],
            )

    def break_for_policy_sharing(self, agent_id: int):
        """If all agents share the same policy, we only train and log with the one at
        index 0.
        """
        return self.config.policy_sharing and agent_id > 0

    def verify_policies_br(self, iteration: int) -> None:
        """Use our verifier to evaluate each learned strategy against the
        current opponents. The verifier estimates the best response of each agent
        against the opponent strategies to estimate the agent utility loss.
        Estimates:
            sup_{beta_i \in \Sigma_i} \hat(u)_i(beta_i, beta_{-i})
            by approximating the best response beta_i^* in the space of step functions.
        """
        if self.verifier.env_is_verifiable and self.config.verify_br:
            # Estimate the ex ante utility loss (i.e., averaged loss over prior)
            print(f"Verifying against current strategies:")
            agent_ids = self.learners.keys()
            if self.config.policy_sharing:
                agent_ids = [0]
            (
                estimated_actual_utilities,
                estimated_utility_loss,
                estimated_relative_utility_loss,
                best_responses,
            ) = self.verifier.verify_br(self.learners, agent_ids=agent_ids)

            # Logging
            log_ut.log_data_dict_to_learner_loggers(
                self.learners,
                estimated_actual_utilities,
                "eval/estimated_actual_utilities",
            )
            log_ut.log_data_dict_to_learner_loggers(
                self.learners, estimated_utility_loss, "eval/estimated_utility_loss"
            )
            log_ut.log_data_dict_to_learner_loggers(
                self.learners,
                estimated_relative_utility_loss,
                "eval/estimated_relative_utility_loss",
            )

            # Plotting
            br_plot = self.env.model.plot_br_strategy(best_responses)
            if br_plot is not None:
                log_ut.log_figure_to_writer(
                    self.writer, br_plot, iteration + 1, "estimated_br_strategies"
                )
            plt.close()

    def verify_policies_against_known_equilibrium(self) -> None:
        """If an equilibrium is available, we can evaluate learned strategies
        using the known equilibrium strategies.
        """
        if not self.verifier.equ_metrics_can_be_measured:
            return
        print(f"\nVerifying against known equilibrium strategies:")
        self.estimate_utility_loss_against_equilibrium()

        ev_ut.log_l2_distance_to_equilibrium(
            env=self.env.model,
            learners=self.learners,
            equ_strategies=self.env.model.equilibrium_strategies,
            num_envs=self.config.verifier.num_simulations,
            num_stages=self.env.model.num_stages,
        )

    def estimate_utility_loss_against_equilibrium(self):
        """Approximate the utility loss if one agent deviates to equilibrium.
        Estimate:
            \hat(u)_i(beta_i^*, beta_{-i}^*) - \hat(u)_i(beta_i, beta_{-i}^*)
            where \beta^*=(\beta_i^*, \beta_{-i}^*) is an equilibrium strategy
            and \beta=(\beta_i, \beta_{-i}) is the learned strategy.
        """
        agent_ids = self.learners.keys()
        if self.config.policy_sharing:
            agent_ids = [0]

        (
            equilibrium_utilities,
            utility_losses,
            relative_utility_losses,
        ) = self.verifier.verify_against_equilibrium(self.learners, agent_ids=agent_ids)

        # Logging
        log_ut.log_data_dict_to_learner_loggers(
            self.learners, equilibrium_utilities, "eval/estimated_utility_in_equ"
        )
        log_ut.log_data_dict_to_learner_loggers(
            self.learners, utility_losses, "eval/utility_loss"
        )
        log_ut.log_data_dict_to_learner_loggers(
            self.learners, relative_utility_losses, "eval/relative_utility_loss"
        )

    def verify_br_against_BNE(self) -> float:
        """Approximate the best responses in equilibrium using our verifier. If
        everything works out, the loss should be zero in expectation for all
        players.
        Estimate:
            sup_{beta_i \in \Sigma_i} \hat(u)_i(beta_i, beta_{-i}^*)
            by approximating the best response beta_i^* in the space of step functions
            and \beta^*=(\beta_i^*, \beta_{-i}^*) is a equilibrium strategy.
        """
        assert (
            self.verifier.env_is_verifiable
            and self.env.model.equilibrium_strategies_known
        )
        agent_ids = self.learners.keys()
        if self.config.policy_sharing:
            agent_ids = [0]
        _, utility_losses, _, _ = self.verifier.verify_br(
            self.env.model.equilibrium_strategies, agent_ids=agent_ids
        )
        return utility_losses

    def learn(self) -> "OnPolicyAlgorithm":
        """Main training loop for multi-agent learning: Here, (1) the agents
        are asked to submit actions, (2) these are passed to the environment,
        and (3) the utilities are passed back to the agents for learning.

        Adapted from Stable-Baselines 3 `OnPolicyAlgorithm`.
        """
        iteration = 0
        init_callbacks = None
        reset_num_timesteps = True  # NOTE: Needed to implement resuming training

        total_timesteps, callbacks = self._setup_learners_for_training(
            self.config.total_training_steps,
            init_callbacks,
            self.config.eval_freq,
            self.config.n_eval_episodes,
            reset_num_timesteps,
        )

        last_obs = self.env.reset()
        last_episode_starts = torch.ones(
            (self.env.num_envs,), dtype=bool, device=self.env.device
        )

        while (
            min(learner.num_timesteps for learner in self.learners.values())
            < total_timesteps
        ):
            if self._iteration_finished(self.config.n_steps_per_iteration):
                print(f"Iteration {iteration} starts.")
                self._log(iteration)

            actions_for_env, actions, additional_data = self.get_ma_action(last_obs)

            new_obs, rewards, dones, infos = self.env.step(actions_for_env)

            self.n_step += 1

            self.ingest_ma_data_into_learners(
                last_obs,
                last_episode_starts,
                actions,
                rewards,
                additional_data,
                dones,
                infos,
                new_obs,
                self.config.policy_sharing,
                callbacks,
            )

            last_obs = {k: v.detach().clone() for k, v in new_obs.items()}
            last_episode_starts = dones.detach().clone()
            # Give access to local variables
            for callback in callbacks:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

            if self._iteration_finished(self.config.n_steps_per_iteration):
                iteration += 1

        for callback in callbacks:
            callback.on_training_end()

        io_ut.wrap_up_learning_logging(self.config)
        return self

    def _log(self, iteration: int):
        """Evaluate & log"""
        # 1. Train log
        if (
            iteration == 0
            or self.config.train_log_freq is not None
            and (iteration + 1) % self.config.train_log_freq == 0
        ):
            log_ut.log_training_progress(
                self.learners, iteration, self.break_for_policy_sharing
            )

        # 2. Eval log
        if (iteration + 1) % self.config.eval_freq == 0:
            print("\nStart evaluation:")
            with torch.no_grad():
                self._evaluate_policies(iteration, self.config.n_eval_episodes)
                self.verify_policies_br(iteration)
                self.verify_policies_against_known_equilibrium()
            self._store_learner(iteration)
            print("\nEnd evaluation.\n")

        # 3. Update `num_timesteps` in logs
        for learner in self.learners.values():
            learner.logger.dump(step=learner.num_timesteps)

    def _store_learner(self, iteration):
        if self.config.store_learner:
            for agent_id, learner in self.learners.items():
                try:
                    learner.save(path=learner.logger.dir + "_iter_" + str(iteration))
                    print("Stored learner for agent " + str(agent_id))
                except AttributeError:
                    warnings.warn(
                        "Could not store learner for agent " + str(agent_id) + "!"
                    )

    def _evaluate_policies(self, iteration: int, n_eval_episodes: int) -> None:
        """Evaluate current training progress."""
        log_ut.evaluate_policies(
            self.learners,
            self.env,
            device=self.config.device,
            n_eval_episodes=n_eval_episodes,
        )
        self.current_parameters, self.running_length = log_ut.change_in_parameter_space(
            self.learners,
            self.current_parameters,
            self.running_length,
            self.break_for_policy_sharing,
        )
        self.env.model.custom_evaluation(
            self.learners, self.env, self.writer, iteration + 1, self.config
        )

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
            if self.break_for_policy_sharing(agent_id):
                callbacks = [callbacks[0] for i in range(len(callbacks))]
                break
            timesteps, callbacks[agent_id] = learner._setup_learn(
                total_timesteps=total_timesteps,
                eval_env=None,
                callback=callbacks[agent_id],
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                log_path=self.config.experiment_log_path,
                reset_num_timesteps=reset_num_timesteps,
            )
            self._load_learner(agent_id, learner)
            callbacks[agent_id].on_training_start(locals(), globals())
            total_timesteps = max(timesteps, total_timesteps)
        return total_timesteps, callbacks

    def _load_learner(self, agent_id, learner):
        if self.config.load_learner and agent_id < len(
            self.config.paths_to_stored_learners
        ):
            path_to_load = self.config.paths_to_stored_learners[agent_id]
            try:
                learner.set_parameters(
                    load_path_or_dict=path_to_load, device=learner.device
                )
                print(
                    "Loaded learner for agent "
                    + str(agent_id)
                    + " from "
                    + path_to_load
                )
            except (AttributeError, FileNotFoundError):
                warnings.warn(
                    "Could not load learner for agent "
                    + str(agent_id)
                    + " from "
                    + path_to_load
                    + "!"
                )
