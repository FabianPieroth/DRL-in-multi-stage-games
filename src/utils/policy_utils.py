from typing import Callable, Dict

import torch.nn as nn
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.simple_soccer import SimpleSoccer
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.gpu_dqn import GPUDQN
from src.learners.ppo import VecPPO
from src.learners.reinforce import Reinforce
from src.learners.rps_dummy_learner import RPSDummyLearner
from src.learners.simple_soccer_policies.block_policy import BlockPolicy
from src.learners.simple_soccer_policies.chase_ball_policy import ChaseBallPolicy
from src.learners.simple_soccer_policies.goal_wall_policy import GoalWallPolicy
from src.learners.simple_soccer_policies.handcrafted_policy import HandcraftedPolicy


def get_policies(config: Dict, env: MATorchVecEnv) -> Dict[int, BaseAlgorithm]:
    set_space_translators_in_env(config, env)
    if config.policy_sharing:
        shared_policy = get_learner_and_policy(0, config, env)
        return {agent_id: shared_policy for agent_id in range(env.model.num_agents)}
    else:
        return {
            agent_id: get_learner_and_policy(agent_id, config, env)
            for agent_id in range(env.model.num_agents)
        }


def set_space_translators_in_env(config: Dict, env: MATorchVecEnv):
    for agent_id in range(env.model.num_agents):
        env.set_space_translators_for_agent(
            agent_id,
            config["algorithm_configs"][get_algo_name(agent_id, config)],
            config["space_translators"],
        )


def get_policy(config_policy):
    return dict(
        net_arch=[OmegaConf.to_container(config_policy.net_arch)],
        activation_fn=eval(str(config_policy.activation_fn)),
        action_activation_fn=eval(str(config_policy.action_activation_fn)),
    )


def get_lr_schedule(lr_schedule_name: str, initial_value: float) -> Callable:
    if lr_schedule_name == "constant":
        return lambda x: initial_value
    elif lr_schedule_name == "linear":
        return lambda progress_remaining: progress_remaining * initial_value
    elif lr_schedule_name == "exponential":
        return lambda progress_remaining: initial_value * (progress_remaining ** 10)
    else:
        raise ValueError(f"Learning rate scheduler {lr_schedule_name} unknown.")


def get_learner_and_policy(
    agent_id: int, config: Dict, env: MATorchVecEnv
) -> BaseAlgorithm:
    algo_name = get_algo_name(agent_id, config)
    env.set_env_for_current_agent(agent_id)
    if algo_name == "ppo":
        algorithm_config = config["algorithm_configs"]["ppo"]
        n_rollout_steps = algorithm_config["n_rollout_steps"]
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return VecPPO(
            policy=algorithm_config["policy"],
            env=env,
            learning_rate=get_lr_schedule(
                algorithm_config["learning_rate_schedule"],
                algorithm_config["learning_rate"],
            ),
            n_steps=n_rollout_steps,
            batch_size=algorithm_config["n_rollout_steps"] * config["num_envs"],
            action_dependent_std=config.policy["action_dependent_std"],
            gamma=algorithm_config["gamma"],
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            policy_kwargs=get_policy(config.policy),
            verbose=0,
            seed=config.seed,
            device=config["device"],
            _init_setup_model=True,
        )
    elif algo_name == "rps_single_action" and isinstance(env.model, RockPaperScissors):
        return RPSDummyLearner(
            agent_id,
            config["algorithm_configs"]["rps_single_action"],
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "reinforce":
        algorithm_config = config["algorithm_configs"]["reinforce"]
        n_rollout_steps = algorithm_config["n_rollout_steps"]
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return Reinforce(
            policy=algorithm_config["policy"],
            env=env,
            learning_rate=algorithm_config["learning_rate"],
            n_steps=n_rollout_steps,
            batch_size=algorithm_config["n_rollout_steps"] * config["num_envs"],
            action_dependent_std=config.policy["action_dependent_std"],
            gamma=algorithm_config["gamma"],
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            policy_kwargs=get_policy(config.policy),
            verbose=0,
            seed=config.seed,
            device=config["device"],
            _init_setup_model=True,
        )
    elif algo_name == "soccer_chase_ball" and isinstance(env.model, SimpleSoccer):
        return ChaseBallPolicy(
            agent_id,
            config,
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"multi_agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "soccer_goal_wall" and isinstance(env.model, SimpleSoccer):
        return GoalWallPolicy(
            agent_id,
            config,
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"multi_agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "soccer_block" and isinstance(env.model, SimpleSoccer):
        return BlockPolicy(
            agent_id,
            config,
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"multi_agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "soccer_handcrafted" and isinstance(env.model, SimpleSoccer):
        return HandcraftedPolicy(
            agent_id,
            config,
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"multi_agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "dqn":
        dqn_config = config["algorithm_configs"]["dqn"]
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return GPUDQN(
            policy=dqn_config["policy"],
            env=env,
            learning_rate=1e-4,
            buffer_size=dqn_config["buffer_size"],
            learning_starts=dqn_config["learning_starts"],
            batch_size=dqn_config["batch_size"],
            tau=dqn_config["tau"],
            gradient_steps=dqn_config["gradient_steps"],
            gamma=dqn_config["gamma"],
            train_freq=(dqn_config["n_rollout_steps"], "step"),
            optimize_memory_usage=False,
            target_update_interval=dqn_config["target_update_interval"],
            exploration_fraction=dqn_config["exploration_fraction"],
            exploration_initial_eps=dqn_config["exploration_initial_eps"],
            exploration_final_eps=dqn_config["exploration_final_eps"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            verbose=0,
            seed=None,
            device=config["device"],
        )
    else:
        raise ValueError(
            "No valid algorithm provided, check again for "
            + algo_name
            + " with model type: "
            + str(type(env.model).__name__)
        )


def get_algo_name(agent_id: int, config: Dict):
    assert not isinstance(config.algorithms, list)
    if config.policy_sharing and all_equal(config.algorithms):
        return config.algorithms[0]
    elif config.policy_sharing and not all_equal(config.algorithms):
        raise ValueError(
            "Policy sharing is true but not all provided policies are equal!"
        )
    elif not config.policy_sharing and len(config.algorithms) == 1:
        return config.algorithms[0]
    else:
        return config.algorithms[agent_id]


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
