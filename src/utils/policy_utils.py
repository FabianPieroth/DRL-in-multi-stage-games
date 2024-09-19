from typing import Callable, Dict, Union

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, MultiDiscrete
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.simple_soccer import SimpleSoccer
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.gpu_dqn import GPUDQN
from src.learners.networks.feature_extractors import CustomCNNExtractor
from src.learners.ppo import VecPPO
from src.learners.random_learner import RandomLearner
from src.learners.reinforce import Reinforce
from src.learners.rps_dummy_learner import RPSDummyLearner
from src.learners.simple_soccer_policies.block_policy import BlockPolicy
from src.learners.simple_soccer_policies.chase_ball_policy import ChaseBallPolicy
from src.learners.simple_soccer_policies.goal_wall_policy import GoalWallPolicy
from src.learners.simple_soccer_policies.handcrafted_policy import HandcraftedPolicy
from src.learners.td3 import TD3


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


def get_policy_kwargs(config_policy, **kwargs):
    policy_kwargs = dict(
        net_arch=[OmegaConf.to_container(config_policy.net_arch)],
        activation_fn=eval(str(config_policy.activation_fn)),
        action_activation_fn=eval(str(config_policy.action_activation_fn)),
    )
    policy_kwargs.update(kwargs)
    return policy_kwargs


def get_lr_schedule(lr_schedule_name: str, initial_value: float) -> Callable:
    MIN_LR = 1e-6
    if lr_schedule_name == "constant":
        return lambda x: initial_value
    elif lr_schedule_name == "linear":
        return (
            lambda progress_remaining: (1 - progress_remaining) * MIN_LR
            + progress_remaining * initial_value
        )
    elif lr_schedule_name == "exponential":
        return lambda progress_remaining: max(
            MIN_LR, initial_value * (progress_remaining**10)
        )
    else:
        raise ValueError(f"Learning rate scheduler {lr_schedule_name} unknown.")


def get_learner_and_policy(
    agent_id: int, config: Dict, env: MATorchVecEnv
) -> BaseAlgorithm:
    algo_name = get_algo_name(agent_id, config)
    env.set_env_for_current_agent(agent_id)
    algorithm_config = config["algorithm_configs"][algo_name]
    n_rollout_steps = algorithm_config["n_rollout_steps"]
    feature_extractor = get_algorithm_feature_extractor(algorithm_config)
    if algo_name == "ppo":
        if algorithm_config.full_batch_updates:
            batch_size = n_rollout_steps * config["num_envs"]
        else:
            batch_size = config["num_envs"]
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
            batch_size=batch_size,
            action_dependent_std=config.policy["action_dependent_std"],
            log_std_init=config.policy["log_std_init"],
            normalize_rewards=algorithm_config["normalize_rewards"],
            gamma=algorithm_config["gamma"],
            gae_lambda=algorithm_config.gae_lambda,
            clip_range=algorithm_config["clip_range"],
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            policy_kwargs=get_policy_kwargs(
                config.policy, **{"features_extractor_class": feature_extractor}
            ),
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
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return Reinforce(
            policy=algorithm_config["policy"],
            env=env,
            learning_rate=get_lr_schedule(
                algorithm_config["learning_rate_schedule"],
                algorithm_config["learning_rate"],
            ),
            n_steps=n_rollout_steps,
            batch_size=algorithm_config["n_rollout_steps"] * config["num_envs"],
            action_dependent_std=config.policy["action_dependent_std"],
            log_std_init=config.policy["log_std_init"],
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
            policy_kwargs=get_policy_kwargs(
                config.policy, **{"features_extractor_class": feature_extractor}
            ),
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
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return GPUDQN(
            policy=algorithm_config["policy"],
            env=env,
            learning_rate=1e-4,
            buffer_size=algorithm_config["buffer_size"],
            learning_starts=algorithm_config["learning_starts"],
            batch_size=algorithm_config["batch_size"],
            tau=algorithm_config["tau"],
            gradient_steps=algorithm_config["gradient_steps"],
            gamma=algorithm_config["gamma"],
            train_freq=(algorithm_config["n_rollout_steps"], "step"),
            optimize_memory_usage=False,
            target_update_interval=algorithm_config["target_update_interval"],
            exploration_fraction=algorithm_config["exploration_fraction"],
            exploration_initial_eps=algorithm_config["exploration_initial_eps"],
            exploration_final_eps=algorithm_config["exploration_final_eps"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            policy_kwargs={"features_extractor_class": feature_extractor},
            verbose=0,
            seed=None,
            device=config["device"],
        )
    elif algo_name == "td3":
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        else:
            n_rollout_steps = algorithm_config["n_rollout_steps"]

        learning_rate = (
            get_lr_schedule(
                algorithm_config["learning_rate_schedule"],
                algorithm_config["learning_rate"],
            )
            if "learning_rate_schedule" in algorithm_config
            else algorithm_config["learning_rate"]
        )

        return TD3(
            policy=algorithm_config["policy"],
            env=env,
            learning_rate=learning_rate,
            buffer_size=algorithm_config["buffer_size"],
            learning_starts=algorithm_config["learning_starts"],
            batch_size=algorithm_config["batch_size"],
            tau=algorithm_config["tau"],
            gradient_steps=algorithm_config["gradient_steps"],
            action_noise=algorithm_config["action_noise"],
            gamma=algorithm_config["gamma"],
            train_freq=(n_rollout_steps, "step"),
            optimize_memory_usage=False,
            policy_delay=algorithm_config["policy_delay"],
            target_policy_noise=algorithm_config["target_policy_noise"],
            target_noise_clip=algorithm_config["target_noise_clip"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            verbose=0,
            seed=config.seed,
            device=config["device"],
        )
    elif algo_name == "random":
        return RandomLearner(
            agent_id,
            config,
            env=env,
            device=config["device"],
            tensorboard_log=config["experiment_log_path"] + f"multi_agent_{agent_id}",
            verbose=0,
            seed=config.seed,
        )
    else:
        raise ValueError(
            "No valid algorithm provided, check again for "
            + algo_name
            + " with model type: "
            + str(type(env.model).__name__)
        )


def get_algorithm_feature_extractor(algorithm_config: Dict) -> BaseFeaturesExtractor:
    feature_extractor = FlattenExtractor
    if "feature_extractor" in algorithm_config:
        if algorithm_config["feature_extractor"] == "flatten":
            feature_extractor = FlattenExtractor
        elif algorithm_config["feature_extractor"] == "customCNN":
            feature_extractor = CustomCNNExtractor
        else:
            raise ValueError(
                "No valid feature extractor selected! Check "
                + algorithm_config["feature_extractor"]
            )
    return feature_extractor


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


def sample_random_actions(
    action_space: gym.Space, batch_size: int, device: Union[str, int]
) -> torch.Tensor:
    if isinstance(action_space, Box):
        actions = sample_random_box_actions(action_space, batch_size, device)
    elif isinstance(action_space, Discrete):
        actions = sample_random_discrete_actions(action_space, batch_size, device)
    elif isinstance(action_space, MultiDiscrete):
        actions = sample_random_multidiscrete_actions(action_space, batch_size, device)
    else:
        raise NotImplementedError(
            "No random sampling implemented for given action_space type!"
        )
    return actions


def sample_random_box_actions(
    action_space: Box, batch_size: int, device: Union[str, int]
) -> torch.Tensor:
    """In creating a sample of the box, each coordinate is sampled (independently) from a distribution
    that is chosen according to the form of the interval:

    * :math:`[a, b]` : uniform distribution
    * :math:`[a, \infty)` : shifted exponential distribution
    * :math:`(-\infty, b]` : shifted negative exponential distribution
    * :math:`(-\infty, \infty)` : normal distribution"""
    actions = torch.zeros((batch_size,) + action_space.shape, device=device)
    unbounded = ~action_space.bounded_below & ~action_space.bounded_above
    upp_bounded = ~action_space.bounded_below & action_space.bounded_above
    low_bounded = action_space.bounded_below & ~action_space.bounded_above
    bounded = action_space.bounded_below & action_space.bounded_above

    # Vectorized sampling by interval type
    actions[:, unbounded] = torch.normal(
        mean=0.0,
        std=1.0,
        size=(batch_size,) + unbounded[unbounded].shape,
        device=device,
    )

    actions[:, low_bounded] = actions[:, low_bounded].exponential_()

    actions[:, upp_bounded] = -1.0 * actions[
        :, upp_bounded
    ].exponential_() + torch.tensor(action_space.high[upp_bounded], device=device)

    # (a - b) * torch.rand(batch_size, bounded_action_size) + b
    low = torch.tensor(action_space.low[bounded], device=device).unsqueeze(0)
    high = torch.tensor(action_space.high[bounded], device=device).unsqueeze(0)
    actions[:, bounded] = (low - high) * torch.rand(
        (batch_size,) + bounded[bounded].shape, device=device
    ) + high
    return actions


def sample_random_discrete_actions(
    action_space: Discrete, batch_size: int, device: Union[str, int]
) -> torch.Tensor:
    return torch.randint(high=action_space.n, size=(batch_size,), device=device)


def sample_random_multidiscrete_actions(
    action_space: MultiDiscrete, batch_size: int, device: Union[str, int]
) -> torch.Tensor:
    actions_list = [
        torch.randint(high=upper_bound, size=(batch_size,), device=device)
        for upper_bound in action_space.nvec
    ]
    return torch.stack(actions_list, dim=1)
