from typing import Dict

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
        shared_policy = get_policy_for_agent(0, config, env)
        return {agent_id: shared_policy for agent_id in range(env.model.num_agents)}
    else:
        return {
            agent_id: get_policy_for_agent(agent_id, config, env)
            for agent_id in range(env.model.num_agents)
        }


def set_space_translators_in_env(config: Dict, env: MATorchVecEnv):
    for agent_id in range(env.model.num_agents):
        env.set_space_translators_for_agent(
            agent_id,
            config["algorithm_configs"][get_algo_name(agent_id, config)],
            config["space_translators"],
        )


def get_policy_for_agent(
    agent_id: int, config: Dict, env: MATorchVecEnv
) -> BaseAlgorithm:
    algo_name = get_algo_name(agent_id, config)
    env.set_env_for_current_agent(agent_id)
    if algo_name == "ppo":
        ppo_config = config["algorithm_configs"]["ppo"]
        n_rollout_steps = ppo_config["n_rollout_steps"]
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return VecPPO(
            policy=ppo_config["policy"],
            env=env,
            device=config["device"],
            n_steps=n_rollout_steps,
            learning_rate=ppo_config["learning_rate"],
            gamma=ppo_config["gamma"],
            action_dependent_std=ppo_config["action_dependent_std"],
            batch_size=ppo_config["n_rollout_steps"] * config["num_envs"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            verbose=0,
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
        reinforce_config = config["algorithm_configs"]["reinforce"]
        n_rollout_steps = reinforce_config["n_rollout_steps"]
        if config["policy_sharing"]:
            n_rollout_steps *= env.model.num_agents
        return Reinforce(
            policy=reinforce_config["policy"],
            env=env,
            device=config["device"],
            n_steps=n_rollout_steps,
            learning_rate=reinforce_config["learning_rate"],
            gamma=reinforce_config["gamma"],
            action_dependent_std=reinforce_config["action_dependent_std"],
            batch_size=reinforce_config["n_rollout_steps"] * config["num_envs"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
            verbose=0,
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
