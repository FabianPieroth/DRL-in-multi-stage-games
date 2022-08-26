from typing import Dict

from stable_baselines3.common.base_class import BaseAlgorithm

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.torch_vec_env import MATorchVecEnv
from src.learners.gpu_dqn import GPUDQN
from src.learners.ppo import VecPPO
from src.learners.reinforce import Reinforce
from src.learners.rps_dummy_learner import RPSDummyLearner


def get_policies(config: Dict, env: MATorchVecEnv) -> Dict[int, BaseAlgorithm]:
    if config["policy_sharing"]:
        shared_policy = get_policy_for_agent(0, config, env)
        return {agent_id: shared_policy for agent_id in range(env.model.num_agents)}
    else:
        return {
            agent_id: get_policy_for_agent(agent_id, config, env)
            for agent_id in range(env.model.num_agents)
        }


def get_policy_for_agent(
    agent_id: int, config: Dict, env: MATorchVecEnv
) -> BaseAlgorithm:
    algo_name = get_algo_name(agent_id, config)
    env.set_env_for_current_agent(agent_id, algo_name)
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
            gamma=ppo_config["gamma"],
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
            batch_size=reinforce_config["n_rollout_steps"] * config["num_envs"],
            tensorboard_log=config["experiment_log_path"] + f"Agent_{agent_id}",
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
            buffer_size=1_000_000,  # 1e6
            learning_starts=50000,
            batch_size=dqn_config["batch_size"],
            tau=1.0,
            gradient_steps=dqn_config["gradient_steps"],
            gamma=dqn_config["gamma"],
            train_freq=(dqn_config["n_rollout_steps"], "step"),
            optimize_memory_usage=False,
            target_update_interval=10000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
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
    if isinstance(config["algorithms"], str):
        return config["algorithms"]
    elif config["policy_sharing"] and all_equal(config["algorithms"]):
        return config["algorithms"][0]
    elif config["policy_sharing"] and not all_equal(config["algorithms"]):
        raise ValueError(
            "Policy sharing is true but not all provided policies are equal!"
        )
    elif (
        not config["policy_sharing"]
        and len(config["algorithms"]) != config["rl_envs"]["num_agents"]
    ):
        raise ValueError(
            "One needs to specify a policy for every agent or use policy sharing!"
        )
    else:
        return config["algorithms"][agent_id]


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
