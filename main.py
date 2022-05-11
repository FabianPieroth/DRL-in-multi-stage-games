from typing import Dict

import hydra
from stable_baselines3.common.base_class import BaseAlgorithm

from src.envs.rock_paper_scissors import RockPaperScissors
from src.envs.sequential_auction import SequentialAuction
from src.envs.torch_vec_env import BaseEnvForVec, MATorchVecEnv
from src.learners.multi_agent_learner import MultiAgentCoordinator
from src.learners.ppo import VecPPO


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def get_env(config: Dict) -> MATorchVecEnv:
    env_name = config["rl_envs"]["name"]
    if env_name == "rockpaperscissors":
        env = RockPaperScissors(config["rl_envs"], device=config["device"])
    elif env_name == "sequential_auction":
        env = SequentialAuction(config["rl_envs"], device=config["device"])
    else:
        raise ValueError("No known env chosen, check again: " + str(env_name))
    return MATorchVecEnv(env, num_envs=config["num_envs"], device=config["device"])


def get_policies(config: Dict, env: MATorchVecEnv) -> Dict[int, BaseAlgorithm]:
    if config["algo_model_sharing"]:
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
    algo_name = get_algo_name(agent_id, config, env)
    env.set_env_for_current_agent(agent_id)
    if algo_name == "ppo":
        ppo_config = config["algorithm_configs"]["ppo"]
        return VecPPO(
            policy=ppo_config["policy"],
            env=env,
            device=config["device"],
            n_steps=ppo_config["n_rollout_steps"],
            batch_size=ppo_config["n_rollout_steps"] * config["num_envs"],
            tensorboard_log=f"logs/multi_agent_{agent_id}",
            verbose=0,
        )
    elif algo_name == "rps_single_action" and isinstance(env.model, RockPaperScissors):
        pass
    else:
        raise ValueError(
            "No valid algorithm provided, check again for "
            + algo_name
            + " with model type: "
            + str(type(env.model).__name__)
        )

    def get_algo_name(agent_id: int, config: Dict, env: MATorchVecEnv):
        if config["policy_sharing"] and isinstance(config["algorithms"], str):
            return config["algorithms"]
        elif config["policy_sharing"] and all_equal(config["algorithms"]):
            return config["algorithms"][0]
        elif config["policy_sharing"] and not all_equal(config["algorithms"]):
            raise ValueError(
                "Policy sharing is true but not all provided policies are equal!"
            )
        elif (
            not config["policy_sharing"]
            and len(config["algorithms"]) != env.model.num_agents
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


def main():
    config = get_config()
    env = get_env(config)
    learner_policies = get_policies(config, env)
    ma_learner = MultiAgentCoordinator()
    # train the agents
    ma_learner.learn(total_timesteps=config["total_training_steps"])


if __name__ == "__main__":
    main()
    print("Done!")
