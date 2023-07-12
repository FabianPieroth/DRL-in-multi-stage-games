"""Base classes for vectorized Gym environments"""
import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

import src.utils.torch_utils as th_ut
from src.envs.equilibria import EquilibriumStrategy
from src.envs.space_translators import (
    BaseSpaceTranslator,
    BoxToDiscreteSpaceTranslator,
    IdentitySpaceTranslator,
    MultiDiscreteToDiscreteSpaceTranslator,
)

VecEnvIndices = Union[None, int, Iterable[int]]
TorchVecEnvStepReturn = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
]


class BaseEnvForVec(ABC):
    """Base Environment used for GPU based environment inference.
    Inherit from this class when writing a new env. Use MATorchVecEnv
    to wrap it."""

    def __init__(self, config: Dict, device):
        super().__init__()
        self.device = device
        self.config = config
        self.num_agents = self._get_num_agents()
        self.observation_spaces = self._init_observation_spaces()
        self.action_spaces = self._init_action_spaces()
        self.observation_space = None
        self.action_space = None
        self.equilibrium_strategies = self._get_equilibrium_strategies()
        self.equilibrium_strategies_known = self._are_equ_strategies_known()

    @abstractmethod
    def _get_num_agents(self) -> int:
        """
        Returns:
            int: number of agents in env
        """

    @abstractmethod
    def _init_observation_spaces(self) -> Dict[int, Space]:
        """Returns dict with agent - observation space pairs.
        Returns:
            Dict[int, Space]: agent_id: observation space
        """

    @abstractmethod
    def _init_action_spaces(self) -> Dict[int, Space]:
        """Returns dict with agent - action space pairs.
        Returns:
            Dict[int, Space]: agent_id: action space
        """

    @abstractmethod
    def to(self, device) -> Any:
        """Takes an available GPU device and shifts all tensors to the newly
        specified device.

        Args:
            device (int or string): The device to send the data to.
        Returns:
            Any: returns self of class
        """
        return self

    @abstractmethod
    def sample_new_states(self, n: int) -> torch.Tensor:
        """
        Args:
            n (int): Number of states to sample.

        Returns:
            States: Returns n states.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step(self, cur_states, actions: torch.Tensor):
        """
        Given a state batch and an action batch makes a step of env.
        Args:
            cur_states ():
            actions (torch.Tensor):
        Returns:
            observations:
            rewards:
            episode-done markers:
            updated_states:
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self, states) -> torch.Tensor:
        """Takes a number of states and returns the corresponding observations for the agents.

        Args:
            states (_type_): _description_

        Returns:
            observations: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state):
        """Takes a state and returns a tensor to render.

        Args:
            state (_type_):

        Returns:
            image:
        """

    def get_ma_actions_for_env(
        self,
        learners,
        observations: Dict[int, torch.Tensor],
        deterministic: bool = True,
        excluded_agents: List = None,
        no_grad: bool = True,
        states: Dict[int, torch.Tensor] = None,
    ):
        ma_actions = th_ut.get_ma_actions(
            learners, observations, deterministic, excluded_agents, no_grad
        )
        ma_actions = self.adapt_ma_actions_for_env(ma_actions, observations, states)
        return ma_actions

    def adapt_ma_actions_for_env(
        self,
        ma_actions: Dict[int, torch.Tensor],
        observations: Optional[Dict[int, torch.Tensor]] = None,
        states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Overwrite this method to apply env-specific adaptations to the ma_actions.

        Args:
            ma_actions (Dict[int, torch.Tensor]):
            observations (Dict[int, torch.Tensor]):

        Returns:
            Dict[int, torch.Tensor]: Adapated actions
        """
        return ma_actions

    def custom_evaluation(
        self,
        learners: Dict[int, BaseAlgorithm],
        env,
        writer,
        iteration: int,
        config: Dict,
    ):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            config: Dict of additional data
        """
        pass

    def _get_equilibrium_strategies(self) -> Dict[int, Optional[EquilibriumStrategy]]:
        """Overwrite this method to enable verification against the known
        equilibrium strategy. Each equilibrium strategy needs to be of type
        `EquilibriumStrategy`. Returns a strategy profile potentially
        containing `None` values.

        Note that verification only works if the Env is also of class `VerfiableEnv`.

        Returns:
            Dict[agent_id: EquilibriumStrategy or None]
        """
        return {agent_id: None for agent_id in range(self.num_agents)}

    def _are_equ_strategies_known(self) -> bool:
        """Checks whether there are valid equilibrium strategies.

        Returns: bool
        """
        return None not in self.equilibrium_strategies.values()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment-specific seeding is not used at the moment.
        In the underlying environment, random numbers are generated through
        calls to methods like `torch.randn` and can be seeded with
        `torch.manual_seed`.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


class VerfierEnvInfo(object):
    """Each VerifiableEnv needs to return such an object
    so that the verifier knows how to discretize the
    observation spaces.
    """

    def __init__(self, num_agents: int, num_stages: int) -> None:
        self.num_agents = num_agents
        self.num_stages = num_stages
        self.discretizations = self._init_info_dict()
        self.obs_info_indices = self._init_info_dict()
        self.boundary_values = self._init_info_dict()

    def _init_info_dict(self, inner_info: Optional[Dict] = None) -> Dict:
        info_dict = {}
        for stage in range(self.num_stages):
            agent_dict = {agent_id: inner_info for agent_id in range(self.num_agents)}
            info_dict[stage] = agent_dict
        return info_dict

    def get_info(self, stage: int, agent_id: int) -> Tuple:
        discr = self.discretizations[stage][agent_id]
        indices = self.obs_info_indices[stage][agent_id]
        boundaries = self.boundary_values[stage][agent_id]
        return discr, indices, boundaries


class VerifiableEnv(ABC):
    """Inherit from this class if you want your env to use the BFVerifier.
    Do NOT implement an __init__ method here as this might give conflicts due to
    multiple inheritance!
    """

    @abstractmethod
    def provide_env_verifier_info(
        self, stage: int, agent_id: int, obs_discretization: int
    ) -> Tuple:
        """The verifier needs some information from the environment to
        discretize the observation and action spaces accordingly.
        By leaving some choices in the env allows for some expert knowledge control.
        We assume each observation to be structured as follows:
            observations: [batch_size, num_agents, num_local_obs]

        For each local_obs we need information when and how to discretize what.
        For every stage and every agent one needs to provide
        the following:
        discretization-for-obs(Tuple[int]): how many discretization
                        points along each local_obs dim
        indices-for-obs-infos(Tuple[int]): which indizes in the
                        local_obs contain the infos to discretize
        boundary-values-for-obs(Dict[str, Tuple[int]]): lower and upper bound
                        for each local_obs dimension
        """

    def get_verifier_env_infos(self, obs_discretization: int) -> VerfierEnvInfo:
        v_env_info = VerfierEnvInfo(
            num_agents=self.num_agents, num_stages=self.num_stages
        )
        for stage in range(self.num_stages):
            for agent_id in range(self.num_agents):
                discre, indices, boundaries = self.provide_env_verifier_info(
                    stage=stage,
                    agent_id=agent_id,
                    obs_discretization=obs_discretization,
                )
                v_env_info.discretizations[stage][agent_id] = discre
                v_env_info.obs_info_indices[stage][agent_id] = indices
                v_env_info.boundary_values[stage][agent_id] = boundaries
        # TODO: Pull default boundaries away from specific env to here
        return v_env_info

    def get_obs_bin_indices(
        self, agent_id: int, agent_obs: torch.Tensor, stage: int
    ) -> torch.Tensor:
        discr, indices, boundaries = self.verfier_env_info.get_info(stage, agent_id)

        obs_bins = torch.zeros(
            (agent_obs.shape[0], len(indices)), dtype=torch.long, device=self.device
        )

        for k, obs_dim in enumerate(indices):
            sliced_agent_obs = agent_obs[:, obs_dim]
            low = float(boundaries["low"][k])
            high = float(boundaries["high"][k])

            obs_bins[:, k] = self._get_single_dim_obs_bins(
                sliced_agent_obs, discr[k], low, high
            )
        return obs_bins

    def _get_single_dim_obs_bins(
        self,
        sliced_agent_obs: torch.Tensor,
        num_discretization: int,
        low: float,
        high: float,
    ) -> torch.LongTensor:
        if num_discretization == 1:  # No additional info along this dim
            return 0
        obs_grid = torch.linspace(low, high, num_discretization, device=self.device)
        single_dim_obs_bins = torch.bucketize(sliced_agent_obs, obs_grid)
        return single_dim_obs_bins


class MATorchVecEnv(VecEnv):
    """Vectorized Gym environment base class"""

    metadata = {"render.modes": ["console"]}

    def __init__(self, model, num_envs, device, render_num_envs=16):
        """
        VecEnv which takes care of parallelization itself natively on GPU
        instead of vectorizing a single non-batch environment.
        Based on VecEnv interface by stable-baselines3.

        :param model:
        :param num_envs:
        :param device:
        :param render_n_envs:
        """
        self.device = device
        self.model = model
        self.num_envs = num_envs
        self.actions = None
        self.current_states = self.model.sample_new_states(num_envs)
        self.ep_stats = {
            "returns": torch.zeros((num_envs,), device=device),
            "lengths": torch.zeros((num_envs,), device=device),
        }
        self.sa_ep_stats = {
            agent_id: {
                "returns": torch.zeros((num_envs,), device=device),
                "lengths": torch.zeros((num_envs,), device=device),
            }
            for agent_id in range(self.model.num_agents)
        }
        # self.render_n_envs = render_num_envs

        self.observation_spaces = model.observation_spaces
        self.action_spaces = model.action_spaces
        self.joint_spaces = {
            "action_space": self.action_spaces,
            "observation_space": self.observation_spaces,
        }
        self.observation_space = None
        self.action_space = None
        self.learner_type = {
            agent_id: None for agent_id in range(self.model.num_agents)
        }
        self.agent_translators = {
            "action_space": {
                agent_id: None for agent_id in range(self.model.num_agents)
            },
            "observation_space": {
                agent_id: None for agent_id in range(self.model.num_agents)
            },
        }

    def set_space_translators_for_agent(
        self, agent_id: int, learner_config: Dict, translator_configs: Dict
    ):
        self._set_translator_for_learner(
            agent_id, learner_config, translator_configs, "action_space"
        )
        self._set_translator_for_learner(
            agent_id, learner_config, translator_configs, "observation_space"
        )

    def set_env_for_current_agent(self, agent_id: int):
        """Sets the environment to the view of provided agent.
        Needed to initialize leaners.
        """
        self.action_space = self.agent_translators["action_space"][agent_id].image_space
        self.observation_space = self.agent_translators["observation_space"][
            agent_id
        ].image_space

    def _set_translator_for_learner(
        self,
        agent_id: int,
        learner_config: Dict,
        translator_configs: Dict,
        space_type: str,
    ) -> Space:
        (
            space_translator_class,
            translator_config,
        ) = self._get_translator_type_and_config(
            agent_id, learner_config, translator_configs, space_type
        )
        agent_translator = space_translator_class(
            domain_space=self.joint_spaces[space_type][agent_id],
            config=translator_config,
        )
        self.agent_translators[space_type][agent_id] = agent_translator

    def _get_translator_type_and_config(
        self,
        agent_id: int,
        learner_config: Dict,
        translator_configs: Dict,
        space_type: str,
    ) -> Tuple[BaseSpaceTranslator, Dict]:
        translator_type = learner_config[space_type + "_translator"]
        agents_to_translate = self._get_list_of_agents_to_translate(
            learner_config, space_type
        )

        if translator_type == "identity" or agent_id not in agents_to_translate:
            space_translator_class, translator_config = IdentitySpaceTranslator, None
        elif translator_type == "multidiscrete_to_discrete":
            space_translator_class = MultiDiscreteToDiscreteSpaceTranslator
            agent_space = self.joint_spaces[space_type][agent_id]
            translator_config = {"multi_space_shape": tuple(agent_space.nvec)}
        elif translator_type == "box_to_discrete":
            space_translator_class = BoxToDiscreteSpaceTranslator
            translator_config = {
                "granularity": translator_configs["box_to_discrete_granularity"],
                "maximum_width": translator_configs["box_to_discrete_maximum_width"],
            }
        else:
            raise ValueError("No valid translation type provided: " + translator_type)
        return space_translator_class, translator_config

    def _get_list_of_agents_to_translate(self, learner_config, space_type):
        agents_to_translate = learner_config["translate_" + space_type + "s_for_agents"]
        if agents_to_translate == "None":
            agents_to_translate = [i for i in range(self.model.num_agents)]
        return agents_to_translate

    def step_async(self, actions: Dict[int, torch.Tensor]) -> None:
        self.actions = self.translate_joint_actions(actions)

    def translate_joint_actions(
        self, actions: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        translated_actions = {}
        for agent_id, agent_actions in actions.items():
            translated_actions[agent_id] = self.agent_translators["action_space"][
                agent_id
            ].inv_translate(agent_actions)
        return translated_actions

    def step_wait(self) -> TorchVecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        with torch.no_grad():
            current_states = self.current_states

            if isinstance(self.actions, np.ndarray) or isinstance(self.actions, list):
                actions = torch.tensor(self.actions, device=self.device)
            else:
                # We are assuming that actions are on the correct device.
                actions = self.actions

            obses, rewards, dones, next_states = self.model.compute_step(
                current_states, actions
            )
        self.current_states = next_states

        self.ep_stats["returns"] += torch.sum(
            torch.stack(tuple(rewards.values())), axis=0
        )  # global rewards
        self.ep_stats["lengths"] += torch.ones((self.num_envs,), device=self.device)
        episode_returns = self.ep_stats["returns"][dones]
        episode_lengths = self.ep_stats["lengths"][dones]
        self.ep_stats["returns"][dones] = 0
        self.ep_stats["lengths"][dones] = 0

        infos = dict(episode_returns=episode_returns, episode_lengths=episode_lengths)
        if dones.any():
            terminal_obs = self.model.get_observations(self.current_states[dones])
            infos["terminal_observation"] = self.clone_tensor_dict(terminal_obs)
        n_dones = dones.sum().cpu().item()
        self.current_states[dones] = self.model.sample_new_states(n_dones)

        for agent_id in range(self.model.num_agents):
            self.sa_ep_stats[agent_id]["returns"] += rewards[agent_id]
            self.sa_ep_stats[agent_id]["lengths"] += torch.ones(
                (self.num_envs,), device=self.device
            )
            sa_episode_returns = self.sa_ep_stats[agent_id]["returns"][dones]
            sa_episode_lengths = self.sa_ep_stats[agent_id]["lengths"][dones]
            self.sa_ep_stats[agent_id]["returns"][dones] = 0
            self.sa_ep_stats[agent_id]["lengths"][dones] = 0
            infos[agent_id] = dict(
                sa_episode_returns=sa_episode_returns,
                sa_episode_lengths=sa_episode_lengths,
            )

        if dones.any():
            # Override observations for resetted environments
            newly_sampled_obses = self.model.get_observations(
                self.current_states[dones]
            )
            for agent_id in range(self.model.num_agents):
                obses[agent_id][dones] = newly_sampled_obses[agent_id]
        # TODO: observation translator not in place yet!
        return (
            self.clone_tensor_dict(obses),
            self.clone_tensor_dict(rewards),
            dones.clone(),
            infos,
        )

    @staticmethod
    def clone_tensor_dict(
        dict_to_clone: Dict[Any, torch.tensor]
    ) -> Dict[Any, torch.tensor]:
        return {key: value.clone() for key, value in dict_to_clone.items()}

    def step(self, actions: np.ndarray):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observations, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment-specific seeding is not used at the moment.
        In the underlying environment, random numbers are generated through
        calls to methods like torch.randn and can be seeded with
        `torch.manual_seed`.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        self.current_states = self.model.sample_new_states(self.num_envs)
        obses = self.model.get_observations(self.current_states)
        return obses

    def get_action_grid(self, agent_id: int, grid_size: int = 4) -> torch.Tensor:
        """
        Generate a grid of alternative actions that are equally spaced on the
        action domain for continuous action spaces or fall back to return all
        actions for discrete action spaces.

        NOTE: Currently, the framework can only handle a static (state/stage
        independent) action space.
        """
        own_action_space = self.model.action_spaces[agent_id]
        own_action_space.shape
        if not all(own_action_space.bounded_above) or not all(
            own_action_space.bounded_below
        ):
            raise NotImplementedError("Action space is unbounded.")

        if own_action_space.shape != (1,):
            raise NotImplementedError(
                "High dimensional action spaces are not supported yet."
            )

        if isinstance(own_action_space, Discrete):
            grid_size = own_action_space.n  # arg grid_size is ignored then
            action_grid = torch.range(0, own_action_space.n, device=self.device)

        else:
            low = own_action_space.low.item()
            high = own_action_space.high.item()
            action_grid = torch.linspace(low, high, grid_size, device=self.device)

        return action_grid.view(grid_size, *own_action_space.shape)

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def render(self, mode: str) -> Optional[np.ndarray]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.model.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return self.model.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return self.model.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs

    def close(self) -> None:
        return self.model.close()
