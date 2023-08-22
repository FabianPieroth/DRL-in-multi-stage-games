import math
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.envs.simple_soccer import SimpleSoccer, SoccerStates
from src.learners.base_learner import MABaseAlgorithm


class SoccerBasePolicy(MABaseAlgorithm):
    def __init__(
        self,
        agent_id: int,
        config: Dict,
        policy: Type[BasePolicy] = None,
        env: Union[GymEnv, str, None] = None,
        policy_base: Type[BasePolicy] = "MABasePolicy",
        learning_rate: Union[float, Schedule] = 0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
        support_multi_env: bool = True,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super().__init__(
            agent_id,
            config,
            policy,
            env,
            policy_base,
            learning_rate,
            policy_kwargs,
            tensorboard_log,
            verbose,
            device,
            support_multi_env,
            create_eval_env,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            supported_action_spaces,
        )
        self.full_config = config
        dummy_env = SimpleSoccer(self.full_config["rl_envs"], device)

        self.n_players = dummy_env._n_players_per_team
        self.agent_team_id = self.agent_id % self.n_players

        self.pi = torch.tensor(math.pi, dtype=torch.float32, device=self.device)

        h_corr_dist = 3.0
        h_corr_dy_shift = 1.0
        h_corr_dy_lscale = 0.1
        h_corr_dx_lscale = 5.0
        v_corr_dist = 1.8
        v_corr_dx_lscale = 1.0
        aim_corr_factor = 0.2
        distance_order_ypos_weight = 0.2

        self.our_goal_pos = torch.tensor(
            [0.0, -dummy_env.halffield_height], dtype=torch.float32, device=self.device
        )
        self.goal_halfwidth = dummy_env.goal_width / 2
        left = -self.goal_halfwidth + dummy_env.player_radius
        target_x = torch.linspace(left, -left, steps=self.n_players, device=device)
        target_y = -dummy_env.halffield_height + 1.5
        target_y = torch.full(
            size=(self.n_players,), fill_value=target_y, device=device
        )
        self.target_pos = torch.column_stack((target_x, target_y))
        self.pos_delta_y = -0.5

        self.visualize = False
        self.viz_elems = []

        self.h_corr_dist = h_corr_dist
        self.h_corr_dy_shift = h_corr_dy_shift
        self.h_corr_dy_lscale = h_corr_dy_lscale
        self.h_corr_dx_lscale = h_corr_dx_lscale

        self.v_corr_dist = v_corr_dist
        self.v_corr_dx_lscale = v_corr_dx_lscale

        self.aim_corr_factor = aim_corr_factor

        self.distance_order_ypos_weight = distance_order_ypos_weight
        self.aranges = dict()  # cache

    def predict(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        raise NotImplementedError()

    def get_actions_with_data(
        self, sa_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Computes actions for the current state for env.step()

        Args:
            agent_id (int): determines which agents_actions will be returned

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_data: additional data needed for algorithm later on
        """
        agent_actions, _ = self.predict(sa_obs)
        return agent_actions, agent_actions, {}


def extract_data_from_obs(
    agent_obs: torch.Tensor, n_players: int
) -> Tuple[torch.Tensor]:
    batch_size = agent_obs.shape[0]
    agent_obs_size = n_players * 5
    full_player_state = agent_obs[:, :agent_obs_size].reshape(
        (batch_size, n_players, 5)
    )
    full_ball_state = agent_obs[:, agent_obs_size : agent_obs_size + 4].reshape(
        (batch_size, 2, 2)
    )
    ball_pos = torch.stack([full_ball_state[:, 0, 0], full_ball_state[:, 1, 0]], dim=1)
    ball_vel = torch.stack([full_ball_state[:, 0, 1], full_ball_state[:, 1, 1]], dim=1)
    player_pos = torch.stack(
        [full_player_state[:, :, 0], full_player_state[:, :, 2]], dim=2
    )
    player_vel = torch.stack(
        [full_player_state[:, :, 1], full_player_state[:, :, 3]], dim=2
    )
    player_ener = full_player_state[:, :, 4].unsqueeze(-1)

    return (
        ball_pos.clone(),
        ball_vel.clone(),
        player_pos.clone(),
        player_vel.clone(),
        player_ener.clone(),
    )


class EvalPolicy:
    def compute_actions(self, states: SoccerStates):
        raise NotImplementedError()
