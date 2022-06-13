import math
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.envs.simple_soccer import SimpleSoccer
from src.learners.base_learner import MABaseAlgorithm
from src.learners.simple_soccer_policies.common import extract_data_from_obs


class GoalWallPolicy(MABaseAlgorithm):
    def __init__(
        self,
        agent_id: int,
        config: Dict,
        policy: Type[BasePolicy] = None,
        env: Union[GymEnv, str, None] = None,
        policy_base: Type[BasePolicy] = ...,
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
            device=device,
            tensorboard_log=tensorboard_log,
        )
        self.full_config = config
        self.config = config["algorithm_configs"]["soccer_chase_ball"]
        dummy_env = SimpleSoccer(self.full_config["rl_envs"], device)

        self.n_players = dummy_env._n_players_per_team
        self.agent_team_id = self.agent_id % self.n_players

        self.pi = torch.tensor(math.pi, dtype=torch.float32, device=self.device)
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

        self.visualize = False
        self.viz_elems = []

    def predict(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        ball_pos, ball_vel, player_pos, player_vel, player_ener = extract_data_from_obs(
            observation, self.n_players * 2
        )
        ours_pos = player_pos[:, : self.n_players, :]
        ball_delta_pos = ball_pos[..., None, :] - ours_pos

        # Get ranks of players according to their x position, such that
        # they don't have to cross when moving toward their target positions
        ranks = torch.argsort(torch.argsort(ours_pos[..., 0], dim=-1), dim=-1)
        target_pos = self.target_pos[ranks]

        target_delta_pos = target_pos - ours_pos
        ball_angle = torch.atan2(target_delta_pos[..., 1], target_delta_pos[..., 0])

        discrete_angle = (
            ((ball_angle + self.pi + 2 * self.pi / 16 / 2 * self.pi / 8) % 8)
            .floor()
            .to(int)
        )

        discrete_angle_to_action = torch.tensor(
            [1, 7, 3, 6, 0, 4, 2, 5, 1], device=self.device
        )
        motion_actions = discrete_angle_to_action[discrete_angle]

        kick_actions = ball_delta_pos[..., :, 1] > 0

        action_template = torch.zeros(
            (observation.shape[0], self.n_players * 3),
            dtype=torch.int64,
            device=self.device,
        )

        action_template[..., ::3] = motion_actions
        action_template[..., 1::3] = 1  # everyone dashes
        action_template[..., 2::3] = 2 * kick_actions

        return (
            action_template[:, self.agent_team_id * 3 : (self.agent_team_id + 1) * 3],
            state,
        )

    def get_actions_with_data(
        self, agent_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Computes actions for the current state for env.step()

        Args:
            agent_id (int): determines which agents_actions will be returned

        Returns:
            actions_for_env: possibly adapted actions for env
            actions: predicted actions by policy
            additional_actions_data: additional data needed for algorithm later on
        """
        agent_obs = self._last_obs[agent_id]
        agent_actions, _ = self.predict(agent_obs)
        return agent_actions, agent_actions, ()
