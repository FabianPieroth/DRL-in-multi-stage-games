import math
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.envs.simple_soccer import SimpleSoccer
from src.learners.base_learner import MABaseAlgorithm
from src.learners.simple_soccer_policies.common import extract_data_from_obs


class HandcraftedPolicy(MABaseAlgorithm):
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
        self.dummy_env = SimpleSoccer(self.full_config["rl_envs"], device)

        h_corr_dist = 3.0
        h_corr_dy_shift = 1.0
        h_corr_dy_lscale = 0.1
        h_corr_dx_lscale = 5.0
        v_corr_dist = 1.8
        v_corr_dx_lscale = 1.0
        aim_corr_factor = 0.2
        distance_order_ypos_weight = 0.2

        self.pi = torch.tensor(math.pi, dtype=torch.float32, device=self.device)
        self.our_goal_pos = torch.tensor(
            [0.0, -self.dummy_env.halffield_height],
            dtype=torch.float32,
            device=self.device,
        )
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

        self.n_players = self.dummy_env._n_players_per_team
        self.agent_team_id = self.agent_id % self.n_players

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

        batch_size = observation.shape[0]

        if batch_size in self.aranges:
            arange_batch_size = self.aranges[batch_size]
        else:
            self.aranges[batch_size] = torch.arange(batch_size, device=self.device)
            arange_batch_size = self.aranges[batch_size]

        ball_delta_pos = ball_pos[..., None, :] - ours_pos

        ball_distances = ball_delta_pos.norm(dim=-1)

        # Determine player roles:
        #  ball distance has highest precedence
        #  y-position has second highest precedence
        distance_order = torch.argsort(
            ball_distances - self.distance_order_ypos_weight * ours_pos[..., 1], dim=-1
        )

        assert self.n_players >= 3
        attacker_idxs = distance_order[..., 0]
        # defender_idxs = distance_order[..., 1:-1]
        defender_idxs = distance_order[..., 1]
        keeper_idxs = distance_order[..., -1]

        # Horizontal correction:
        #  If player above ball (dy > -h_corr_dy_shift, dx small), correct the target toward the center
        ours_x = ours_pos[..., 0]
        ball_x = ball_pos[..., 0]

        h_corr = (
            torch.sigmoid(
                (-ball_delta_pos[..., 1, None] + self.h_corr_dy_shift)
                / self.h_corr_dy_lscale
            )
            * torch.exp(-ball_delta_pos[..., 0, None].abs() / self.h_corr_dx_lscale)
            * ball_x.sign()[..., None, None]
            * (
                1
                - (
                    torch.sigmoid(ball_delta_pos[..., 1, None] / self.h_corr_dy_lscale)
                    * torch.exp(
                        -ball_delta_pos[..., 0, None].abs() / self.h_corr_dx_lscale
                    )
                )
            )  # Term that makes sure that if we are directly behind ball, there is no correction
            * torch.tensor([-self.h_corr_dist, 0.0], device=self.device)
        )

        # Vertical correction:
        #  If abs(dx) large, our target shifts below the ball
        v_corr = (
            1 - torch.exp(-ball_delta_pos[..., 0, None].abs() / self.v_corr_dx_lscale)
        ) * torch.tensor([0.0, -self.v_corr_dist], device=self.device)

        # Aim correction
        aim_corr = (
            torch.sigmoid(ball_delta_pos[..., 1, None] / self.h_corr_dy_lscale)
            * self.aim_corr_factor
            * ball_x[..., None, None]
            * torch.tensor([1.0, 0.0], device=self.device)
        )

        target_pos = torch.zeros(
            (observation.shape[0], self.n_players, 2), device=self.device
        )

        # Target positions for attackers
        target_pos[arange_batch_size, attacker_idxs, :] = (
            ball_pos
            + h_corr[arange_batch_size, attacker_idxs, :]
            + v_corr[arange_batch_size, attacker_idxs, :]
            + aim_corr[arange_batch_size, attacker_idxs, :]
        )

        # Target positions for defenders: midpoint between goal and ball
        target_pos[arange_batch_size, defender_idxs, :] = (
            ball_pos + self.our_goal_pos
        ) / 2

        # Target positions for keepers: between goal and ball, but closer to goal
        target_pos[arange_batch_size, keeper_idxs, :] = (
            ball_pos * 0.2 + self.our_goal_pos * 0.8
        )

        target_delta_pos = target_pos - ours_pos
        ball_angle = torch.atan2(target_delta_pos[..., 1], target_delta_pos[..., 0])

        discrete_angle = (
            ((((ball_angle + self.pi + 2 * self.pi / 16) / (2 * self.pi / 8))) % 8)
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
        action_template[arange_batch_size, 3 * attacker_idxs + 1] = 1  # attacker dashes
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
