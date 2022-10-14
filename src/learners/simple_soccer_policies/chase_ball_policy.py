from typing import Optional, Tuple

import torch

from src.learners.simple_soccer_policies.common import (
    SoccerBasePolicy,
    extract_data_from_obs,
)


class ChaseBallPolicy(SoccerBasePolicy):
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

        ball_distances = (ball_pos[:, None, :] - ours_pos).norm(dim=-1, keepdim=True)
        projected_ball_pos = (
            ball_pos[:, None, :] + 1e-1 * ball_vel[:, None, :] * ball_distances
        )

        target_pos = projected_ball_pos + torch.tensor(
            [0, self.pos_delta_y], device=self.device
        )

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
