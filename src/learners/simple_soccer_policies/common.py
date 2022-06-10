from typing import Tuple

import torch

from src.envs.simple_soccer import SoccerStates


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
