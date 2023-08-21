import heapq
import math
from collections import namedtuple
from typing import Dict, Optional

import gizeh
import moviepy.editor as mpy
import torch
from gym import spaces
from gym.spaces import Space

import src.utils.torch_utils as th_ut
from src.envs.torch_vec_env import BaseEnvForVec

ObjectSpec = namedtuple("ObjectSpec", ["radius", "mass", "color"])


class SoccerStates:
    """
    Holds a batch of SimpleSoccer states.
    """

    def __init__(
        self,
        objects: torch.Tensor,
        energies: torch.Tensor,
        times: torch.Tensor,
        opponent_states: Optional[torch.Tensor],
    ):
        self.objects = objects
        self.energies = energies
        self.times = times
        self.opponent_states = opponent_states
        batch_shape = objects.shape[:-3]
        assert self.times.shape == batch_shape
        assert self.energies.shape[:-1] == batch_shape, f"{self.energies.shape}"
        if opponent_states is not None:
            self.opponent_states = opponent_states

    @property
    def batch_shape(self):
        return self.times.shape

    def __getitem__(self, item):
        if self.opponent_states is None:
            opponent_states = None
        else:
            opponent_states = tuple(s[item] for s in self.opponent_states)

        return SoccerStates(
            objects=self.objects[item],
            energies=self.energies[item],
            times=self.times[item],
            opponent_states=opponent_states,
        )

    def __setitem__(self, item, value):
        assert (
            len(value.__dict__) == 4
        ), f"Please account for other fields here: {value.__dict__}"

        self.objects[item] = value.objects
        self.energies[item] = value.energies
        self.times[item] = value.times
        if self.opponent_states is None:
            assert value.opponent_states is None
        else:
            assert isinstance(
                self.opponent_states, tuple
            )  # For now, only tuples are supported
            if value.opponent_states is None:
                v = 0
            else:
                v = value.opponent_states

            for s in self.opponent_states:
                s[item] = v

    @staticmethod
    def interpolate(states1, states2, alpha):
        if states1.opponent_states is None:
            assert states2.opponent_states is None
            opponent_states = None
        else:
            opponent_states = (
                1 - alpha
            ) * states1.opponent_states + alpha * states2.oppponent_states
        return SoccerStates(
            objects=(1 - alpha) * states1.objects + alpha * states2.objects,
            energies=(1 - alpha) * states1.energies + alpha * states2.energies,
            times=(1 - alpha) * states1.times + alpha * states2.times,
            opponent_states=opponent_states,
        )


class SimpleSoccer(BaseEnvForVec):
    BALL_IDX = 0

    def __init__(self, config, device):
        super().__init__(config, device)
        self._n_players_per_team = int(self.num_agents / 2)

        self.player_velocity = 3.5
        diag_vel = math.sqrt(0.5) * self.player_velocity
        self.action_target_speeds = torch.tensor(
            [
                [self.player_velocity, 0.0],
                [-self.player_velocity, 0.0],
                [0.0, self.player_velocity],
                [0.0, -self.player_velocity],
                [diag_vel, diag_vel],
                [-diag_vel, diag_vel],
                [diag_vel, -diag_vel],
                [-diag_vel, -diag_vel],
                [0.0, 0.0],  # No-op
            ],
            device=self.device,
        )

        self.dt = 0.05
        self.substeps = 1
        self.episode_max_steps = 165

        self.player_radius = 0.75
        self.ball_radius = 0.25
        ball = ObjectSpec(radius=self.ball_radius, mass=0.3, color=(1, 1, 1))
        controlled_players = []
        opponents = []
        for _ in range(self._n_players_per_team):
            player = ObjectSpec(
                radius=self.player_radius, mass=1.0, color=(1.0, 1.0, 0.2)
            )
            opponent = ObjectSpec(
                radius=self.player_radius, mass=1.0, color=(0.2, 1.0, 1.0)
            )
            controlled_players.append(player)
            opponents.append(opponent)
        self.objects = [ball] + controlled_players + opponents

        self.collision_distance_map = torch.tensor(
            [[o1.radius + o2.radius for o1 in self.objects] for o2 in self.objects],
            device=self.device,
        )
        self.collision_distance_map *= 1 - torch.eye(
            len(self.objects), device=self.device
        )

        self.collision_force_coef = 1000.0

        self.restitution_scale = 3.0  # Length-scale inside sigmoid (units: velocity)
        self.restitution_shift = 7.0  # velocity shift inside sigmoid
        self.ball_init_vel = 4.0

        self._radii = torch.tensor([o.radius for o in self.objects], device=self.device)
        self._masses = torch.tensor([o.mass for o in self.objects], device=self.device)

        assert len(self._masses) == len(self._radii)

        self.ball_ground_friction_coef = 0.2
        self.drag_coef = 0.2
        self.object_max_vel = 10.0

        # Dashing parameters
        self.resting_recovery = 0.3  # recovery rate when not dashing
        self.dashing_cost = 1.0  # cost of dashing
        self.dash_gain = (
            2.0  # how much faster when dashing (1.0 = 100% = twice as fast)
        )

        self.min_kick_displacement = 0.1
        self.kick_strengths = torch.tensor([0.0, 1200.0, 4000.0], device=self.device)

        self.player_outside_coef = 12.0

        self.field_height = 20.0
        self.halffield_height = self.field_height / 2
        self.field_width = 20.0
        self.halffield_width = self.field_width / 2
        self.goal_height = 0.5
        self.goal_width = 8.5
        self._goal_halfwidth = self.goal_width / 2
        self.player_force_coef = 10.0
        self.display_scale = 10.0

    def _get_num_agents(self) -> int:
        """
        Returns:
            int: number of agents in env
        """
        assert (
            self.config["num_agents"] % 2 == 0
        ), "Currently, the total number of agents needs to be even!"
        return self.config["num_agents"]

    def _init_observation_spaces(self) -> Dict[int, Space]:
        return {
            agent_id: spaces.Box(
                -10.0, 10.0, shape=(5 * 2 * int(self.num_agents / 2) + 5 + 1,)
            )
            for agent_id in range(self.num_agents)
        }

    def _init_action_spaces(self) -> Dict[int, Space]:
        # For each player: movement, dash [no/yes], kick [no/soft/strong]
        return {
            agent_id: spaces.MultiDiscrete([9, 2, 3])
            for agent_id in range(self.num_agents)
        }

    def to(self, device):
        self.device = device
        self.action_target_speeds = self.action_target_speeds.to(device)
        self.collision_distance_map = self.collision_distance_map.to(device)
        self._radii = self._radii.to(device)
        self._masses = self._masses.to(device)
        self.kick_strengths = self.kick_strengths.to(device)
        return self

    def sample_new_states(self, n):
        # Shape: [envs, objects, xy, pos/vel]

        # Player positions: sample the boxes first (without replacement, so
        # each player gets their own box). Then sample positions within boxes
        n_players = len(self.objects) - 1
        box_length = 4 * self.objects[-1].radius
        n_boxes_x = math.floor(self.field_width / box_length)
        n_boxes_y = math.floor(self.field_height / box_length)
        n_boxes = n_boxes_x * n_boxes_y
        box_probs = torch.full((n, n_boxes), fill_value=1 / n_boxes, device=self.device)
        box_indices = torch.multinomial(
            box_probs, num_samples=n_players, replacement=False
        )
        box_idxs_x = box_indices % n_boxes_x
        box_idxs_y = torch.div(box_indices, n_boxes_x, rounding_mode="trunc")
        player_xs = (
            box_idxs_x / n_boxes_x + 1 / (2 * n_boxes_x) - 1 / 2
        ) * self.field_width
        player_ys = (
            box_idxs_y / n_boxes_y + 1 / (2 * n_boxes_y) - 1 / 2
        ) * self.field_height

        player_vx = torch.zeros_like(player_xs)
        player_vy = torch.zeros_like(player_ys)

        ball_positions = self.halffield_width * (
            2 * torch.rand(n, 1, 2, 1, device=self.device) - 1
        )
        ball_positions[:, :, 1, :] /= 10.0  # Set y-position close to 0.
        ball_vels = self.ball_init_vel * (
            2 * torch.rand(n, 1, 2, 1, device=self.device) - 1
        )
        ball_vels[:, :, 0, :].abs_().mul_(-ball_positions[:, :, 0, :].sign())

        # Shape: [envs, objects, xy, pos/vel]
        # [B, O, XY, PV]
        player_positions = torch.stack((player_xs, player_ys), dim=-1)
        player_vels = torch.stack((player_vx, player_vy), dim=-1)

        # Add noise on top of player positions
        assert player_positions.shape == (n, 2 * self._n_players_per_team, 2)
        player_positions += (box_length - 2 * self.objects[-1].radius) * (
            -0.5 + torch.rand(n, len(self.objects) - 1, 2, device=self.device)
        )

        player_states = torch.stack((player_positions, player_vels), dim=-1)

        objects = torch.cat(
            (torch.cat((ball_positions, ball_vels), dim=-1), player_states), dim=-3
        )
        times = torch.zeros(n, dtype=torch.int, device=self.device)
        energies = torch.ones((n, n_players), dtype=torch.float32, device=self.device)

        states = SoccerStates(
            objects=objects,
            energies=energies,
            times=times,
            opponent_states=None,  # Even for LSTM policies, None is the default initial (batch) state
        )
        return states

    def compute_step(self, cur_states: SoccerStates, actions: Dict[int, torch.Tensor]):
        """
        Given a state batch and an action batch, returns:
         - observations
         - rewards
         - episode-done markers
         - updated states
        """

        first_team_actions = torch.concat(
            [actions[agent_id] for agent_id in range(self._n_players_per_team)], axis=1
        )
        second_team_actions = torch.concat(
            [
                actions[agent_id + self._n_players_per_team]
                for agent_id in range(self._n_players_per_team)
            ],
            axis=1,
        )

        object_states = cur_states.objects
        energies = cur_states.energies

        substep_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            # Ralston step RK2
            d_state_1, d_energies_1 = self.compute_state_derivative(
                object_states, energies, first_team_actions, second_team_actions
            )
            d_state_2, d_energies_2 = self.compute_state_derivative(
                object_states + substep_dt * 2.0 / 3 * d_state_1,
                energies + substep_dt * 2.0 / 3 * d_energies_1,
                first_team_actions,
                second_team_actions,
            )
            object_states = (
                object_states + substep_dt * (1 * d_state_1 + 3 * d_state_2) / 4
            )
            energies = energies + substep_dt * (1 * d_energies_1 + 3 * d_energies_2) / 4

        post_states = SoccerStates(
            objects=object_states,
            energies=energies,
            times=cur_states.times + 1,
            opponent_states=None,
        )

        positions_pre = cur_states.objects[..., 0]
        positions_post = object_states[..., 0]
        potential_pre = self._goal_potential(
            positions_pre[..., SimpleSoccer.BALL_IDX, :]
        )
        potential_post = self._goal_potential(
            positions_post[..., SimpleSoccer.BALL_IDX, :]
        )

        obses = self.get_observations(post_states)

        goal_rewards_first_team = potential_post - potential_pre
        goal_rewards_second_team = -goal_rewards_first_team
        rewards = {}
        for agent_id in range(self._n_players_per_team):
            rewards[agent_id] = goal_rewards_first_team.clone()
            rewards[
                agent_id + self._n_players_per_team
            ] = goal_rewards_second_team.clone()

        dones = (
            (goal_rewards_first_team != 0)
            | (
                positions_post[:, self.BALL_IDX, 1] > self.halffield_height
            )  # Ball over line
            | (positions_post[:, self.BALL_IDX, 1] < -self.halffield_height)
            | (positions_post[:, self.BALL_IDX, 0] > self.halffield_width)
            | (positions_post[:, self.BALL_IDX, 0] < -self.halffield_width)
            | (post_states.times > self.episode_max_steps)
        )

        return obses, rewards, dones, post_states

    def compute_state_derivative(
        self,
        pre_states: torch.Tensor,
        energies: torch.Tensor,
        actions: torch.Tensor,
        opponent_actions: torch.Tensor,
    ):
        """
        Computes the change of state given the current state and action.
        Used by an integrator to actually update the state.
        """
        assert actions.shape == (
            pre_states.shape[0],
            3 * self._n_players_per_team,
        ), f"{actions.shape}"

        dir_actions = actions[..., ::3]
        dash_actions = actions[..., 1::3]
        kick_actions = actions[..., 2::3]
        opponent_dir_actions = opponent_actions[..., ::3]
        opponent_dash_actions = opponent_actions[..., 1::3]
        opponent_kick_actions = opponent_actions[..., 2::3]

        # # The following seems slightly faster (combined with different action space)
        # dir_actions = actions[..., :self._n_players_per_team]
        # dash_actions = actions[..., self._n_players_per_team:2*self._n_players_per_team]
        # kick_actions = actions[..., 2*self._n_players_per_team:]
        # opponent_dir_actions = opponent_actions[..., :self._n_players_per_team]
        # opponent_dash_actions = opponent_actions[..., self._n_players_per_team:2*self._n_players_per_team]
        # opponent_kick_actions = opponent_actions[..., 2*self._n_players_per_team:]

        # positions, velocities = torch.split(pre_states, 2*len(self.radii), dim=-1)
        positions, velocities = pre_states[..., 0], pre_states[..., 1]

        # Enforce maximum velocity right off the bat.
        velocity_magnitudes = torch.linalg.norm(velocities, dim=-1, keepdim=True)
        # velocity_magnitudes = velocities.norm(dim=-1, keepdim=True)
        velocities = torch.where(
            velocity_magnitudes > self.object_max_vel,
            self.object_max_vel * velocities / velocity_magnitudes,
            velocities,
        )

        # Rotate target speed by inverting it
        opponent_target_velocities = -self.action_target_speeds[opponent_dir_actions]
        our_target_velocities = self.action_target_speeds[dir_actions]

        # Add dashing: check where energy is left, and make multipliers for the target velocities
        assert energies.shape[-1] == self._n_players_per_team * 2
        all_dash_actions = torch.cat((dash_actions, opponent_dash_actions), dim=-1)
        actual_dashing = (energies >= 0) * all_dash_actions
        recovering = (energies <= 1.0) * (1 - all_dash_actions)
        d_energies = -self.dashing_cost * actual_dashing
        d_energies += self.resting_recovery * recovering
        our_target_velocities *= (
            1.0 + self.dash_gain * actual_dashing[..., : self._n_players_per_team, None]
        )
        opponent_target_velocities *= (
            1.0 + self.dash_gain * actual_dashing[..., self._n_players_per_team :, None]
        )

        # It's a first-order system since we are controlling the velocity, not the position.
        our_forces = self.player_force_coef * (
            our_target_velocities - velocities[..., 1 : 1 + self._n_players_per_team, :]
        )
        opponent_forces = self.player_force_coef * (
            opponent_target_velocities
            - velocities[
                ..., 1 + self._n_players_per_team : 1 + 2 * self._n_players_per_team, :
            ]
        )

        # Ball ground friction
        ball_forces = -(
            self.ball_ground_friction_coef
            * velocities[..., 0:1, :]
            / (velocity_magnitudes[..., 0:1, :] + 1e-4)
        )
        forces = torch.cat((ball_forces, our_forces, opponent_forces), dim=-2)

        # velocity-proportional drag (all objects)
        forces -= self.drag_coef * velocities

        # Push back players which have left the field
        forces[..., 1:, 0] += (
            self.player_outside_coef
            * self.player_force_coef
            * (positions[..., 1:, 0] < -self.field_width / 2)
        )
        forces[..., 1:, 0] -= (
            self.player_outside_coef
            * self.player_force_coef
            * (positions[..., 1:, 0] > self.field_width / 2)
        )
        forces[..., 1:, 1] += (
            self.player_outside_coef
            * self.player_force_coef
            * (positions[..., 1:, 1] < -self.field_height / 2)
        )
        forces[..., 1:, 1] -= (
            self.player_outside_coef
            * self.player_force_coef
            * (positions[..., 1:, 1] > self.field_height / 2)
        )

        # Collision potential
        pos_diffs = (
            positions[..., :, None, :] - positions[..., None, :, :]
        )  # [B, K, K, 2]
        dist_matrix = torch.linalg.norm(pos_diffs, dim=-1)  # [B, K, K]
        radial_vectors = pos_diffs / (
            dist_matrix[..., None] + 1e-8
        )  # Add epsilon to avoid NaNs on the diagonal
        radial_vels = torch.einsum("...ik,...ijk->...ij", velocities, radial_vectors)
        joint_radial_vels = radial_vels + radial_vels.transpose(-2, -1)

        unclipped_displacements = self.collision_distance_map - dist_matrix
        displacements = torch.relu(unclipped_displacements)
        collision_force_matrix = radial_vectors * displacements[..., None]

        # Simulating a restitution<1 effect by letting the collision force be
        # modified by the sign of the joint radial velocity
        # collision_force_matrix[joint_radial_vels > 0] *= 0.0001
        collision_force_matrix *= torch.sigmoid(
            -(joint_radial_vels + self.restitution_shift) / self.restitution_scale
        )[..., None]

        collision_forces = self.collision_force_coef * collision_force_matrix.sum(
            dim=-2
        )  # [B, K, 2]
        forces = forces + collision_forces

        # Kicking
        # If ball is close to a player, and the player is kicking,
        # Give ball momentum in radial direction
        all_kick_actions = torch.cat((kick_actions, opponent_kick_actions), dim=-1)
        # ball_distances = dist_matrix[..., 0, :]
        ball_distances = unclipped_displacements[..., 0, 1:]
        kick_eligibility = (
            ball_distances > -self.min_kick_displacement
        ) * self.kick_strengths[all_kick_actions]
        kick_impulses = kick_eligibility[..., :, None] * radial_vectors[..., 0, 1:, :]

        kick_velocity_sum = kick_impulses.sum(dim=-2)
        forces[..., 1:, :] -= self.dt / self.substeps * kick_impulses
        forces[..., self.BALL_IDX, :] += self.dt / self.substeps * kick_velocity_sum
        accelerations = forces / self._masses[:, None]
        d_state = torch.stack((velocities, accelerations), dim=-1)
        return d_state, d_energies

    def _goal_potential(self, ball_pos):
        x_in = ball_pos[..., 0].abs() < self._goal_halfwidth
        y_in = ball_pos[..., 1].abs() > self.halffield_height - self.goal_height
        return (x_in & y_in) * torch.sign(ball_pos[..., 1])

    def _flip_states(self, states: SoccerStates):
        """
        Rotate objects by 180 degrees, swap players of team 1 with players of
        Team 2, flip the energies, but keep times and "opponent_states" the same.
        This is done to swap the roles of us and the opponents without changing
        the situation.
        """
        object_states = states.objects
        energies = states.energies
        swapped_players = object_states.clone()
        mid = 1 + self._n_players_per_team
        upper = 1 + 2 * self._n_players_per_team
        swapped_players[..., 1:mid, :, :] = object_states[..., mid:upper, :, :]
        swapped_players[..., mid:upper, :, :] = object_states[..., 1:mid, :, :]
        object_states = (
            -swapped_players
        )  # rotate by 180 degrees = negate, because (0, 0) is the center.
        energies = torch.cat(
            (
                energies[..., self._n_players_per_team :],
                energies[..., : self._n_players_per_team],
            ),
            dim=-1,
        )
        return SoccerStates(
            object_states, energies, states.times, states.opponent_states
        )

    def get_observations(self, states: SoccerStates):
        agent_obs = {}
        first_team_state_obs = self._get_sa_state_obs(states)
        second_team_state_obs = self._get_sa_state_obs(self._flip_states(states))
        for agent_playing_id in range(self._n_players_per_team):
            playing_id_info = (
                torch.ones((states.objects.shape[0], 1), device=self.device)
                * agent_playing_id
            )
            agent_obs[agent_playing_id] = torch.concat(
                [first_team_state_obs, playing_id_info], axis=1
            )
            agent_obs[agent_playing_id + self._n_players_per_team] = torch.concat(
                [second_team_state_obs, playing_id_info], axis=1
            )

        return agent_obs

    def _get_sa_state_obs(self, states):
        object_states = states.objects
        energies = states.energies
        normalized_energies = energies * 2 - 1.0
        player_states = object_states[..., 1:, :, :].flatten(start_dim=-2)
        full_player_states = torch.cat(
            (player_states, normalized_energies.unsqueeze(-1)), dim=-1
        )
        ball_states = object_states[..., :1, :, :]
        normalized_time = (
            torch.exp(
                -2.0 * (self.episode_max_steps - states.times) / self.episode_max_steps
            )
            * 2
            - 1.0
        )
        sa_full_obs = torch.cat(
            (
                full_player_states.flatten(start_dim=-2),
                ball_states.flatten(start_dim=-3),
                normalized_time[:, None],
            ),
            dim=-1,
        )

        return sa_full_obs

    def _translate_units_to_px(self, x, y):
        return (
            (x + 0.5 * self.field_width) * self.display_scale,
            (-y + 0.5 * self.field_height) * self.display_scale,
        )

    def custom_evaluation(self, learners, env, writer, iteration: int, config: Dict):
        """Method is called during training process and allows environment specific logging.

        Args:
            learners (Dict[int, BaseAlgorithm]):
            env (_type_): evaluation env
            writer: tensorboard summary writer
            config: Dict of additional data
        """
        self._store_eval_videos(learners, env, iteration, config, num_videos=3)

    def _store_eval_videos(self, learners, env, iteration, config, num_videos):
        filepath = config["experiment_log_path"]  # TODO: handle ind dir
        for video_iter in range(num_videos):
            clip = self._get_single_rendered_video(learners, env)
            filename = (
                "iteration_" + str(iteration) + "_clip_num_" + str(video_iter) + ".mp4"
            )
            clip.write_videofile(filepath + filename)

    def _get_single_rendered_video(self, learners, env):
        images = []
        states = self.sample_new_states(1)
        images.append(env.model.render(states[0]))

        dones = torch.zeros((1,), dtype=bool)

        while not dones.detach().item():
            observations = self.get_observations(states)
            actions = th_ut.get_ma_actions(learners, observations, deterministic=True)
            translated_actions = env.translate_joint_actions(actions)
            observations, rewards, dones, states = env.model.compute_step(
                states, translated_actions
            )
            episode_starts = dones
            images.append(env.model.render(states[0]))

        clip = mpy.ImageSequenceClip(images, fps=int(round(1 / env.model.dt)))
        return clip

    def get_ma_action(self, learners):
        actions_for_env = {}
        actions = {}
        additional_data = {}
        for agent_id, learner in learners.items():
            (
                sa_actions_for_env,
                sa_actions,
                sa_additional_data,
            ) = learner.get_actions_with_data(agent_id)
            actions_for_env[agent_id] = sa_actions_for_env
            actions[agent_id] = sa_actions
            additional_data[agent_id] = sa_additional_data
        return actions_for_env, actions, additional_data

    def render(self, state: SoccerStates):
        time = state.times.cpu().numpy()
        object_state = state.objects

        # no batch shape
        assert object_state.dim() == 3
        assert object_state.shape[1:] == (2, 2)
        assert time.ndim == 0

        eye_radius = 0.52 * self.objects[1].radius
        pupil_radius = 0.2 * self.objects[1].radius
        shadow_rel_displacement = 0.3

        # show potential
        positions, _ = object_state[..., 0], object_state[..., 1]
        potential = self._goal_potential(positions[SimpleSoccer.BALL_IDX])
        surface = gizeh.Surface(
            width=int(round(self.field_width * self.display_scale)),
            height=(int(round(self.field_height * self.display_scale)) + 10),
            bg_color=(0.5, 0.5, 0.5),
        )
        bg_rect = gizeh.rectangle(
            lx=0.98 * self.field_width * self.display_scale,
            ly=0.98 * self.field_height * self.display_scale,
            xy=(
                0.5 * self.field_width * self.display_scale,
                0.5 * self.field_height * self.display_scale,
            ),
            fill=(0.56, 0.9, 0.6),
        )
        bg_rect.draw(surface)

        time_rect_width = (
            0.9 * surface.width * (1 - time / (self.episode_max_steps + 1e-6))
        )
        time_rect_height = 2
        time_rect = gizeh.rectangle(
            lx=time_rect_width,
            ly=time_rect_height,
            xy=(0.05 * surface.width + time_rect_width / 2, surface.height - 7),
            fill=(0.6, 0.8, 1.0),
        )
        time_rect.draw(surface)

        for m in [1, -1]:
            box_line = gizeh.rectangle(
                lx=0.6 * self.field_width * self.display_scale,
                ly=0.15 * self.field_height * self.display_scale,
                xy=self._translate_units_to_px(0.0, m * (self.field_height * 0.425)),
                # xy=(0.5 * self.field_width * self.display_scale,
                #     0.5 + m*(.5 - 0.1) * self.field_height * self.display_scale),
                stroke_width=0.1 * self.display_scale,
                stroke=(1, 1, 1, 0.5),
                fill=None if m * potential <= 0 else (0.56, 1.0, 0.6),
            )
            box_line.draw(surface)

        mid_line = gizeh.polyline(
            points=[
                self._translate_units_to_px(-self.field_height / 2, 0.0),
                self._translate_units_to_px(+self.field_height / 2, 0.0),
            ],
            stroke_width=0.1 * self.display_scale,
            stroke=(1, 1, 1, 0.5),
        )

        mid_line.draw(surface)
        mid_circle = gizeh.circle(
            r=0.15 * self.field_width * self.display_scale,
            xy=self._translate_units_to_px(0.0, 0.0),
            stroke_width=0.1 * self.display_scale,
            stroke=(1, 1, 1, 0.5),
        )
        mid_circle.draw(surface)

        for m in [1, -1]:
            goal = gizeh.rectangle(
                lx=self.goal_width * self.display_scale,
                ly=self.goal_height * self.display_scale,
                xy=(
                    self.field_width / 2 * self.display_scale,
                    (
                        self.field_height / 2
                        + m * (self.halffield_height - self.goal_height / 2)
                    )
                    * self.display_scale,
                ),
                stroke=(1.0, 1.0, 1.0),
                stroke_width=0.2 * self.display_scale,
            )
            goal.draw(surface)

        draw_elements = []  # (order, element)

        for i, obj in enumerate(self.objects):
            positions = object_state[i, :, 0]
            px, py = positions.detach().cpu().numpy()
            energy = state.energies[i - 1].item() if i > 0 else 0.0

            shadow_displacement = obj.radius * shadow_rel_displacement
            for scale in [1.0]:
                obj_shadow = gizeh.circle(
                    r=1.1 * obj.radius * self.display_scale,
                    xy=self._translate_units_to_px(
                        px + scale * shadow_displacement,
                        py - scale * shadow_displacement,
                    ),
                    fill=(0, 0, 0, 0.25),
                )
                heapq.heappush(draw_elements, (-100 + i + scale, obj_shadow))

            obj_shape = gizeh.circle(
                r=obj.radius * self.display_scale,
                xy=self._translate_units_to_px(px, py),
                fill=obj.color,
            )
            heapq.heappush(draw_elements, (100 + i, obj_shape))
            obj_specular_highlight = gizeh.circle(
                r=0.75 * obj.radius * self.display_scale,
                xy=self._translate_units_to_px(
                    px - 0.20 * obj.radius, py + 0.20 * obj.radius
                ),
                fill=(1, 1, 1, 0.5),
            )
            heapq.heappush(draw_elements, (150 + i, obj_specular_highlight))

            if i > 0:
                team_id = (i - 1) // self._n_players_per_team
                team_sign = 1 - (team_id * 2)

                energy_bar = gizeh.rectangle(
                    lx=energy * 2.0 * obj.radius * self.display_scale,
                    ly=0.2 * obj.radius * self.display_scale,
                    xy=self._translate_units_to_px(
                        px, py - 1.6 * obj.radius * team_sign
                    ),
                    fill=(0.0, 0.4, 1.0, 0.5),
                )
                heapq.heappush(draw_elements, (100000 + i, energy_bar))

                eye_l_x = px + obj.radius / 2 * team_sign
                eye_r_x = px - obj.radius / 2 * team_sign

                eye_l_y = py + obj.radius / 4 * team_sign
                eye_r_y = py + obj.radius / 4 * team_sign
                eye_l = gizeh.circle(
                    r=eye_radius * self.display_scale,
                    xy=self._translate_units_to_px(eye_l_x, eye_l_y),
                    fill=(0.95, 0.95, 0.95),
                )
                heapq.heappush(draw_elements, (200 + 2 * i, eye_l))
                eye_r = gizeh.circle(
                    r=eye_radius * self.display_scale,
                    xy=self._translate_units_to_px(eye_r_x, eye_r_y),
                    fill=(0.95, 0.95, 0.95),
                )
                heapq.heappush(draw_elements, (201 + 2 * i, eye_r))

                def _get_pupil(eye_x, eye_y):
                    ball_x, ball_y = object_state[0, 0, 0], object_state[0, 1, 0]
                    pupil_angle = math.atan2(ball_y - eye_y, ball_x - eye_x)
                    pupil_x = eye_x + 0.8 * (eye_radius - pupil_radius) * math.cos(
                        pupil_angle
                    )
                    pupil_y = eye_y + 0.8 * (eye_radius - pupil_radius) * math.sin(
                        pupil_angle
                    )
                    return gizeh.circle(
                        r=pupil_radius * self.display_scale,
                        xy=self._translate_units_to_px(pupil_x, pupil_y),
                        fill=(0, 0, 0),
                    )

                pupil_l = _get_pupil(eye_l_x, eye_l_y)
                heapq.heappush(draw_elements, (400 + 2 * i, pupil_l))
                pupil_r = _get_pupil(eye_r_x, eye_r_y)
                heapq.heappush(draw_elements, (401 + 2 * i, pupil_r))

        while draw_elements:
            _, elem = heapq.heappop(draw_elements)
            elem.draw(surface)

        return surface.get_npimage(transparent=False, y_origin="top")
