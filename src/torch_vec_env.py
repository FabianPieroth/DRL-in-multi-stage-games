from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

TorchVecEnvStepReturn = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
]


class TorchVecEnv:
    def __init__(self, model, num_envs, device, render_n_envs=16):
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
            "returns": torch.zeros((num_envs, self.model.num_agents), device=device),
            "lengths": torch.zeros((num_envs,), device=device),
        }
        self.render_n_envs = render_n_envs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> TorchVecEnvStepReturn:
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

        self.ep_stats["returns"] += rewards
        self.ep_stats["lengths"] += torch.ones((self.num_envs,), device=self.device)

        self.current_states = next_states

        n_dones = dones.sum()
        self.current_states[dones] = self.model.sample_new_states(n_dones)

        episode_returns = self.ep_stats["returns"][dones]
        episode_lengths = self.ep_stats["lengths"][dones]
        infos = (episode_returns, episode_lengths)

        self.ep_stats["returns"][dones] = 0
        self.ep_stats["lengths"][dones] = 0

        # Override observations for resetted environments after using them to
        # set "terminal_observation"
        obses[dones] = self.model.get_observations(self.current_states[dones])

        return obses.clone(), rewards.clone(), dones.clone(), infos

    def step(self, actions: np.ndarray):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
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
        raise NotImplementedError()

    def reset(self):
        self.current_states = self.model.sample_new_states(self.num_envs)
        obses = self.model.get_observations(self.current_states)
        return obses

    def get_images(self) -> Sequence[np.ndarray]:
        return [
            self.model.render(state)
            for state in self.current_states[: self.render_n_envs]
        ]

    def render(self, mode: str) -> Optional[np.ndarray]:
        """
        Vendored from stable-baselines3
        """
        imgs = self.get_images()

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")


def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Vendored from stable-baselines3
    https://github.com/DLR-RM/stable-baselines3/blob/e9a8979022d7005560d43b7a9c1dc1ba85f7989a/stable_baselines3/common/vec_env/base_vec_env.py

    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
