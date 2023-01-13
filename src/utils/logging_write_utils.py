"""Utilities for logging"""
import os
import warnings
from typing import Dict

import imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def logging_plots_to_gif(log_path: str, num_frames: int = 10):
    """Create GIF from plots."""

    # collect path to pictures
    paths = list()
    for file in os.listdir(log_path):
        if file.endswith(".png"):
            paths.append(f"{log_path}/{file}")

    # sort and subselect
    paths = sorted(paths, key=lambda x: int(x[x.rfind("_") + 1 : -4]))
    paths = paths[0 :: max(1, int(len(paths) / num_frames))]

    # create GIF
    images = []
    for path in paths:
        images.append(imageio.imread(path))
    imageio.mimsave(f"{log_path}/movie.gif", images, duration=0.5)


def log_data_dict_to_learner_loggers(
    learners, data_dict: Dict[int, float], data_name: str
):
    for agent_id, learner in learners.items():
        learner.logger.record(data_name, data_dict[agent_id])


def log_figure_to_writer(
    writer: SummaryWriter, fig: plt.Figure, iteration: int, name: str
):
    if fig is not None:
        writer.add_figure(name, fig, iteration)
