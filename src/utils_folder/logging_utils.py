"""Utilities for logging"""
import os

import imageio


def logging_plots_to_gif(log_path: str, num_frames: int = 10):
    """Create GIF from plots."""

    # collect path to pictures
    paths = list()
    for file in os.listdir(log_path):
        if file.endswith(".png"):
            paths.append(f"{log_path}/{file}")

    # sort and subselect
    paths = sorted(paths, key=lambda x: int(x[x.rfind("_") + 1 : -4]))
    paths = paths[0 :: int(len(paths) / num_frames)]

    # create GIF
    images = []
    for path in paths:
        images.append(imageio.imread(path))
    imageio.mimsave(f"{log_path}/movie.gif", images, duration=0.5)
