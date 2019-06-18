import os
from pathlib import Path

import imageio
import numpy as np


def get_file_paths(image_root_path, file_extensions=("jpg", "png")):
    """Return a list of paths to all files with the given in a directory
    Does not check subdirectories.
    """
    image_file_paths = []

    for root, dirs, filenames in os.walk(image_root_path):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            file_extension = filename.split(".")[-1]
            if file_extension.lower() in file_extensions:
                image_file_paths.append(Path(file_path))

        break  # prevent descending into subfolders

    return image_file_paths


def make_gif(folder, max_num_frames=100):
    frame_paths = get_file_paths(folder)

    if len(frame_paths) > max_num_frames:
        frame_indexes = np.linspace(
            start=0, stop=len(frame_paths) - 1, endpoint=True, num=max_num_frames, dtype=np.int
        ).tolist()
        frame_indexes = set(frame_indexes)
        frame_paths = [path for i, path in enumerate(frame_paths) if i in frame_indexes]

    gif_output_path = os.path.join(folder, "movie.gif")

    durations = [0.08] * len(frame_paths)
    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(gif_output_path, images, duration=durations)
