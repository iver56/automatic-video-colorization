import random

import numpy as np
from PIL import Image
from keras.utils import Sequence

from tcvc.util import get_image_file_paths


class NoisyImageGenerator(Sequence):
    def __init__(
        self,
        image_dir,
        target_image_dir,
        source_noise_model,
        target_noise_model,
        batch_size=32,
        image_size=128,
    ):
        self.input_image_paths = get_image_file_paths(image_dir)
        self.target_image_paths = get_image_file_paths(target_image_dir)
        assert len(self.input_image_paths) == len(self.target_image_paths)
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.input_image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError(
                "image dir '{}' does not include any image".format(image_dir)
            )

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_idx = random.randint(0, self.image_num - 1)
            input_image_path = self.input_image_paths[image_idx]
            target_image_path = self.target_image_paths[image_idx]
            image = np.array(Image.open(str(input_image_path)))
            target_image = np.array(Image.open(str(target_image_path)))
            assert image.shape == target_image.shape
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_input_patch = image[i : i + image_size, j : j + image_size]
                clean_target_patch = target_image[
                    i : i + image_size, j : j + image_size
                ]
                x[sample_id] = self.source_noise_model(clean_input_patch)
                y[sample_id] = self.target_noise_model(clean_target_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(
        self, image_dir, target_image_dir, val_noise_model, max_images_to_load=2000
    ):
        self.input_image_paths = get_image_file_paths(image_dir)
        self.target_image_paths = get_image_file_paths(target_image_dir)
        assert len(self.input_image_paths) == len(self.target_image_paths)
        self.image_num = len(self.input_image_paths)
        self.approve_rate = 1.0
        if self.image_num > max_images_to_load:
            self.approve_rate = max_images_to_load / self.image_num
        self.data = []

        if self.image_num == 0:
            raise ValueError(
                "image dir '{}' does not include any image".format(image_dir)
            )

        for i in range(len(self.input_image_paths)):
            if len(self.data) > 1 and random.random() > self.approve_rate:
                continue
            if len(self.data) >= max_images_to_load:
                break
            input_path = self.input_image_paths[i]
            target_path = self.target_image_paths[i]
            x = np.array(Image.open(str(input_path)))
            y = np.array(Image.open(str(target_path)))
            h, w, _ = y.shape
            y = y[: (h // 16) * 16, : (w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(x)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

        print("Loaded {} validation images".format(len(self.data)))

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
