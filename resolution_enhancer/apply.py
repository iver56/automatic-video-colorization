import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from keras_noise2noise.model import get_model
from keras_noise2noise.noise_model import get_noise_model
from resolution_enhancer.prepare_data import INPUT_SIZE
from resolution_enhancer.settings import MODELS_DIR, MODEL_ARCHITECTURE, DATA_DIR
from tcvc.util import get_image_file_paths


class ResolutionEnhancer:
    WEIGHT_FILE_PATH = os.path.join(
        MODELS_DIR, "resolution_enhancer_{}.h5".format(MODEL_ARCHITECTURE)
    )
    INSTANCE = None  # Singleton

    @staticmethod
    def get_instance():
        """
        Get the current instance of ResolutionEnhancer. Instantiate one if needed.
        :return:
        """
        if ResolutionEnhancer.INSTANCE is None:
            ResolutionEnhancer.INSTANCE = ResolutionEnhancer()
        return ResolutionEnhancer.INSTANCE

    def __init__(self):
        self.model = get_model(MODEL_ARCHITECTURE)
        self.model.load_weights(self.WEIGHT_FILE_PATH)

    def enhance(self, image):
        """
        Enhance a 128x128 patch of an image.

        :param image: 128x128x3 numpy array with dtype uint8 and values in range [0,255].
        :return: enhanced/denoised image (numpy array)
        """
        assert image.shape == (INPUT_SIZE[1], INPUT_SIZE[0], 3)

        denoised_image = self.model.predict(np.expand_dims(image, 0))[0]

        return denoised_image


def get_args():
    parser = argparse.ArgumentParser(
        description="Test trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-dir",
        dest="image_dir",
        type=str,
        required=False,
        help="test image dir",
        default=str(
            DATA_DIR / "resolution_enhancer_dataset" / "validation" / "input_images"
        ),
    )
    parser.add_argument(
        "--test_noise_model",
        type=str,
        default="clean",
        help="noise model for test images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            DATA_DIR / "resolution_enhancer_dataset" / "validation" / "denoised_images"
        ),
        help="if set, save resulting images otherwise show result using imshow",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    val_noise_model = get_noise_model(args.test_noise_model)

    denoiser = ResolutionEnhancer.get_instance()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_file_paths(args.image_dir)

    for image_path in tqdm(image_paths):
        image = np.array(Image.open(str(image_path)))
        h, w, _ = image.shape
        image = image[: (h // 16) * 16, : (w // 16) * 16]  # for stride (maximum 16)

        noise_image = val_noise_model(image)

        pred = denoiser.enhance(noise_image)
        denoised_image = np.clip(pred, 0, 255).astype(np.uint8)

        Image.fromarray(denoised_image).save(
            os.path.join(args.output_dir, Path(image_path).name)
        )


if __name__ == "__main__":
    main()
