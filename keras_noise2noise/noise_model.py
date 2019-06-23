import argparse

import cv2
import numpy as np


def get_noise_model(noise_type="gaussian,0,12"):
    tokens = noise_type.split(sep=",")

    if tokens[0] == "gaussian":
        gaussian_min_stddev = int(tokens[1])
        gaussian_max_stddev = int(tokens[2])

        def gaussian_noise(img):
            noise_img = img.astype(np.float)

            # Add some pixelwise gaussian noise
            stddev = np.random.uniform(gaussian_min_stddev, gaussian_max_stddev)
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise

            # TODO: Maybe add some JPEG compression artifacts?

            # Ensure valid value range and data type
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)

            return noise_img

        return gaussian_noise
    elif tokens[0] == "clean":
        return lambda img: img
    else:
        raise ValueError("noise_type should be 'gaussian' or 'clean'")


def get_args():
    parser = argparse.ArgumentParser(
        description="test noise model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image_size", type=int, default=128, help="training patch size")
    parser.add_argument(
        "--noise_model", type=str, default="gaussian,0,12", help="noise model to be tested"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_size = args.image_size
    noise_model = get_noise_model(args.noise_model)

    while True:
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
        noisy_image = noise_model(image)
        cv2.imshow("noise image", noisy_image)
        key = cv2.waitKey(-1)

        # "q": quit
        if key == 113:
            return 0


if __name__ == "__main__":
    """Run this script to see examples of the generated noise"""
    main()
