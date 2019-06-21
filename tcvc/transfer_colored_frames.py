from __future__ import print_function

import argparse
import os

import numpy as np
from PIL import Image
from hasel import rgb2hsl, hsl2rgb
from skimage.color import rgb2hsv, hsv2rgb
from tqdm import tqdm

from tcvc.gif import get_file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply the color of the small 256x256 to the corresponding"
        " higher-resolution original greyscale frames"
    )
    parser.add_argument(
        "--input-path",
        dest="input_path",
        type=str,
        required=True,
        help="The path to the folder that contains the greyscale images (frames) and a"
        ' subfolder named "colored" where the corresponding colored frames reside',
    )
    parser.add_argument(
        "--via-color-space",
        dest="via_color_space",
        type=str,
        default="hsl",
        choices=["hsl", "hsv"],
        help="Transfer hue and saturation from the color image to the greyscale image via HSL"
        " (Hue, Saturation, Lightness) or HSV (Hue, Saturation, Value)?",
    )
    args = parser.parse_args()

    greyscale_file_paths = get_file_paths(args.input_path)
    colored_file_paths = get_file_paths(os.path.join(args.input_path, "colored"))

    output_path = os.path.join(args.input_path, "colored_full_res")
    os.makedirs(output_path, exist_ok=True)

    assert len(greyscale_file_paths) == len(colored_file_paths)

    for i, greyscale_file_path in enumerate(tqdm(greyscale_file_paths)):
        greyscale_image = Image.open(greyscale_file_path)
        greyscale_image_np = np.array(greyscale_image)

        colored_file_path = colored_file_paths[i]
        colored_image = Image.open(colored_file_path).resize(
            greyscale_image.size, resample=Image.LANCZOS
        )
        colored_image_np = np.array(colored_image)

        # Transfer hue and saturation from the color image to the greyscale image via the
        # selected color space
        if args.via_color_space == "hsl":
            greyscale_image_np_hsl = rgb2hsl(greyscale_image_np)
            colored_image_np_hsl = rgb2hsl(colored_image_np)

            greyscale_image_np_hsl[:, :, 0] = colored_image_np_hsl[:, :, 0]  # hue
            greyscale_image_np_hsl[:, :, 1] = colored_image_np_hsl[
                :, :, 1
            ]  # saturation

            full_res_colored_image = hsl2rgb(greyscale_image_np_hsl)
        else:
            greyscale_image_np_hsv = rgb2hsv(greyscale_image_np)
            colored_image_np_hsv = rgb2hsv(colored_image_np)

            greyscale_image_np_hsv[:, :, 0] = colored_image_np_hsv[:, :, 0]  # hue
            greyscale_image_np_hsv[:, :, 1] = colored_image_np_hsv[
                :, :, 1
            ]  # saturation

            full_res_colored_image = hsv2rgb(greyscale_image_np_hsv)
            full_res_colored_image = (255 * full_res_colored_image).astype(np.uint8)

        Image.fromarray(full_res_colored_image).save(
            os.path.join(output_path, greyscale_file_path.name)
        )
