import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from hasel import rgb2hsl, hsl2rgb
from tqdm import tqdm


from resolution_enhancer.settings import DATA_DIR
from tcvc.util import get_image_file_paths

INPUT_SIZE = (128, 128)  # (width, height)


def get_random_coordinates(image):
    width = image.shape[1]
    height = image.shape[0]
    x1 = random.randint(0, width - INPUT_SIZE[0])
    y1 = random.randint(0, height - INPUT_SIZE[1])
    x2 = x1 + INPUT_SIZE[0]
    y2 = y1 + INPUT_SIZE[1]
    return x1, y1, x2, y2


if __name__ == "__main__":
    """
    Run this script to collect and images, preprocess them and split the set of
    images into two sets: training and validation.
    
    When using the --mode=prod argument, all available data is used for training, and the
    validation set becomes empty.
    
    resolution_enhancer.train uses the files written by this script.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        help="Path to a folder that contains (subfolders with) image frames with color",
    )
    arg_parser.add_argument(
        "--mode",
        dest="mode",
        help="Either split (use some of the data for validation) or prod (use all available"
        " data for training)",
        type=str,
        required=False,
        default="split",
    )
    args = arg_parser.parse_args()

    training_fraction = 1.0 if args.mode == "prod" else 0.9

    image_file_paths = [
        p for p in get_image_file_paths(args.dataset_path, include_subfolders=True)
    ]

    assert len(image_file_paths) > 0

    random.seed(1337)

    num_images = len(image_file_paths)
    print("Found {} images".format(num_images))
    split_index = int(round(num_images * training_fraction))

    directories_to_wipe = [
        DATA_DIR / "resolution_enhancer_dataset" / "training" / "input_images",
        DATA_DIR / "resolution_enhancer_dataset" / "training" / "target_images",
        DATA_DIR / "resolution_enhancer_dataset" / "validation" / "input_images",
        DATA_DIR / "resolution_enhancer_dataset" / "validation" / "target_images",
    ]
    for directory in directories_to_wipe:
        if os.path.exists(directory):
            # Remove all existing files in the directory
            files = glob.glob(os.path.join(directory, "*"))
            for f in files:
                os.remove(f)
        os.makedirs(directory, exist_ok=True)

    for i in tqdm(range(len(image_file_paths))):
        img_pil = Image.open(image_file_paths[i])
        target_img = np.array(img_pil)

        small_img_pil = img_pil.resize((256, 256), resample=Image.LANCZOS)
        blurry_img_pil = small_img_pil.resize(img_pil.size, resample=Image.LANCZOS)
        blurry_img = np.array(blurry_img_pil)

        dataset = "training" if i < split_index else "validation"
        num_examples = 1 if dataset == "training" else 1

        for j in range(num_examples):
            x1, y1, x2, y2 = get_random_coordinates(target_img)

            target_window = target_img[y1:y2, x1:x2, :]
            blurry_window = blurry_img[y1:y2, x1:x2, :]

            img_hsl = rgb2hsl(target_window)
            blurry_img_hsl = rgb2hsl(blurry_window)
            # Input image has original lightness channel, but blurry hue and saturation
            img_hsl[:, :, 0] = blurry_img_hsl[:, :, 0]  # hue
            img_hsl[:, :, 1] = blurry_img_hsl[:, :, 1]  # saturation
            input_window = hsl2rgb(img_hsl)

            if random.random() > 0.5:
                # Apply horizontal flip to 50 % of the images
                input_window = np.fliplr(input_window)
                target_window = np.fliplr(target_window)

            Image.fromarray(input_window).save(
                DATA_DIR
                / "resolution_enhancer_dataset"
                / dataset
                / "input_images"
                / (Path(image_file_paths[i]).stem + "_{}.png".format(j))
            )
            Image.fromarray(target_window).save(
                DATA_DIR
                / "resolution_enhancer_dataset"
                / dataset
                / "target_images"
                / (Path(image_file_paths[i]).stem + "_{}.png".format(j))
            )
