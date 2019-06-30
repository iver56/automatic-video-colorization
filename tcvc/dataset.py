import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import feature, color, util
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose

from tcvc.util import load_img, get_image_file_paths


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, use_line_art=True, include_subfolders=False):
        super(DatasetFromFolder, self).__init__()
        self.use_line_art = use_line_art
        self.image_file_paths = get_image_file_paths(
            image_dir, include_subfolders=include_subfolders
        )
        transform_list = [ToTensor()]
        self.transform = Compose(transform_list)

    @staticmethod
    def get_frame_number(filename):
        filename = Path(filename).name
        m = re.search(r"\D*(\d{1,7})\.(jpg|jpeg|png)$", filename)
        if m:
            padded_frame_number_as_string = m.group(1)
            frame_number = 0
            stripped_number_as_string = m.group(1).lstrip("0")  # remove leading zeroes
            if len(stripped_number_as_string) > 0:
                frame_number = int(stripped_number_as_string)
            return frame_number, padded_frame_number_as_string
        raise Exception('Could not find a frame number in "{}"'.format(filename))

    @staticmethod
    def get_previous_frame_file_path(file_path):
        file_path = Path(file_path)
        frame_number, padded_frame_number_as_string = DatasetFromFolder.get_frame_number(
            file_path.name
        )
        num_digits = len(padded_frame_number_as_string)
        format_string = "{{:0{}d}}".format(num_digits)
        padded_previous_frame_number = format_string.format(frame_number - 1)
        previous_frame_file_path = file_path.with_name(
            file_path.name.replace(
                padded_frame_number_as_string, padded_previous_frame_number
            )
        )
        return previous_frame_file_path

    def __getitem__(self, index):
        """Load the image at the given index."""
        try:
            target_path = self.image_file_paths[index]
            frame_prev = self.get_prev(target_path)  # will be either black or colored
            target = load_img(target_path)
            input_image = color.rgb2gray(target)
            if self.use_line_art:
                # needed for lineart only, not grayscale
                input_image = feature.canny(input_image, sigma=1)
                input_image = util.invert(input_image)
            input_image = Image.fromarray((input_image * 255).astype(np.uint8))
            frame_prev = self.transform(frame_prev)
            target = self.transform(target)
            input_image = self.transform(input_image)

            return input_image, target, frame_prev
        except Exception as e:
            print("Something went wrong frame:")
            print(e)
            return self[0]

    def __len__(self):
        return len(self.image_file_paths)

    def get_prev(self, file_path):
        frame_number, _ = self.get_frame_number(file_path)
        previous_frame_file_path = self.get_previous_frame_file_path(file_path)

        if frame_number == 0 or not previous_frame_file_path.exists():
            initial_prev_frame = Image.new("RGB", [256, 256])
            return initial_prev_frame
        else:
            # define rnd num generator and if statement <0.5 take black or color
            rnd = np.random.uniform(0, 1)
            if rnd <= 0.5:
                prev = load_img(previous_frame_file_path)
            else:
                prev = Image.new("RGB", [256, 256])

            return prev
