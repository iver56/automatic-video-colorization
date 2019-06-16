"""
Rough work and tests to understand/fix errors
"""

import os
from os import listdir
from os.path import join

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from pix2pix_temporal.data import get_test_set

transform_list_rgb = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform_list_la = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

transform_rgb = transforms.Compose(transform_list_rgb)
transform_la = transforms.Compose(transform_list_la)

root_path = "E:/DBZ_Dataset/"
dataset = "Tf_Baseline"
train_set = join(root_path, dataset)
train_dir = join(train_set, "train")
print(train_dir)

photo_path = "E:\DBZ_Dataset\Tf_Baseline\Train"
image_filenames = [x for x in listdir(photo_path)]
test_set = get_test_set("E:/DBZ_Dataset/Tf_Baseline")


def create_iterator(sample_size):
    while True:
        sample_loader = DataLoader(
            dataset=test_set, batch_size=sample_size, drop_last=True
        )

        for item in sample_loader:
            yield item


smple_itr = create_iterator(8)


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new(
        "RGB",
        (
            width * img_per_row * columns + gap * (img_per_row - 1),
            height * int(len(inputs) / img_per_row),
        ),
    )
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def sample(iteration):
    input_image, target = next(smple_itr)

    input_image = postprocess(input_image)
    target = postprocess(target)
    img = stitch_images(input_image, target, target)
    samples_dir = root_path + "/samples_Tf_Baseline"
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    sample = "dataset" + "_" + str(iteration).zfill(5) + ".png"
    print("\nsaving sample " + sample + " - learning rate: " + str(3))
    img.save(os.path.join(samples_dir, sample))


sample(700)
