from os import listdir
from os.path import join, exists

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from skimage import color
from torch.utils.data import DataLoader

from pix2pix_temporal.util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.photo_path = image_dir
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        target_path = join(self.photo_path, self.image_filenames[index])
        frame_num = target_path.split("e")[-1]
        frame_num = int(frame_num.split(".")[0]) + 1
        frame_1, frame_1_gray = self.get_prev(frame_num)
        target = load_img(target_path)
        input_image = color.rgb2gray(target)
        input_image = Image.fromarray(input_image)
        frame_1 = self.transform(frame_1)
        frame_1_gray = self.transform(frame_1_gray)
        target = self.transform(target)
        input_image = self.transform(input_image)

        return input_image, target, frame_1, frame_1_gray

    def __len__(self):
        return len(self.image_filenames)

    def get_prev(self, num):
        if not exists(join(self.photo_path, "frame" + str(num) + ".jpg")):
            prev = load_img(join(self.photo_path, "frame" + str(num - 1) + ".jpg"))
            prev_gray = color.rgb2gray(prev)
            prev_gray = Image.fromarray(prev_gray)
            return prev, prev_gray
            # frame_1="nothing!"
        else:
            prev = load_img(join(self.photo_path, "frame" + str(num) + ".jpg"))
            prev_gray = color.rgb2gray(prev)
            prev_gray = Image.fromarray(prev_gray)
            return prev, prev_gray
            # frame_1 = join(self.photo_path,"frame"+str(frame_num)+".jpg")


train_path = "E:/DBZ_Dataset/Tf_Baseline/Tester"
train_set = DatasetFromFolder(train_path)
training_data_loader = DataLoader(
    dataset=train_set, num_workers=0, batch_size=5, shuffle=False
)

for i in range(10):
    x = np.random.uniform(0, 1)
    print(x)
