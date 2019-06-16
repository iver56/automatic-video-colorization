"""
Rough work and tests to understand/fix errors
"""

import os
from os import listdir
from os.path import join

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.misc import imread, imresize
from torch.utils.data import DataLoader

from pix2pix_temporal.data import get_test_set


# im = imread('/Users/harrythasarathan/Downloads/Mighty_Morphin_Power_YEngers.jpg')
# print(im.shape)
# im = np.resize(im,(256,256,3))
# print(im.shape)
# im = color.rgb2gray(im)
# im = feature.canny(im,sigma=2)
# im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# im = cv2.Canny(im, 100,200)
# im = util.invert(im)
# im = resize(im,(1,im.shape[0],im.shape[1]))
# print(im.shape)
# im = cv2.bilateralFilter(im,15,75,75)
# io.imshow(im)
# io.show()
# io.imsave('/Users/harrythasarathan/Desktop/Python/transform2.jpg',img_as_uint(im))


def load_img(filepath):
    # img = Image.open(filepath).convert('RGB')
    # img = img.resize((256, 256), Image.BICUBIC)
    img = imread(filepath)
    img = imresize(img, (256, 256))
    return img


transform_list_rgb = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform_list_la = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

# transform_list_rgb = [transforms.ToTensor()]
# transform_list_la = [transforms.ToTensor()]

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

# convert_gray = transforms.Grayscale(1)
# flepth = "/Users/harrythasarathan/Desktop/Python/mighty_morphin_kanye_rangers.jpg"
# target = np.array(target)
"""
target = load_img(join(train_dir, image_filenames[1]))
input = color.rgb2gray(target)
input = feature.canny(input,sigma = 2)
input = util.invert(input)
print(input.shape)
print(target.shape)
io.imshow(input)
io.show()
input = Image.fromarray(input)
target = transform_rgb(target)
input = transform_la(input)
"""


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
    input, target = next(smple_itr)
    """
    input = Variable(input,volatile = True)
    if opt.cuda: 
        input = input.cuda()
        target = target.cuda()
    prediction = netG(input)
    prediction = postprocess(prediction)
    """
    input = postprocess(input)
    target = postprocess(target)
    img = stitch_images(input, target, target)
    samples_dir = root_path + "/samples_Tf_Baseline"
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    sample = "dataset" + "_" + str(iteration).zfill(5) + ".png"
    print("\nsaving sample " + sample + " - learning rate: " + str(3))
    img.save(os.path.join(samples_dir, sample))


sample(700)

"""
target = load_img(flepth)
input = color.rgb2gray(target)
input = feature.canny(input,sigma = 2)
input = util.invert(input)
print(input.shape)
print(target.shape)
io.imshow(input)
io.show()
input = Image.fromarray(np.uint8(input)*255)
target = transform_rgb(target)
input = transform_la(input)
#input = transforms.ToPILImage()(input)
#input.save('/Users/harrythasarathan/Desktop/Python/transformF.jpg')
save_image(input,'/Users/harrythasarathan/Desktop/Python/transformT.jpg' )
"""
