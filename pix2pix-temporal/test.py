from __future__ import print_function
from PIL import Image
from skimage import io, feature, color, util
from othernetworks import InpaintGenerator
import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from util import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--model', type=str, default='checkpoint/facades/netG_model_epoch_200.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

checkpoint = torch.load(opt.model)
netG = InpaintGenerator()
netG.load_state_dict(checkpoint['generator'])

#image_dir = "dataset/{}/test/a/".format(opt.dataset)
image_dir = "/home/paperspace/Desktop/Colorization/DBZ_Dataset/{}/Val/".format(opt.dataset)
image_filenames = [x for x in sorted(os.listdir(image_dir)) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

#first previous frame needs to be zero image
initial = Image.new("RGB",[256,256])
initial = transform(initial)
previous = Variable(initial,volatile = True).view(1,-1,256,256)
previous.cuda()

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = color.rgb2gray(img)
    img = feature.canny(img,sigma = 2)
    img = util.invert(img)
    img = Image.fromarray(np.uint8(img)*255)
    img = transform(img)
    input = Variable(img, volatile=True).view(1, -1, 256, 256)
    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()
        #previous = previous.cuda()
    gen_input = torch.cat((input,previous),1)
    out = netG(gen_input)
    previous = out
    out = out.cpu()
    out_img = out.data[0]
    #out_img = np.array(out_img).astype(np.uint8)
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
