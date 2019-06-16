from __future__ import print_function
from PIL import Image
from skimage import io, feature, color, util
from pix2pix_temporal.othernetworks import InpaintGenerator
from pix2pix_temporal.util import postprocess
import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pix2pix_temporal.data import get_training_set, get_test_set, get_val_set, create_iterator
from pix2pix_temporal.dataset import DatasetFromFolder
from pix2pix_temporal.util import is_image_file, load_img,save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--model', type=str, default='checkpoint/facades/netG_model_epoch_200.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

root_path = "/home/paperspace/Desktop/Temporal-Anime"
val_set = get_val_set(os.path.join(root_path , opt.dataset))
test_set = get_test_set(os.path.join(root_path , opt.dataset))

seq_sampler = SequentialSampler(val_set)

val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False,sampler = seq_sampler)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=16, shuffle=False)

checkpoint = torch.load(opt.model)
netG = InpaintGenerator()
netG.load_state_dict(checkpoint['generator'])
netG.cuda()

#image_dir = "dataset/{}/test/a/".format(opt.dataset)
image_dir = "/home/paperspace/Desktop/Temporal-Anime/{}/Val/".format(opt.dataset)
image_filenames = [x for x in sorted(os.listdir(image_dir)) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

#first previous frame needs to be zero image
#initial = Image.new("RGB",[256,256])
#initial = transform(initial)
#previous = Variable(initial,volatile = True).view(1,-1,256,256)
counter = 0
with torch.no_grad():
    for batch in val_data_loader:
        input, target, prev_frame = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            prev_frame = prev_frame.cuda()
        if counter != 0:
            prev_frame = tmp
            print("success")
        pred_input = torch.cat((input,prev_frame),1)
        out = netG(pred_input)
        tmp = out
        #out = postprocess(out)
        if not os.path.exists(os.path.join("result", opt.dataset)):
            os.makedirs(os.path.join("result", opt.dataset))
        #save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
        image_name = opt.dataset + "_" + str(counter).zfill(5) + ".jpg"
        save_image(out,"result/{}/{}".format(opt.dataset, image_name))
        print("saving:"+image_name)
        #imsave(out,"result/{}/{}".format(opt.dataset, image_name))
        counter += 1

'''
for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = color.rgb2gray(img)
    #img = feature.canny(img,sigma = 2)
    #img = util.invert(img)
    #img = Image.fromarray(np.uint8(img)*255)
    img = Image.fromarray(img)
    img = transform(img)
    input = Variable(img, volatile=True).view(1, -1, 256, 256)
    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()
        #previous = previous.cuda()
    gen_input = torch.cat((input,previous.cuda()),1)
    out = netG(gen_input)
    previous = out
    #out = out.cpu()
    #out_img = out.data[0]
    out = postprocess(out)
    #out_img = np.array(out_img).astype(np.uint8)
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    #save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
    imsave(out,"result/{}/{}".format(opt.dataset, image_name))
'''
