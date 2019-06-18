from __future__ import print_function

import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.utils import save_image

from tcvc.data import get_val_set
from tcvc.gif import make_gif
from tcvc.othernetworks import InpaintGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pix2pix-PyTorch-implementation")
    parser.add_argument(
        "--model",
        type=str,
        default="tcvc/checkpoint/Temporal/netG_weights_epoch_1.pth",
        help="model file to use",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default="D:\\code\\demo-style\\data\\content_images\\zeven-bw\\zeven",
        help="The path to the folder that contains the images (frames)",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA (GPU)")
    opt = parser.parse_args()
    print(opt)

    val_set = get_val_set(opt.dataset_path)

    seq_sampler = SequentialSampler(val_set)

    val_data_loader = DataLoader(
        dataset=val_set, num_workers=0, batch_size=1, shuffle=False, sampler=seq_sampler
    )

    checkpoint = torch.load(opt.model)
    netG = InpaintGenerator()
    netG.load_state_dict(checkpoint["generator"])
    netG.cuda()

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose(transform_list)

    result_dir = os.path.join(opt.dataset_path, "colored")
    os.makedirs(result_dir, exist_ok=True)

    # first previous frame needs to be zero image
    # initial = Image.new("RGB",[256,256])
    # initial = transform(initial)
    # previous = Variable(initial,volatile = True).view(1,-1,256,256)
    counter = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input_image, target, prev_frame = (
                Variable(batch[0], volatile=True),
                Variable(batch[1], volatile=True),
                Variable(batch[2], volatile=True),
            )
            if not opt.cpu:
                input_image = input_image.cuda()
                target = target.cuda()
                prev_frame = prev_frame.cuda()
            if counter != 0:
                prev_frame = tmp
            pred_input = torch.cat((input_image, prev_frame), 1)
            out = netG(pred_input)
            tmp = out

            image_name = "frame{}.png".format(str(counter).zfill(5))
            save_image(out, os.path.join(result_dir, image_name))
            print("saving:" + image_name)
            counter += 1

    make_gif(result_dir)
