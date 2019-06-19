from __future__ import print_function

import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.utils import save_image
from tqdm import tqdm

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
        "--input-path",
        dest="input_path",
        type=str,
        default="D:\\code\\demo-style\\data\\content_images\\zeven-bw\\zeven",
        help="The path to the folder that contains the images (frames)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Use CPU instead of CUDA (GPU)"
    )
    parser.add_argument(
        "--make-gif",
        dest="make_gif",
        action="store_true",
        help="Make a GIF with the output frames",
    )
    opt = parser.parse_args()

    val_set = get_val_set(opt.input_path)

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

    result_dir = os.path.join(opt.input_path, "colored")
    os.makedirs(result_dir, exist_ok=True)

    # first previous frame needs to be zero image
    # initial = Image.new("RGB",[256,256])
    # initial = transform(initial)
    # previous = Variable(initial,volatile = True).view(1,-1,256,256)
    counter = 0
    with torch.no_grad():
        for batch in tqdm(val_data_loader):
            input_image, target, prev_frame = (batch[0], batch[1], batch[2])
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
            counter += 1

    if opt.make_gif:
        print("\nMaking gif...")
        make_gif(result_dir)
