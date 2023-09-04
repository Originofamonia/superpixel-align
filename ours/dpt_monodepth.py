"""
DPT model to generate depth map
"""
import os
import sys
import glob
import torch
import cv2
import argparse
from torchvision.transforms import Compose

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:])  # -1 for terminal
add_path(root_path)

import dpt.io
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


def monodepth(input_path, output_path, th, optimize=True):
    """
    th: crop ceiling ratio (<0.5)
    includes cropping depth here
    """
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model_path = f'/home/qiyuan/2023spring/DPT/weights/dpt_hybrid-midas-501f0c75.pt'
    net_w = net_h = 384
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    img_names = glob.glob(os.path.join(input_path, "*"))

    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        img = dpt.io.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            h,w = prediction.shape
            pred = prediction[int(h*th):]
            dpt.io.write_depth(filename, pred, bits=2, absolute_depth=False)
