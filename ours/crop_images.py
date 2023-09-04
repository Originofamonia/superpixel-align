"""
1. exclude ceilings
2. DPT to generate pfm format depth/pgm
3. generate road masks using superpixel-align
"""
import glob
import os
import sys
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import cv2

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:])  # -1 for terminal
add_path(root_path)

from dpt_monodepth import monodepth
from road_mask import generate_mask


def crop():
    """
    TODO: do DPT first, then crop
    """
    img_folder = f"data/liga/"
    files = []
    for root, _, filenames in os.walk(img_folder):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    # 1. DPT
    # cropped_path = f'output/cropped/'
    output_path = f'output/dpt/'
    # 2. crop ceiling
    th = 0.4  # ceiling height ratio, should be < 0.5
    monodepth(img_folder, output_path, th, True)  # cropped depth inside

    # 2. crop ceiling of original images
    cropped_files = []
    for i, f in enumerate(files):
        base_name = os.path.basename(f)
        img = np.array(Image.open(f))
        h, w, ch = img.shape
        cropped_img = img[int(h*th):]
        # cropped_filename = f'cropped_{}'
        # plt.imshow(cropped_img)
        cropped_f = f'output/cropped/{base_name}'
        cv2.imwrite(cropped_f, cropped_img.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cropped_files.append(cropped_f)
    # 3. generate road masks
    mask_path = f'output/mask/'
    generate_mask(mask_path, cropped_files)

    # return files


if __name__ == '__main__':
    crop()
