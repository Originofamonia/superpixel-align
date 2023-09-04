# pip install segments-ai
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segments.utils import get_semantic_bitmap
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset


def uc_challenging():
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('d72b743e4abeed82eb7ffc10221a4422f7059318')
    release = client.get_release('anqiyuan/uc', 'v0.1') # Alternatively: release = 'flowers-v1.0.json'
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export to COCO panoptic format
    export_dataset(dataset, export_format='coco-panoptic')



    for sample in dataset:
        # Print the sample name and list of labeled objects
        print(sample['name'])
        print(sample['annotations'])
        
        # # Show the image
        # plt.imshow(sample['image'])
        # plt.show()
        
        # # Show the instance segmentation label
        # plt.imshow(sample['segmentation_bitmap'])
        # plt.show()
        
        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        # plt.imshow(semantic_bitmap)
        # plt.savefig(f"output/uc_challenging/{sample['name'].split('.')[0]}.png")
        scaled_map = (semantic_bitmap*255).astype(np.uint8)
        cv2.imwrite(f"output/uc_challenging/{sample['name'].split('.')[0]}.png", scaled_map)
        np.save(f"output/uc_challenging/{sample['name'].split('.')[0]}", semantic_bitmap)
    

def uc_positive():
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('d72b743e4abeed82eb7ffc10221a4422f7059318')
    release = client.get_release('anqiyuan/uc_positive', 'v0.1') # Alternatively: release = 'flowers-v1.0.json'
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export to COCO panoptic format
    export_dataset(dataset, export_format='coco-panoptic')



    for sample in dataset:
        # Print the sample name and list of labeled objects
        print(sample['name'])
        print(sample['annotations'])
        
        # # Show the image
        # plt.imshow(sample['image'])
        # plt.show()
        
        # # Show the instance segmentation label
        # plt.imshow(sample['segmentation_bitmap'])
        # plt.show()
        
        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        # plt.imshow(semantic_bitmap)
        # plt.savefig(f"output/uc_challenging/{sample['name'].split('.')[0]}.png")
        scaled_map = (semantic_bitmap*255).astype(np.uint8)
        cv2.imwrite(f"output/uc_positive/{sample['name'].split('.')[0]}.png", scaled_map)
        np.save(f"output/uc_positive/{sample['name'].split('.')[0]}", semantic_bitmap)


if __name__ == '__main__':
    # uc_challenging()
    uc_positive()
