# -*- coding: utf-8 -*-
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import glob
from natsort import natsorted


def prepare_data_path(dataset_path):
    import os
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data = natsorted(data)
    filenames = natsorted(filenames)
    return data, filenames

class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, transform=None, crop_size=None):
        super(Fusion_dataset, self).__init__()
        self.transform = transform
        self.crop_size = crop_size

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'


        self.filepath_vis, self.filenames_vis = prepare_data_path(vi_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)

        self.split = split
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        image_vis = Image.open(vis_path).convert('RGB')

        image_inf = cv2.imread(ir_path, 0)
        image_inf = Image.fromarray(image_inf).convert('RGB')


        if self.crop_size is not None:
            if self.split == 'train':
                crop_transform = transforms.RandomCrop(self.crop_size)
            else:
                crop_transform = transforms.CenterCrop(self.crop_size)
            image_vis = crop_transform(image_vis)
            image_inf = crop_transform(image_inf)

        if self.transform is not None:
            image_vis = self.transform(image_vis)
            image_inf = self.transform(image_inf)

        name = self.filenames_vis[index]
        return (image_vis, image_inf, name)

    def __len__(self):
        return self.length


