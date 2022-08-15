# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from justpfm import justpfm
from PIL import Image
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import csv


class UniversalStereoDataset(data.Dataset):
    def __init__(self, aug_params=None, root_path: str=None, file_list_path: str=None):
        self.augmentor = None
        #self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.img_pad = None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        self.file_list = []

        with open(Path(root_path)/Path(file_list_path)) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.file_list.append(row)

        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(worker_info.id)
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)
        
    def __getitem__(self, index):
            
        index = index % len(self.file_list)
        img1 = justpfm.read_pfm(self.file_list[index][0])
        img2 = justpfm.read_pfm(self.file_list[index][1])
        disp = justpfm.read_pfm(self.file_list[index][2])
        valid = Image.open(self.file_list[index][3])
        
        print(f"left:{img1.shape} right:{img2.shape} disp:{disp.shape} valid:{np.array(valid).shape}")


        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        #if self.augmentor is not None:
        #    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            
        img1 = torch.from_numpy(img1.copy()).float()
        img2 = torch.from_numpy(img2.copy()).float()
        flow = torch.from_numpy(flow.copy()).float()
        valid = torch.from_numpy(np.array(valid).copy()).float()

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.file_list[index], img1, img2, flow, valid


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.file_list = v * copy_of_self.file_list
        return copy_of_self
        
    def __len__(self):
        return len(self.file_list)


  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = UniversalStereoDataset(args, root_path="data/ETH3D/universal_stereo_dataset_original_size", file_list_path="samples.csv")

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

