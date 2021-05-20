import torch
from torch import random
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import random
from PIL import Image
import numpy as np 

from .utils import KittiTransforms

class KITIIDataset(data.Dataset):
    def __init__(self, dataroot, datafile, training=False):
        super().__init__()

        self.K = np.array([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        # self.im_size = (1242, 375)
        self.im_size = (640, 192)

        self.side_idx = {'2': 2, '3': 3, 'l': 2, 'r': 3}
        self.fram_idx = [0, -1, 1, 's']
        self.training = training
        # self.num_scale = 4

        self.dataroot = dataroot
        with open(datafile, 'r') as f:
            lines = f.readlines()
            self.lines = lines

        self.kitti_transforms = KittiTransforms()

    def __len__(self, ):
        return len(self.lines)

    def __getitem__(self, idx):
        '''
        ('image', 0/1/-1/'s'),
        'K':, 
        '''
        do_flip = random.random() < 0.5 and self.training

        inputs = {}
        line = self.lines[idx].split()
        if len(line) == 3:
            root, fram, side = line
            fram = int(fram)
            root = os.path.join(self.dataroot, root)
        else:
            raise NotImplementedError

        # frames
        for i in self.fram_idx:
            if i == 's':
                _side = 'r' if i == 'l' else 'l'
                inputs[("image", i)] = self.load_image(root, fram, _side, do_flip)
            else:
                inputs[("image", i)] = self.load_image(root, fram + i, side, do_flip)

        # scales 
        inputs['k'] = self.K.copy()
        for s in range(self.num_scale):
            K = self.K.copy()
            K[0, :] *= self.im_size[0] // (2 ** s)
            K[1, :] *= self.im_size[1] // (2 ** s)
            inputs[('K', s)] = K 

        # augument
        # if self.training and random.random() < 0.5:
        #     inputs = self.kitti_transforms(inputs)

        return inputs


    def load_image(self, root, idx, side, do_flip=False):
        '''
        '''
        file = os.path.join(root, 
                            'image_0{}'.format(self.side_idx[side]), 
                            'data', 
                            '{:0>10}.png'.format(idx))
        im = Image.open(file).convert('RGB')
        im = im.resize(self.im_size, Image.BILINEAR)

        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        return im 
