
import torch
import torch.nn as nn 

from .base_resnet import ResnetBase
from .depth import DepthDecoder
from .pose import PoseDecoder


class DepthNet(nn.Module):
    def __init__(self, name, ):
        super().__init__()

        self.encoder = ResnetBase(name, in_channels=3, num_layers=5, pretrained=True)
        self.decoder = DepthDecoder(self.encoder.out_channels_list, )

    def forward(self, data):
        '''
        '''
        return self.decoder(self.encoder(data))


class PoseNet(nn.Module):
    def __init__(self, name, ):
        super().__init__()

        self.encoder = ResnetBase(name, in_channels=6, num_layers=1, pretrained=True)
        self.decoder = PoseDecoder(self.encoder.out_channels_list, num_frames=2)

    def forward(self, data):
        '''
        '''
        return self.decoder(self.encoder(data))
