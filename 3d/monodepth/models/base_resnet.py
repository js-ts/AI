import torch
import torch.nn as nn
import torchvision.models as models

import re
import numpy as np 

class ResnetBase(nn.Module):
    """
    """
    def __init__(self, name, in_channels=3, num_layers=4, pretrained=True, ):
        super().__init__()

        net = getattr(models, name)(pretrained)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net = nn.Sequential(*list(net.children())[:-2]) 

        self.net = net
        self.num_layers = num_layers
        self.out_channels_list = [64, 64, 128, 256, 512] 

    def forward(self, data):
        outputs = []
        for m in self.net.children():
            data = m(data)
            outputs.append(data)
            
        return (outputs[2:3] + outputs[4:])[-self.num_layers:]



if __name__ == '__main__':

    mm = ResnetBase('resnet18', 3, 4, True)


    data = torch.rand(1, 3, 120, 120)
    outs = mm(data)

    for out in outs:
        print(out.shape)

