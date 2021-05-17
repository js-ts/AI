import torch
import torch.nn as nn
import torch.nn.functional as F


from .resnet import ResnetBase


# DEMO
def pose_encoder(name, num_imgs=2, pretrained=True):
    '''
    '''
    return ResnetBase(name, in_channels=3 * num_imgs, num_layers=4, pretrained=pretrained)


class PoseDecoder(nn.Module):
    def __init__(self, in_channels_list, num_frames=2):
        super().__init__()

        self.posedecoder = nn.Sequential(
            nn.Conv2d(in_channels_list[-1], 256, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 6 * num_frames, 1, 1,),
        )

        self.num_frames = num_frames #TODO here to 1 will be better
        self.in_channels_list = in_channels_list

    def forward(self, features):
        '''axis-angle, translate
        '''
        out = self.posedecoder(features[-1])
        out = out.mean(dim=(2, 3))
        out = 0.01 * out.view(-1, self.num_frames, 1, 6)

        return out[..., :3], out[..., 3:]