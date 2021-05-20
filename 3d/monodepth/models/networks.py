
import torch
import torch.nn as nn 

from .resnet import ResnetBase
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


class Pixel2Cam(nn.Module):
    '''
    K_invert dot pixel coor mul depth -> cam coor
    '''
    def __init__(self, size):
        super().__init__()
        w, h = size

        j, i = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        pixels = torch.cat([i.unsqueeze(0), j.unsqueeze(0), torch.ones(1, h, w)], dim=0) # (x, y, 1)
        pixels = pixels.view(3, -1).unsqueeze(0)
        
        self.register_buffer('pixels', pixels) # 1, 3, h * w
        self.register_buffer('ones', torch.ones(1, 1, w * h))
        
    def forward(self, depth, k):
        '''
        depth [n, 1, h, w]
        k [n, 4, 4]
        
        p [n, 4, h, w]
        '''
        n, _, h, w = depth.shape 

        k_invert = torch.linalg.pinv(k)
        cam_points = torch.bmm(k_invert[:, :3, :3], self.pixels.repeat(n, 1, 1))
        cam_points = depth.view(n, 1, -1) * cam_points
        cam_points = torch.cat((cam_points, self.ones.repeat(n, 1, 1)), dim=1)

        return cam_points.view(n, 4, h, w)


class Cam2Pixel(nn.Module):
    '''
    K T cam coor -> pixel coor 
    '''
    def __init__(self, size):
        super().__init__()
        pass

    def forward(self, points, K, T):
        '''
        points: [n, 4, h, w]
        k: [n, 4, 4]
        T: [n, 4, 4]

        p: [n, 2, h, w]
        '''
        n, c, h, w = points.shape

        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points.view(n, c, -1))

        pix_coords = cam_points[:, :2, :] / cam_points[:, 2:, :]
        pix_coords = pix_coords.view(points.shape[0], 2, h, w)
        
        # # (-1, -1) left-top (1, 1) right-bottom
        # pix_coords = pix_coords.permute(0, 2, 3, 1)
        # pix_coords[..., 0] /= w - 1
        # pix_coords[..., 1] /= h - 1
        # pix_coords = (pix_coords - 0.5) * 2 
        
        return pix_coords
