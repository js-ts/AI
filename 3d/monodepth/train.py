
import torch
from torch.functional import meshgrid
import torch.nn as nn 
import torch.nn.functional as F 

import os

import models
from utils import save_color_depth


def train():
    pass



def load_model(device):
    ''' load depthnet and posenet'''
    depth = models.DepthNet('resnet18')
    pose = models.PoseNet('resnet18')

    paths = ['encoder.pth', 'depth.pth', 'pose_encoder.pth', 'pose.pth']
    netws = [depth.encoder, depth.decoder, pose.encoder, pose.decoder]
    for m, p in zip(netws, paths):
        params = torch.load(os.path.join('./outputs/mono_640x192/', p), map_location=torch.device('cpu'))
        params = {kt: params[k] for kt, k in zip(m.state_dict(), params)} 
        m.load_state_dict(params, strict=True)

    return depth.to(device), pose.to(device)


def load_dataset(device):
    ''' load dataset '''
    return [torch.rand(10, 3, 320, 320).to(device) for _ in range(3)] + [torch.rand(10, 4, 4).to(device)]


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    depthnet, posenet = load_model(device)

    im_size = (320, 320)
    pix2cam = models.Pixel2Cam(im_size).to(device)
    cam2pix = models.Cam2Pixel(im_size).to(device)

    ssimloss = models.SSIM().to(device)

    im0, im1, im2, k = load_dataset(device)


    for e in range(3):

        d_outputs = depthnet(im0)
        axisangle, translate = posenet( torch.cat([im1, im0], dim=1) )

        matrix = models.params_to_matrix(axisangle[:, 0, 0], translate[:, 0, 0], True)

        loss = 0.

        for i in range(2):

            disp = d_outputs[('disp', i)]
            disp = F.interpolate(disp, im_size, mode='bilinear', align_corners=False)
            _, depth = models.disp_to_depth(disp, 1e-3, 80)

            points = pix2cam(depth, k)
            pixels = cam2pix(points, k, matrix)
            
            preds = models.reprojection(im1, pixels)

            ssim_loss = ssimloss(preds, im0)
            l1_loss = F.smooth_l1_loss(preds, im0, reduction='none')

            loss += ssim_loss * 0.85 + l1_loss * 0.15

            # print(ssim_loss.shape, l1_loss.shape)
            
            if i == 0:
                metrics = models.depth_metrics(depth.detach(), torch.rand(10, 2, 320, 320).to(device))
                print(metrics)

        loss.mean().backward()
        print(loss.mean().item())

        




