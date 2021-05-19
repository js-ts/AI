import torch
import torch.nn as nn 
import torch.optim as optim 

import models
from options import OptionsV1

class Solver(object):

    def __init__(self, options) -> None:
        super().__init__()

        im_size = (320, 320)

        self.im_size = im_size
        self.posenet = models.PoseNet('resnet18')
        self.depthnet = models.DepthNet('resnet18')
        self.pix2cam = models.Pixel2Cam(im_size)
        self.cam2pix = models.Cam2Pixel(im_size)
        self.ssim = models.SSIM()

        self.optimizer = None
        self.scheduler = None


    def train(self, ):
        pass


    def test(self, ):
        pass




if __name__ == '__main__':
    
    opt = OptionsV1().parse()

    solver = Solver(opt)

