import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 


from dataset import KITIIDataset
from utils import misc

import models
from options import OptionsV1


class Solver(object):

    def __init__(self, options) -> None:
        super().__init__()

        im_size = (640, 192)
        
        self.args = options
        self.args.im_size = im_size


    def init(self, ):
        for k in dir(self.__class__):
            if k.startswith('build_'):
                getattr(self, k)()
                
        self.to(self.args.device)
        print('init solver done...')
        
        
    def build_model(self, ):
        im_size = self.args.im_size
        
        self.posenet = models.PoseNet('resnet18')
        self.depthnet = models.DepthNet('resnet18')
        
        self.pix2cam = models.Pixel2Cam(im_size)
        self.cam2pix = models.Cam2Pixel(im_size)
        self.ssim = models.SSIM()


    def build_train_dataloader(self, ):
        args = self.args
        dataset = KITIIDataset('../../../dataset/kitti/', './dataset/splits/train.txt', training=True)
        dataloader, sampler = misc.build_dataloader(dataset, args.batch_size, True, args.num_workers, True, args.distributed)
        
        self.train_dataloader = dataloader
        self.train_sampler = sampler

        return dataloader, sampler
    
    
    def build_val_dataloader(self, ):
        args = self.args
        dataset = KITIIDataset('../../../dataset/kitti/', './dataset/splits/train.txt', training=False)
        dataloader, sampler = misc.build_dataloader(dataset, args.batch_size, False, args.num_workers, False, args.distributed)
        
        self.val_dataloader = dataloader
        self.val_sampler = sampler

        return dataloader, sampler
    
    
    def build_optimizer(self, ):
        params = []
        for k in dir(self):
            if isinstance(getattr(self, k), nn.Module):
                params.extend( getattr(self, k).parameters() )
        
        self.optimizer = optim.SGD(params, lr=self.args.lr)
    
    
    def build_scheduler(self, ):
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, ], gamma=0.1)
    
    
    def print_losses(self, ):
        pass
    
    
    def train_batch(self, inputs):

        outputs = self.depthnet( inputs[('image', 0)] )
        disp = outputs[('disp', 0)]
        disp = F.interpolate(disp, self.args.im_size[::-1], mode='bilinear', align_corners=False)
        
        _, depth = models.disp_to_depth(disp, 1e-3, 80)
        points = self.pix2cam(depth, inputs[('K', 0)])
        
        for i in (-1, 1):
            if i == -1:
                _pairs = torch.cat([inputs[('image', i)], inputs[('image', 0)]], dim=1)
            else:
                _pairs = torch.cat([inputs[('image', 0)], inputs[('image', i)]], dim=1)
                
            axisangle, translate = self.posenet(_pairs)
            matrix = models.params_to_matrix(axisangle[:, 0, 0], translate[:, 0, 0], True)
            pixels = self.cam2pix(points, inputs[('K', 0)], matrix)

            outputs[('reprojection', i)] = models.reprojection(inputs[('image', i)], pixels)
            
        losses = self.compute_loss(inputs, outputs)
        

    def train_epoch(self, dataloader):
        pass
    
    
    def compute_loss(self, inputs, outputs):
        losses = {}
        for i in (-1, 1):
            losses[('reprojection', i)] = self.ssim(outputs[('pred', i)], inputs[('image', 0)])
        
        return losses
    
    
    def to(self, device):
        for k in dir(self):
            if isinstance(getattr(self, k), (nn.Module, torch.Tensor)):
                getattr(self, k).to(device)
          
        
    def set_train(self, ):
        for k in dir(self):
            if isinstance(getattr(self, k), nn.Module):
                getattr(self, k).train()

                
    def set_eval(self, ):
        for k in dir(self):
            if isinstance(getattr(self, k), nn.Module):
                getattr(self, k).eval()


if __name__ == '__main__':
    
    opt = OptionsV1().parse()

    solver = Solver(opt)
    solver.init()
    solver.set_train()
    
    for inputs in solver.train_dataloader:
        
        solver.train_batch(inputs)
        