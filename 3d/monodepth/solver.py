import torch
import torch.nn as nn 
import torch.optim as optim 


from dataset import KITIIDataset
from utils import misc

import models
from options import OptionsV1


class Solver(object):

    def __init__(self, options) -> None:
        super().__init__()

        im_size = (320, 320)
        
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
        pass
    
    
    def build_scheduler(self, ):
        pass
    
    
    def print_losses(self, ):
        pass
    
    
    def train_batch(self, inputs):
        pass
    
    
    def train_epoch(self, dataloader):
        pass
    
    
    def to(self, device):
        for k in dir(self):
            if isinstance(getattr(self, k), (nn.Module, torch.Tensor)):
                getattr(self, k).to(device)
        pass
    
    
    
if __name__ == '__main__':
    
    opt = OptionsV1().parse()

    solver = Solver(opt)
    solver.init()

    for inputs in solver.train_dataloader:
        print(inputs)
        
        
        solver.depthnet(inputs[('image', 0)])