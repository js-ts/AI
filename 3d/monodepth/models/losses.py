import torch
import torch.nn as nn 
import torch.nn.functional as F 


class SSIM(nn.Module):
    '''
    https://ece.uwaterloo.ca/~z70wang/research/ssim/
    structure similarity index
    '''
    def __init__(self, sizes=3, padding=1):
        super().__init__()

        self.padding = nn.ReflectionPad2d(padding)
        self.pooling = nn.AvgPool2d(kernel_size=sizes, stride=1, )

        self.l = 1
        self.c1 = (0.01 * self.l) ** 2
        self.c2 = (0.03 * self.l) ** 2

    def forward(self, x, y):
        '''
        '''
        x = self.padding(x)
        y = self.padding(y)

        mu_x = self.pooling(x)
        mu_y = self.pooling(y)

        sig_x = self.pooling(x ** 2) - mu_x ** 2
        sig_y = self.pooling(y ** 2) - mu_y ** 2
        sig_xy = self.pooling(x * y) - mu_x * mu_y

        n = (2 * mu_x * mu_y + self.c1) * (2 * sig_xy + self.c2)
        d = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sig_x + sig_y + self.c2)
        v = (1 - n / d) / 2

        return torch.clamp(v, 0, 1)


def reprojection_loss(pred, target, use_ssim=True):
    '''
    '''
    l1loss = F.l1_loss(pred, target)

    if use_ssim:
        pass

    else:
        return l1loss
