import torch
import torch.nn as nn
import torch.nn.functional as F 



def conv3x3(in_channels, out_channels, use_refl=True):
    '''
    '''
    mode = 'reflect' if use_refl else 'zeros'
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=mode)


def block(in_channels, out_channels, inspace=True):
    '''
    '''
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.ELU(inplace=inspace),
    )


class DepthDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=1, scales=4, use_concat=True):
        super().__init__()

        self.use_concat = use_concat
        self.scales = scales
        
        hidden_dims = [16, 32, 64, 128, 256]

        self.convs = nn.ModuleDict()
        for i in range(scales, -1, -1):
            c_in = in_channels_list[-1] if i == scales else hidden_dims[i+1]
            c_out = hidden_dims[i]
            self.convs[str((i, 0))] = block(c_in, c_out)

            c_in = hidden_dims[i]
            if use_concat and i > 0:
                c_in += in_channels_list[i-1]
            c_out = hidden_dims[i]
            self.convs[str((i, 1))] = block(c_in, c_out)
        
        self.disp_conv = nn.ModuleList([conv3x3(hidden_dims[i], out_channels) for i in range(scales)])

    
    def forward(self, features):
        
        outputs = {}

        x = features[-1]
        for i in range(self.scales, -1, -1):
            x = self.convs[str((i, 0))](x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')

            if self.use_concat and i > 0:
                x = torch.cat([x, features[i - 1]], dim=1)

            x = self.convs[str((i, 1))](x)

            if i < self.scales:
                outputs[('disp', i)] = self.disp_conv[i](x).sigmoid()

        return outputs
        