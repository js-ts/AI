
import torch
import torch.nn.functional as F 
import torchvision.transforms as transforms

import numpy as np 
from PIL import Image

import models
from utils import save_color_depth



def load_models():
    '''
    '''
    encoder = models.ResnetBase('resnet18', in_channels=3, num_layers=5, pretrained=True)
    dict_encoder = torch.load('./output/mono_640x192/encoder.pth', map_location=torch.device('cpu'))

    params = {kt: dict_encoder[k] for kt, k in zip(encoder.state_dict(), dict_encoder)} 
    encoder.load_state_dict(params, strict=True)
    print(dict_encoder['height'], dict_encoder['width'], dict_encoder['use_stereo'])


    dict_depth = torch.load('./output/mono_640x192/depth.pth', map_location=torch.device('cpu'))
    depth = models.DepthDecoder(encoder.out_channels_list)

    params = {kt: dict_depth[k] for kt, k in zip(depth.state_dict(), dict_depth)} 
    depth.load_state_dict(params, strict=True)

    return encoder, depth


def load_image(path, size):
    '''
    '''
    im = Image.open(path).convert('RGB')
    ow, oh = im.size 
    im = im.resize(size, Image.LANCZOS)

    return im, (ow, oh)


if __name__ == '__main__':
        
    encoder, depth = load_models()
    encoder.eval(); depth.eval()

    data, (ow, oh) = load_image('./data/test_image.jpg', size=(640, 192))
    data = transforms.ToTensor()(data).unsqueeze(0)

    # data = torch.ones(1, 3, 192, 640)
    data = (data - 0.45) / 0.225
    # for out in encoder(data):
    #     print(out.mean(), out.sum())
    #     pass

    outputs = depth(encoder(data))

    # for k in outputs:
    #     print(k, outputs[k].mean(), outputs[k].shape)

    disp = outputs['disp', 0]
    disp = F.interpolate(disp, (oh, ow), mode='bilinear', align_corners=False)
    disp = disp.squeeze().cpu().data.numpy()

    save_color_depth(disp, 'output/test.jpg')
