
import torch
import torch.nn.functional as F 
import torchvision.transforms as transforms

import os 
import numpy as np 
from PIL import Image

import models
from utils import save_color_depth



def load_models():
    '''
    '''
    # depth encoder
    encoder = models.ResnetBase('resnet18', in_channels=3, num_layers=5, pretrained=True)
    dict_encoder = torch.load('./output/mono_640x192/encoder.pth', map_location=torch.device('cpu'))

    params = {kt: dict_encoder[k] for kt, k in zip(encoder.state_dict(), dict_encoder)} 
    encoder.load_state_dict(params, strict=True)
    print(dict_encoder['height'], dict_encoder['width'], dict_encoder['use_stereo'])

    # depth decoder
    dict_depth = torch.load('./output/mono_640x192/depth.pth', map_location=torch.device('cpu'))
    depth = models.DepthDecoder(encoder.out_channels_list)

    params = {kt: dict_depth[k] for kt, k in zip(depth.state_dict(), dict_depth)} 
    depth.load_state_dict(params, strict=True)

    # pose encoder 
    poseencoder = models.ResnetBase('resnet18', in_channels=6, num_layers=5, pretrained=True)
    dict_poseencoder = torch.load('./output/mono_640x192/pose_encoder.pth', map_location=torch.device('cpu'))
    params = {kt: dict_poseencoder[k] for kt, k in zip(poseencoder.state_dict(), dict_poseencoder)} 
    poseencoder.load_state_dict(params, strict=True)

    # pose decoder 
    posedecoder = models.PoseDecoder(poseencoder.out_channels_list)
    dict_posedecoder = torch.load('./output/mono_640x192/pose.pth', map_location=torch.device('cpu'))
    params = {kt: dict_posedecoder[k] for kt, k in zip(posedecoder.state_dict(), dict_posedecoder)} 
    posedecoder.load_state_dict(params, strict=True)

    return encoder, depth


def load_model_v1():
    
    depth = models.DepthNet('resnet18')
    pose = models.PoseNet('resnet18')

    paths = ['encoder.pth', 'depth.pth', 'pose_encoder.pth', 'pose.pth']
    netws = [depth.encoder, depth.decoder, pose.encoder, pose.decoder]
    for m, p in zip(netws, paths):
        params = torch.load(os.path.join('./output/mono_640x192/', p), map_location=torch.device('cpu'))
        params = {kt: params[k] for kt, k in zip(m.state_dict(), params)} 
        m.load_state_dict(params, strict=True)

    return depth, pose


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

    print('load_model_v1...')

    depth, pose = load_model_v1()   
    outputs = depth(data)

    disp = outputs['disp', 0]
    disp = F.interpolate(disp, (oh, ow), mode='bilinear', align_corners=False)
    disp = disp.squeeze().cpu().data.numpy()
    save_color_depth(disp, 'output/test_v1.jpg')
