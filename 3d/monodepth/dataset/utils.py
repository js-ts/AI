import torch
import torch.nn as nn
import torchvision.transforms as transforms



class KittiTransforms(nn.Module):
    def __init__(self):
        super().__init__()

        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)

        self.to_tensor = transforms.ToTensor()

        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        ])

    def forward(self, inputs):
        '''
        inputs (dict), {('image', 1), Image}
        '''
        _keys = [k for k in inputs if 'image' in k] 
        _images = [self.to_tensor(inputs[k]).unsqueeze(0) for k in _keys] # [1 c h w]

        _images = torch.cat(_images, dim=0)
        _images = self.transform(_images)
        _images = torch.chunk(_images, chunks=len(_keys), dim=0)

        inputs.update(dict(zip(_keys, _images)))

        return inputs
        
