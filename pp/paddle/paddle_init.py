import torch

import math
import numpy as np

import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.initializer as initializer

#  'Constant',
#  'KaimingNormal',
#  'KaimingUniform',
#  'Normal',
#  'TruncatedNormal',
#  'Uniform',
#  'XavierNormal',
#  'XavierUniform',


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor


def _no_grad_normal_(tensor, mean=0, std=1):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def _no_grad_fill_(tensor, value=0):
    with paddle.no_grad():
        v = paddle.rand(shape=tensor.shape, dtype=tensor.dtype)
        v[...] = value
        tensor.set_value(v)
    return tensor


@paddle.no_grad()
def _reset_parameter_as_torch(model, include_self=True):
    for n, m in model.named_sublayers(include_self=include_self):
        if isinstance(m, nn.Conv2D):
            k = m._groups / (m._in_channels * m._kernel_size[0] * m._kernel_size[0])
            k = math.sqrt(k)
            _no_grad_uniform_(m.weight, -k, k)
            if m.bias is not None:
                _no_grad_uniform_(m.bias, -k, k)
                
        elif isinstance(m, nn.Linear):
            k = math.sqrt(1 / m.weight.shape[0])
            _no_grad_uniform_(m.weight, -k, k)
            if m.bias is not None:
                _no_grad_uniform_(m.bias, -k, k)

        elif isinstance(m, nn.Embedding):
            _no_grad_normal_(m.weight)

        elif isinstance(m, nn.BatchNorm2D):
            # same as torch 1, 0
            pass
        else:
            print(type(m))
        

class MM(nn.Layer):
    def __init__(self, ):
        super().__init__()
        
        self.embedding = nn.Embedding(100, 64, weight_attr=ParamAttr(initializer=initializer.Normal()))
        self.linear = nn.Linear(10, 8, )
        
        k = 1 / math.sqrt(8)
        weight_attr = ParamAttr(initializer=initializer.Uniform(low=-k, high=k))
        bias_attr = ParamAttr(initializer=initializer.Constant())
        self.conv2d = nn.Conv2D(3, 8, 2, 1, weight_attr=weight_attr, bias_attr=bias_attr)
        
        self.layers = nn.Sequential(nn.Conv2D(8, 32, 2, 1), nn.ReLU())
        
        _reset_parameter_as_torch(self)
        

    def forward(self, ):
        pass
    

mm = MM()