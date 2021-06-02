import torch

import math

import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.initializer as initializer


embedding = nn.Embedding(100, 64, weight_attr=ParamAttr(initializer=initializer.Normal()))


#  'Constant',
#  'KaimingNormal',
#  'KaimingUniform',
#  'Normal',
#  'TruncatedNormal',
#  'Uniform',
#  'XavierNormal',
#  'XavierUniform',


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
        
        self._reset_as_torch()
        

    def _reset_as_torch(self, ):
        for n, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                k = m._groups / (m._in_channels * m._kernel_size[0] * m._kernel_size[0])
                k = math.sqrt(k)
                v = paddle.uniform(shape=m.weight.shape, dtype=m.weight.dtype, min=-k, max=k)
                m.weight.set_value(v)
                if m.bias is not None:
                    v = paddle.uniform(shape=m.bias.shape, dtype=m.bias.dtype, min=-k, max=k)
                    m.bias.set_value(v)
                    
            elif isinstance(m, nn.Linear):
                k = math.sqrt(1 / m.weight.shape[0])
                v = paddle.uniform(shape=m.weight.shape, dtype=m.weight.dtype, min=-k, max=k)
                m.weight.set_value(v)
                if m.bias is not None:
                    v = paddle.uniform(shape=m.bias.shape, dtype=m.weight.dtype, min=-k, max=k)
                    m.bias.set_value(v)
                    
            elif isinstance(m, nn.Embedding):
                v = paddle.normal(shape=m.weight.shape)
                m.weight.set_value(v)

            elif isinstance(m, nn.BatchNorm2D):
                # same as torch 1, 0
                pass
            
            
            
    

mm = MM()