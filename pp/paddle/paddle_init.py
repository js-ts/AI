
import math
import numpy as np
import functools

import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.initializer as initializer


__all__= ['uniform_', 'normal_', 'constant_', 'ones_', 'zeros_', 
          'xavier_uniform_', 'xavier_normal_',
          'kaiming_uniform_', 'kaiming_normal_',
          'reset_parameter_as_torch']

#  'Constant',
#  'KaimingNormal',
#  'KaimingUniform',
#  'Normal',
#  'TruncatedNormal',
#  'Uniform',
#  'XavierNormal',
#  'XavierUniform',


# https://pytorch.org/docs/stable/_modules/torch/nn/init.html#calculate_gain

def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor


def _no_grad_normal_(tensor, mean=0., std=1.):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def _no_grad_fill_(tensor, value=0):
    with paddle.no_grad():
        v = paddle.rand(shape=tensor.shape, dtype=tensor.dtype)
        v[...] = value
        tensor.set_value(v)
    return tensor


def uniform_(tensor, a, b):
    '''
    '''
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor, mean=0., std=1.):
    '''
    '''
    return _no_grad_normal_(tensor, mean, std)


def constant_(tensor, value=0):
    '''
    '''
    return _no_grad_fill_(tensor, value)


def ones_(tensor):
    '''
    '''
    return _no_grad_fill_(tensor, 1)


def zeros_(tensor):
    '''
    '''
    return _no_grad_fill_(tensor, 0)



def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    '''
    reverse: 
        conv: weight [cout, cin, kh, kw], 
        linear: weight [cin, cout]
        default, [cout, cin, ...]
    '''
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]
    
    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = functools.reduce(lambda a, b: a * b, tensor.shape[2:]) 
        
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


def xavier_uniform_(tensor, gain=1., reverse=False):
    '''
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -k, k)



def xavier_normal_(tensor, gain=1., reverse=False):
    '''
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0, std)
    
    

def _calculate_correct_fan(tensor, mode, reverse=False):
    '''
    '''
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)
    
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    '''
    '''
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', reverse=False):
    '''
    '''
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -k, k)


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', reverse=False):
    '''
    '''
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0, std)
    
    

@paddle.no_grad()
def reset_parameter_as_torch(model, include_self=True):
    for n, m in model.named_sublayers(include_self=include_self):
        if isinstance(m, nn.Conv2D):
            k = float(m._groups) / (m._in_channels * m._kernel_size[0] * m._kernel_size[0])
            k = math.sqrt(k)
            _no_grad_uniform_(m.weight, -k, k)
            if m.bias is not None:
                _no_grad_uniform_(m.bias, -k, k)
                
        elif isinstance(m, nn.Linear):
            k = math.sqrt(1. / m.weight.shape[0])
            _no_grad_uniform_(m.weight, -k, k)
            if m.bias is not None:
                _no_grad_uniform_(m.bias, -k, k)

        elif isinstance(m, nn.Embedding):
            _no_grad_normal_(m.weight)

        elif isinstance(m, nn.BatchNorm2D):
            # same as torch 1, 0
            pass
        

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
    

    
class _MM(nn.Layer):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 2, 1)
        self.linear = nn.Linear(10, 11)
        self.seq = nn.Sequential(nn.Linear(3, 3), nn.ReLU())

        init_parameter_as_torch(self)
        self._reset_parameters()

    def forward(self, ):
        pass

    def _reset_parameters(self, ):
        for _, m in self.named_sublayers():

            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                zeros_(m.bias)

            if isinstance(m, nn.Conv2D):
                xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=1, reverse=True)

if __name__ == '__main__':
    
    mm = MM()

