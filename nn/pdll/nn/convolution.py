import math
import numpy as np 

from .parameter import Parameter
from .module import Module
from .functional import op_conv2d



class Conv2d(Module):
    '''
    image: C_in H_in W_in
    kernel: C_out C_in H_kernel W_kernel
    output: C_out H_out W_out
    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.groups = groups
        
        assert in_channels % groups == 0, ''

        k = math.sqrt(1. / (groups * in_channels * kernel_size[0] * kernel_size[1]))
        weight_init = np.random.uniform(low=-k, high=k, size=(out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1]))
        self.weight = Parameter(data=weight_init)
        
        if bias:
            bias_init = np.random.uniform(low=-k, high=k, size=(self.out_channels, ))
            self.bias = Parameter(data=bias_init)
        else:
            self.bias = None

    def forward(self, data):
        return op_conv2d(self.kernel_size, self.stride, self.padding)(data, self.weight, self.bias)[0]

    def ext_repr(self, ):
        return f'({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

