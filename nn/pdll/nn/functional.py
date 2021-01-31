
import numpy as np 
import math
from typing import Tuple
from ..autograd import Function, Tensor


class op_sigmoid(Function):
    """sigmoid
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = 1. / (1. + np.exp(-t)) 
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.out * (1. - self.out)


class op_relu(Function):
    """relu 
    """ 
    def forward(self, t: Tensor) -> Tensor:
        self.mask = t > 0
        return t * self.mask
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.mask


class op_tanh(Function):
    """
    formul: (exp(x) + exp(-x)) / (exp(x) - exp(-x))
    derive : 1 - tanh(x) ** 2
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.tanh(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad  * (1 - self.out ** 2)



# 

def im2col(data: Tensor, kernel: Tuple[int], stride: Tuple[int], padding: Tuple[int], dilation: int=1):
    '''im2col
    N C H W -> n h w c k k
    '''
    n, c, h, w = data.shape
    out_h = math.floor((h + 2 * padding[0] - dilation * (kernel[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((w + 2 * padding[1] - dilation * (kernel[1] - 1) - 1) / stride[1] + 1)

    # hpad = (padding[0]//2, padding[0] - padding[0]//2)
    # wpad = (padding[1]//2, padding[1] - padding[1]//2)
    hpad = (padding[0], padding[1])
    wpad = (padding[2], padding[3])
    data = np.pad(data, pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant')
    
    matrix = np.zeros((n, c, kernel[0], kernel[1], out_h, out_w))

    for i, i_data in enumerate(range(dilation * (kernel[0] - 1) + 1)[::dilation]):
        for j, j_data in enumerate(range(dilation * (kernel[1] - 1) + 1)[::dilation]):
            matrix[:, :, i, j, :, :] = data[:, :, i_data::stride[0], j_data::stride[1]][:, :, :out_h, :out_w]
        
    return matrix, out_h, out_w
    

def col2im(matrix: Tensor, shape: Tuple[int], kernel: Tuple[int], stride: Tuple[int], padding: Tuple[int], dilation: int=1):
    '''
    matrix: (n, cin, hk, wk, hout, wout)
    '''
    _, _, _, _, ho, wo = matrix.shape
    # matrix = matrix.transpose(0, 3, 4, 5, 1, 2) # (n, c, hk, wk, ho, wo)

    hpad = (padding[0], padding[1])
    wpad = (padding[2], padding[3])
    data = np.pad(np.zeros(shape), pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant',)
    _, _, H, W = data.shape
    
    for i, i_data in enumerate(range(dilation * (kernel[0] - 1) + 1)[::dilation]):
        iend = i_data + stride[0] * ho
        for j, j_data in enumerate(range(dilation * (kernel[1] - 1) + 1)[::dilation]):
            jend = j_data + stride[1] * wo
            data[:, :, i_data:iend:stride[0], j_data:jend:stride[1]] += matrix[:, :, i, j, :, :]

    return data[:, :, hpad[0]:H-hpad[1], wpad[0]:W-wpad[1]]


class op_conv2d(Function):
    '''conv
    '''
    def __init__(self, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, data: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        '''
        n c h w
        co ci kh kw
        '''
        # self.data = data
        self.weight = weight
        self.data_shape = data.shape
        
        n, cin, _, _ = data.shape
        cout, _, _, _ = weight.shape
        
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding, self.dilation) # -> n*hout*wout cin*hk*wk
        
        # matrix = matrix.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, cin * self.kernel[0] * self.kernel[1])
        # weight = weight.transpose(1, 2, 3, 0).reshape(-1, cout) # -> cin*hk*wk cout  [groups * cout/groups * cin
        # self.matrix = matrix
        # output = (matrix @ weight).reshape(n, out_h, out_w, cout).transpose(0, 3, 1, 2)

        matrix = matrix.reshape(n, self.groups, cin//self.groups, self.kernel[0], self.kernel[1], out_h, out_w)
        matrix = matrix.transpose(1, 0, 5, 6, 2, 3, 4).reshape(self.groups, n * out_h * out_w, cin//self.groups * self.kernel[0] * self.kernel[1])

        weight = weight.reshape(self.groups, cout//self.groups, cin//self.groups, self.kernel[0], self.kernel[1])
        weight = weight.transpose(0, 2, 3, 4, 1).reshape(self.groups, cin//self.groups * self.kernel[0] * self.kernel[1], cout//self.groups)
        # output = matrix.bmm(weight) # groups n*out_h*out_w cout/groups
        
        self.matrix = matrix # groups n*hout*wout cin//groups*hk*wk
        self.weight = weight # groups cin//groups*hk*wk cout//groups
        output = (matrix @ weight).transpose(1, 0, 2).reshape(n * out_h * out_w, self.groups * cout//self.groups)
        output = output.reshape(n, out_h, out_w, cout).transpose(0, 3, 1, 2)
        
        if bias is not None:
            return output + bias.reshape(1, -1, 1, 1)
        else:
            return output


    def backward(self, grad: Tensor):
        '''grad n cout hout wout
        '''
        n, cout, hout, wout = grad.shape
        # _, cin, hk, wk = self.weight.shape
        _, cin, _, _ = self.data_shape

        bias_grad = grad.sum(axis=(0, 2, 3))

        # indx_reverse = np.argsort([0, 3, 1, 2])
        # grad_reverse = grad.transpose(0, 2, 3, 1)
        # grad_reverse = grad_reverse.reshape(n * hout * wout, cout)
        
        # weight_grad = self.matrix.T @ grad_reverse # cin hk wk cout
        # weight_grad = weight_grad.reshape(cin, hk, wk, cout)
        # weight_grad = weight_grad.transpose(3, 0, 1, 2)

        # weight = self.weight.transpose(1, 2, 3, 0).reshape(-1, cout)  # -> cin*hk*wk cout
        # data_grad = grad_reverse @ weight.T # n*hout*wout cin*hk*wk
        # data_grad = data_grad.reshape(n, hout, wout, cin, hk, wk)
        # data_grad = data_grad.transpose(0, 3, 4, 5, 1, 2) # (n, cin, hk, wk, hout, wout)
        # data_grad = col2im(data_grad, self.data_shape, self.kernel, self.stride, self.padding)

        grad_reverse = grad.transpose(0, 2, 3, 1) # n, hout, wout, cout
        grad_reverse = grad_reverse.reshape(n * hout * wout, self.groups, cout//self.groups)
        grad_reverse = grad_reverse.transpose(1, 0, 2) # groups, n*hout*wout, cout//groups
        
        weight_grad = self.matrix.transpose(0, 2, 1) @ grad_reverse # bmm
        weight_grad = weight_grad.reshape(self.groups, cin//self.groups*self.kernel[0]*self.kernel[1], cout//self.groups)
        weight_grad = weight_grad.transpose(0, 2, 1).reshape(cout, cin//self.groups, self.kernel[0], self.kernel[1])

        data_grad = grad_reverse @ self.weight.transpose(0, 2, 1) # groups, n*hout*wout, cin//groups*hk*wk
        data_grad = data_grad.transpose(1, 0, 2).reshape(n * hout * wout, cin * self.kernel[0] * self.kernel[1])
        data_grad = data_grad.reshape(n, hout, wout, cin, self.kernel[0], self.kernel[1])
        data_grad = data_grad.transpose(0, 3, 4, 5, 1, 2)
        data_grad = col2im(data_grad, self.data_shape, self.kernel, self.stride, self.padding)

        return data_grad, weight_grad, bias_grad


class op_pool2d(Function):
    '''pooling
    '''
    def __init__(self, kernel, stride, padding, mode='max'):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def forward(self, data: Tensor):
        ''''''
        self.shape = data.shape
        n, c, _, _ = data.shape
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding)
        matrix = matrix.reshape(n, c, self.kernel[0] * self.kernel[1], out_h, out_w)
        self.matrix = matrix

        if self.mode.lower() == 'max': # TODO
            out = np.max(matrix, axis=2)
        elif self.mode.lower() == 'avg':
            out = np.average(matrix, axis=2)
        else:
            raise RuntimeError
    
        return out


    def backward(self, grad: Tensor):
        n, c, oh, ow = grad.shape
        grad = grad[:, :, np.newaxis, :, :]
        if self.mode.lower() == 'max':
            mask = self.matrix == np.max(self.matrix, axis=2, keepdims=True)
            grad = grad * mask
        elif self.mode.lower() == 'avg':
            grad = grad * np.ones_like(self.matrix) / (self.kernel[0] * self.kernel[1])
        else:
            raise RuntimeError

        grad = grad.reshape(n, c, self.kernel[0], self.kernel[1], oh, ow)

        return col2im(grad, self.shape, self.kernel, self.stride, self.padding)
