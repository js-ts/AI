from .module import Module
from .functional import op_relu, op_sigmoid, op_tanh


class Tanh(Module):
    '''tanh
    '''
    def __init__(self, ):
        pass

    def forward(self, data):
        return op_tanh()(data)[0]

    def ext_repr(self, ):
        return ''


class Simoid(Module):
    '''sigmoid
    ''' 
    def forward(self, data):
        return op_sigmoid()(data)[0]
    
    def ext_repr(self, ):
        return ''

    
class ReLU(Module):
    '''relu
    '''
    def forward(self, data):
        return op_relu()(data)[0]

    def ext_repr(self, ):
        return ''
