import paddle

from collections import OrderedDict
import re
import json


__all__ = ['repr_string']


def _get_name(layer:paddle.nn.Layer) -> str:
    ''''''
    return layer.__class__.__name__


def repr_string(layer:paddle.nn.Layer) -> str:
    ''''''
    s = OrderedDict()

    if isinstance(layer, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        _s = ''
        _s += f'{layer._in_channels}, {layer._out_channels}, '
        _s += f'kernel_size={tuple(layer._kernel_size)}, stride={tuple(layer._stride)}, '
        if layer._padding != [0] * len(layer._padding):
            _s += f'padding={tuple(layer._padding)}, '
        if layer._dilation != [0] * len(layer._dilation):
            _s += f'dilation={tuple(layer._dilation)}, '
        if layer._groups != 1:
            _s += f'groups={layer._groups}, '
        return '{}({})'.format(_get_name(layer), _s)
    
    elif isinstance(layer, paddle.nn.Linear):
        _s = ''
        _s += f'input_dim={layer.weight.shape[0]}, output_dim={layer.weight.shape[1]}'
        return '{}({})'.format(_get_name(layer), _s)
    
    elif isinstance(layer, paddle.nn.Dropout):
        _s = ''
        _s += f'p={layer.p}'
        return '{}({})'.format(_get_name(layer), _s)
    
    elif isinstance(layer, paddle.nn.ReLU):
        _s = ''
        return '{}({})'.format(_get_name(layer), _s)
    
    elif _get_name(layer) in set(dir(paddle.nn)).difference(set(['Sequential', 'LayerList', 'ParameterList'])):
        _s = ''
        return '{}({})'.format(_get_name(layer), _s)
    
    for _name, _layer in layer.named_children():
        _s = _name
        # if isinstance(_layer, (paddle.nn.Sequential, paddle.nn.LayerList, paddle.nn.ParameterList)):
        #     _s += f' ({_get_name(_layer)})'
        s.update({_s: repr_string(_layer)})
    
    return {f'{_get_name(layer)}': s}
    # return s


def test_example():
    '''
    '''
    paddle.device.set_device('cpu')
    class MM(paddle.nn.Layer):
        '''
        '''
        def __init__(self, ):
            super(MM, self).__init__()
            
            self.f1 = paddle.nn.Linear(10, 10)
            self.conv1 = paddle.nn.Conv2D(12, 10, 3, 2, padding=1, dilation=2, groups=3)
            self.conv3d = paddle.nn.Conv3D(3, 3, 3, 2, padding=2)
            self.relu = paddle.nn.ReLU()
            self.seq1 = paddle.nn.Sequential(paddle.nn.Linear(10, 10), paddle.nn.Conv2D(3, 10, 3, 2)) 
            self.seq2 = paddle.nn.Sequential(self.seq1, paddle.nn.Linear(10, 10), paddle.nn.Conv2D(3, 10, 3, 2))
            self.mlist = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(3)])
            self.dropout = paddle.nn.Dropout(p=0.5)
            
        def forward(self, ):
            '''
            '''
            pass

        def __repr__(self,):
            ''''''
            # s = repr_string(self)
            return json.dumps(repr_string(self), indent=3)
    
    print()
    print(MM())

if __name__ == '__main__':
    test_example()