from typing import Tuple, Union

from pdll.autograd import Tensor
from pdll.backend import np, support_types

class Parameter(Tensor):
    '''parameter
    '''
    def __init__(self, *shape, data: Union[support_types]=None):
        if shape:
            data = np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)
        self.grad = Tensor(np.zeros(self.shape))
