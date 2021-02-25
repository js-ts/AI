from typing import Tuple, Union

from pdll.autograd import Tensor
from pdll.backend import executor

class Parameter(Tensor):
    '''parameter
    '''
    def __init__(self, *shape, data: Union[executor.support_types]=None):
        if shape:
            data = executor.np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)
        self.grad = Tensor(executor.np.zeros(self.shape))
