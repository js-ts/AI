# TODO
# deepcopy
# func_params and return value
# reference

import numpy as np 

def test_reference(x: np.ndarray) -> np.ndarray:
    _x = x
    return _x

x = np.random.rand(3, 3)
y = test_reference(x)

assert x is y, 'x is y'

