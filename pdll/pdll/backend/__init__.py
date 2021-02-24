
import numpy as np 
support_types = (np.ndarray, np.float, np.float32, np.float64, np.int, np.bool)

assert np.int is int, ''
assert np.float is float, ''
assert np.bool is bool, ''


from .utils import set_engine
from .engine import ENGINES, register