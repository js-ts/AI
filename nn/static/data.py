import numpy as np 
from typing import Iterator, Tuple

from .tensor import Tensor


class BatchIterator:
    def __init__(self, bz: int=32, shuffle: bool=True) -> None:
        self.bz = bz
        self.shuffle = shuffle
    
    def __call__(self, inputs: Tensor, labels: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        starts = np.arange(0, len(inputs), self.bz)
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.bz
            _inputs = inputs[start: end]
            _labels = labels[start: end]
            yield _inputs, _labels
            