
import math

from pdll.autograd import Tensor

from ..functional import softmax
from .module import Module


class Attention(Module):
    '''Attention Is All You Need
    '''
    def __init__(self, ):
        pass

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        pass


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        pass
