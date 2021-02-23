
import math

from pdll.autograd import Variable

from ..functional import softmax
from .module import Module


class Attention(Module):
    '''Attention Is All You Need
    '''
    def __init__(self, ):
        pass

    def forward(self, q: Variable, k: Variable, v: Variable) -> Variable:
        pass


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        pass
