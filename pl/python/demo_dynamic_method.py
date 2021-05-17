import functools
from types import MethodType

class Tensor(object):
    def __init__(self, data):
        self.data = data
    
def register(cls):
    def decorator(method):
        @functools.wraps(method)
        # def wrap(self, *args, **kwargs):
        def wrap(*args, **kwargs):
            return method(*args, **kwargs)
        setattr(cls, method.__name__, wrap)
    return decorator


class Function(object):
    def __init__(self, ):
        self.s = 'function'

    def __call__(self, *args):
        print(self.s + ' ' + self.__class__.__name__)
        args = (t.data for t in args)
        return Tensor(self.forward(*args))

    def forward(self, *args):
        raise NotImplementedError


class Add(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


@register(Tensor)
def add(self: Tensor, o: Tensor) -> Tensor:
    return Add()(self, o)

@register(Tensor)
def __add__(self: Tensor, o: Tensor) -> Tensor:
    return Add()(self, o)

# setattr(Tensor, add.__name__, classmethod(add))
# setattr(Tensor, add.__name__, staticmethod(add))
# setattr(Tensor, add.__name__, add)


if __name__ == '__main__':

    a = Tensor(10)
    b = Tensor(20)

    c = Add()(a, b)
    print(type(c))
    print(c.data)

    d = a.add(b)
    print(d.data)

    e = a + b
    print(e.data)