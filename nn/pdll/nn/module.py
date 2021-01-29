import inspect
from .parameter import Parameter

class Module(object):
    '''Module
    '''
    def named_parameters(self, ):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield name, value
            elif isinstance(value, Module):
                yield from value.named_parameters()
            else:
                pass
        
    def parameters(self, ):
        for _, value in self.named_parameters():
            yield value

    def zero_grad(self, ):
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self, ):
        '''str
        '''
        s = self.__class__.__name__ + self.ext_repr()

        for n, m in inspect.getmembers(self):
            if isinstance(m, Module):
                _s = f'\n  {n} {str(m)}'
                s += _s

        return s
    
    def ext_repr(self, ):
        return ''
