
from typing import Any, NoReturn

# __setattr__ 
# __getattr__ 
# __delattr__


# __class__ 
class TENSOR():
    def __init__(self, data: Any , requires_grad: bool=False) -> NoReturn:
        self.data = data
        self.grad = None
        print(self.__class__, type(self))

        if requires_grad:
            self.zero_grad()
            
    def zero_grad(self, ) -> NoReturn:
        print(self.__class__)
        # self.grad = self.__class__(0) # RecursionError: maximum recursion depth exceeded while calling a Python object
        self.grad = TENSOR(0)
        

class PARAMETER(TENSOR):
    def __init__(self, data):
        data = data 
        super().__init__(data, requires_grad=True)



if __name__ == '__main__':
    # a = A()    

    t = TENSOR(0)
    p = PARAMETER(0)