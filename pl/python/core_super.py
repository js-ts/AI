from typing import Any, NoReturn


# MRO
# Method Resolution Order


# def super(cls, inst):
#     mro = inst.__class__.mro()
#     return mro[mro.index(cls) + 1]


class Base:
    def __init__(self, ):
        """
        """
        print('base cls')


class A(Base):
    def __init__(self, ):
        mro = type(self).mro()
        id_next = mro.index(A) + 1
        for m in mro[id_next:]:
            if hasattr(m, '__init__'):
                m.__init__(self)

        print(mro, id_next)

        # super().__init__()           # Python3
        # super(A, self).__init__()    # Python2



class X():
    def __init__(self, x):
        self.x = x
    
    def printf(self, ):
        print('x', self.x ** 2)
    
    def test(self, ):
        self.printf()
        
        
class Y():
    def __init__(self, y):
        self.y = y
    
    def printf(self, ):
        print('y', self.y ** 2)
        
    def test(self, ):
        self.printf()

class Z(X, Y):
    def __init__(self, x, y):
        super(Z, self).__init__(x=x)
        super(X, self).__init__(y=y)
    
    def printf(self, ):
        print('z', '---')
        
    
if __name__ == '__main__':
    a = A()    
    
    z = Z(1, 2)
    z.printf()
    z.test()