import _init_path

import pdll 
import pdll.autograd
import pdll.nn
import pdll as L 

import torch
import numpy as np
import unittest

import os 

class Testing(unittest.TestCase):

    def test_eq(self, ):
        ''' 
        '''
        a = L.randn(2, 3)
        b = L.randn(2, 3)
        data = np.random.randn(2, 3)
        # print(a, data)

        print(isinstance(a, L.Tensor), isinstance(b, L.Tensor))

        c = a == True
        d = a < b
        e = a == data 
        print(c)
        print(d)
        print(e)




if __name__ == '__main__':
    
    unittest.main(verbosity=1)

