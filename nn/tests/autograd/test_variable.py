import unittest 
import numpy as np 

import pdll

class Testing(unittest.TestCase):
    
    def test_add(self, ):
        
        a = np.random.rand(2, 2, 3)
        b = np.random.rand(1, 3)

        var_a = Variable(a)
        var_b = Variable(b)

