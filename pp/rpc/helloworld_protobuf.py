import numpy as np 
import base64

import helloworld_pb2

tmp = helloworld_pb2.ModelRequest()
tmp.a = 1
tmp.b = 2
tmp.data = base64.b64encode(np.random.rand(2, 2))
print(str(tmp))

tmp1 = helloworld_pb2.ModelRequest()
tmp1.ParseFromString(tmp.SerializeToString())
print('tmp1', str(tmp1))