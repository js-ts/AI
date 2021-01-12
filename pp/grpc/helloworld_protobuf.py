import numpy as np 
import base64
import json

import helloworld_pb2


data = np.random.rand(2, 2)

tmp = helloworld_pb2.ModelRequest()
tmp.a = 1
tmp.b = 2
tmp.data = base64.b64encode(data)
print(str(tmp))


tmp1 = helloworld_pb2.ModelRequest()
tmp1.ParseFromString(tmp.SerializeToString())
print('tmp1', str(tmp1))


tmp2 = helloworld_pb2.ModelRequest(data=base64.b64encode(data), a=2, b=2)
print('tmp2', str(tmp2))

# tmp3 = {'a': 1, 'b': 2, 'data': 'data'}
# print(json.dumps(tmp3))


data = helloworld_pb2.DataExample()
Blob = helloworld_pb2.DataExample.Blob

data.data = base64.b64encode(np.random.rand(1,2))
data.c = 10
data.d = 10

blob1 = data.blobs.add()
# blob1 = Blob(info='123', data=123) # wrong
blob1.info = str(123)
blob1.data = 123

data.blobs.append(Blob(info='234', data=234))
data.blobs.extend([Blob(info=str(i), data=i) for i in range(5)]) 

print(str(data))