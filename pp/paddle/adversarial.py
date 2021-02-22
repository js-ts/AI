

import paddle
import numpy as np  

mm = paddle.nn.Linear(2, 2)

_data = np.array([[2, 3], [4, 5]], dtype='float32')
data = paddle.to_tensor(_data)

data.stop_gradient = False
for n, p in mm.named_parameters():
    p.stop_gradient = False   


for _ in range(3):

    for i in range(2):     
        out = mm(data)
        out = (out ** 2).mean()
        out.backward(retain_graph=True)

        if i == 0:
            # update data
            data = data - 0.05 * paddle.to_tensor(data.grad)
        else:
            # optimizer(parameters)
            pass

        print(i, data.grad)

        for _, p in mm.named_parameters():
            p.clear_grad()

        data.clear_grad()


