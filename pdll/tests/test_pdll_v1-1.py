import _init_path

import pdll as L 
import torch
import numpy as np


mm = L.nn.Linear(4, 10)
a = L.randn(2, 3, requires_grad=True)
b = L.randn(3, 4, requires_grad=True)

c = a @ b
c = mm(c)

c.sum().backward()

print(a.grad)
print(b.grad)
mm.zero_grad()