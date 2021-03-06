# PDLL😊
> including autograd, nn modules, optimizer, io, .etc.


## Getting Started

```python
In [1]: import pdll as L

In [2]: a = L.rand(2, 2, 3, requires_grad=True)
In [3]: b = L.rand(3, 3)
In [4]: c = a @ b + L.rand(2, 3)
In [5]: d = (c ** 2) * 2 - L.ones_like(c)
In [6]: d.mean().backward()

In [7]: a.grad
Out[7]: 
Tensor([[[0.71458999 0.87984239 0.73015823]
  [0.76491385 1.04176047 0.89780678]]
  
 [[1.07044392 1.33654949 1.12387667]
  [0.74022419 1.01408228 0.86196845]]])

```

- [More Examples](./examples/)
- [Completed Functions](./docs/todolist.md)


Name | Performance | Commits 
---|---|---
[mnist](./examples/mnist.py) | acc=0.99 | an exmaple midified from pytorch [mnist](https://github.com/pytorch/examples/tree/master/mnist), but with new network achitecture.


## About PDLL

PDLL is Python Deep Learning Library, a backend-agnostic python deep learning framework, pluging and playing for different numpy-like computing engines.

Module | Description
---|---
[pdll.backend]() | a numpy-like library, including types and operations
[pdll.autograd]() | an automatic differentiation library, including Tensor type, Function which recording computing history, and Engine for backpropagation 
[pdll.nn]() | a neural network library based on autograd
[pdll.optim]() | an optimizer library for deep learning
[pdll.io]() | dataset, dataloader and serialization

To learn more about contributing to PDLL, please contact me.

## About Me
- Email
- LinkedIn


## Citation
> caffe  
> pytorch