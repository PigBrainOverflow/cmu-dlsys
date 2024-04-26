"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, shape=(in_features, out_features)))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, shape=(1, out_features)))

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))

        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size, *dims = X.shape
        new_dim = math.prod(dims)
        return X.reshape((batch_size, new_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        return ops.relu(x)

        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        res = x
        for module in self.modules:
            res = module(res)
        return res

        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION

        batch_size, num_classes = logits.shape
        one_hot_y = init.one_hot(num_classes, y)
        # print(logits.shape, one_hot_y.shape)
        total_loss = ops.logsumexp(logits, (1,)) - ops.summation(logits * one_hot_y, (1,))
        return ops.summation(total_loss) / batch_size

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.training = True
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            # reduce along the batch
            mean = x.sum((0, )) / batch_size
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - mean_broadcast) ** 2).sum((0, )) / batch_size
            # print(mean, var)
            # update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean_broadcast = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var

        var_broadcast = var.reshape((1, self.dim)).broadcast_to(x.shape)
        weight_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        y = weight_broadcast * (x - mean_broadcast) / ((var_broadcast + self.eps) ** 0.5) + bias_broadcast

        return y

        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, k = x.shape
        mean = x.sum(axes=(1, )) / self.dim
        mean = mean.reshape((n, 1)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=(1, )) / self.dim
        var = var.reshape((n, 1)).broadcast_to(x.shape)
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias
        ### END YOUR SOLUTION


# class LayerNorm1d(Module):
#     def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         ### BEGIN YOUR SOLUTION

#         self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
#         self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)

#         ### END YOUR SOLUTION

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION

#         print(x)
#         batch_size = x.shape[0]
#         e_x = ops.summation(x, (1,)) / self.dim
#         # Var[X] = E[X^2] - E[X]^2
#         e2_x = ops.summation(x * x, (1,)) / self.dim
#         var_x = e2_x - e_x * e_x
#         print(e_x, var_x)

#         # broadcast
#         d = self.dim
#         e_x_b = e_x.reshape((batch_size, 1)).broadcast_to((batch_size, d))
#         var_x_b = var_x.reshape((batch_size, 1)).broadcast_to((batch_size, d))
#         w_b = self.weight.reshape((1, d)).broadcast_to((batch_size, d))
#         b_b = self.bias.reshape((1, d)).broadcast_to((batch_size, d))

#         y = w_b * ((x - e_x_b) / (var_x_b + self.eps) ** 0.5) + b_b

#         return y

#         ### END YOUR SOLUTION


# This is a BatchNorm:
# class LayerNorm1d(Module):
#     def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         ### BEGIN YOUR SOLUTION

#         self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
#         self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)

#         ### END YOUR SOLUTION

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION

#         print(x)
#         batch_size = x.shape[0]
#         e_x = ops.summation(x, (0,)) / batch_size
#         # Var[X] = E[X^2] - E[X]^2
#         e2_x = ops.summation(x * x, (0,)) / batch_size
#         var_x = e2_x - e_x * e_x
#         print(e_x, var_x)

#         # broadcast
#         if not isinstance(self.dim, tuple):
#             d = (self.dim,)
#         else:
#             d = self.dim
#         e_x_b = e_x.reshape((1, *d)).broadcast_to((batch_size, *d))
#         var_x_b = var_x.reshape((1, *d)).broadcast_to((batch_size, *d))
#         w_b = self.weight.reshape((1, *d)).broadcast_to((batch_size, *d))
#         b_b = self.bias.reshape((1, *d)).broadcast_to((batch_size, *d))

#         y = w_b * ((x - e_x_b) / (var_x_b + self.eps) ** 0.5) + b_b

#         return y

#         ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p, requires_grad=False)
            return mask * x / (1.0 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
