"""
Modules
"""
from __future__ import annotations
from ..auto_grad import Tensor
from ..initializer import *

from typing import List
import numpy as np


class Parameter(Tensor):
    pass

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for v in value.values():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

def _child_modules(value: object) -> List[Module]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    elif isinstance(value, dict):
        modules = []
        for v in value.values():
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
    """
    Base class for all modules.
    """
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """
        Return the list of parameters in the module.
        """
        return _unpack_params(self.__dict__)

    def _children(self) -> List[Module]:
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
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=np.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(kaiming_uniform(in_features, out_features, device=device, dtype=dtype, shape=(in_features, out_features)))
        if bias:
            self.bias = Parameter(kaiming_uniform(out_features, 1, device=device, dtype=dtype, shape=(1, out_features)))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias.broadcast_to((x.shape[0], self.out_features))


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        # First dimension is Batch.
        batch_size, *dims = x.shape
        from math import prod
        new_dim = prod(dims)
        return x.reshape((batch_size, new_dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        res = x
        for m in self.modules:
            res = m(res)
        return res

# class SoftmaxLoss(Module):
#     def forward(self, logits: Tensor, y: Tensor) -> Tensor:
#         batch_size, num_classes = logits.shape
#         one_hot_y = init.one_hot(num_classes, y)
#         # print(logits.shape, one_hot_y.shape)
#         total_loss = ops.logsumexp(logits, (1,)) - ops.summation(logits * one_hot_y, (1,))
#         return ops.summation(total_loss) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype=np.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(constant(dim, 1, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(constant(dim, 0, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = constant(dim, 0, device=device, dtype=dtype)
        self.running_var = constant(dim, 1, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        if self.training:
            # reduce along the batch
            mean = x.sum((0,)) / batch_size
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - mean_broadcast) ** 2).sum((0,)) / batch_size
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

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=np.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(constant(dim, 1, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(constant(dim, 0, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        mean = x.sum(axes=(1,)) / self.dim
        mean = mean.reshape((n, 1)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=(1,)) / self.dim
        var = var.reshape((n, 1)).broadcast_to(x.shape)
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = rand_binary(x.shape, p=self.p, requires_grad=False)
            return mask * x / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
