from ..auto_grad import NDArray, Tensor, Op

from typing import Tuple

class Transpose(Op):
    def __init__(self, axes: Tuple[int] =None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.transpose(self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        if self.axes is None:   # just swap last two axes
            return out_grad.transpose()
        from numpy import argsort
        to_original_axes = tuple(argsort(self.axes))
        return out_grad.transpose(to_original_axes)

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.reshape(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad.reshape(node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)

class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        input = node.inputs[0]
        extended_shape = input.shape + (1,) * (len(out_grad.shape) - len(input.shape))
        reduced_axes = tuple(i for i in range(len(out_grad.shape)) if extended_shape[i] == 1 and out_grad.shape[i] != 1)
        return out_grad.sum(reduced_axes)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

# don't need to keep dimensions
class Summation(Op):
    def __init__(self, axes: Tuple[int] =None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.sum(self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        input = node.inputs[0]
        if self.axes is not None:
            extended_shape = list(input.shape)
            for axis in self.axes:
                extended_shape[axis] = 1
            out_grad = out_grad.reshape(extended_shape)
        return out_grad.broadcast_to(input.shape)

def summation(a, axes=None):
    return Summation(axes)(a)