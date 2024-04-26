from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def _max_exp_sum(self, val: NDArray):
        # keep all axes not appears in self.axes
        # if self.axes is None, max_val is a scalar
        max_val = array_api.max(val, self.axes)
        reshaped_max_val = max_val
        # suppose val.shape == [2, 3, 4, 5], self.axes=(1, 3)
        # max_val.shape == [2, 4], reshape to [2, 1, 4, 1] for broadcasting
        if self.axes:
            new_shape = list(val.shape)
            for axis in self.axes:
                new_shape[axis] = 1
            reshaped_max_val = array_api.reshape(max_val, new_shape)
        # exp_val.shape == val.shape
        exp_val = array_api.exp(
            val - array_api.broadcast_to(reshaped_max_val, val.shape)
        )
        # sum_val.shape = [2, 4] in this example
        sum_val = array_api.sum(exp_val, axis=self.axes)
        return max_val, exp_val, sum_val

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_val, exp_val, sum_val = self._max_exp_sum(Z)
        return array_api.log(sum_val) + max_val
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        in_val = node.inputs[0].numpy()
        _, exp_val, sum_val = self._max_exp_sum(in_val)
        # refer to the example given above, an extra reshape is still needed here
        if self.axes:
            new_sum_shape = list(in_val.shape)
            for axis in self.axes:
                new_sum_shape[axis] = 1
            sum_val = array_api.reshape(sum_val, new_sum_shape)
            out_grad = out_grad.reshape(new_sum_shape)
        sum_val = array_api.broadcast_to(sum_val, in_val.shape)
        out_grad = out_grad.broadcast_to(in_val.shape)
        grad = Tensor(exp_val / sum_val)
        return grad * out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

