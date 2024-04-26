"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION

        return a ** self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        in1 = node.inputs[0]
        return out_grad * self.scalar * (in1 ** (self.scalar - 1))
        # return (out_grad * self.scalar * node / in1,)   # unsure about its correctness

        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION

        return a / b

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        in1, in2 = node.inputs
        return (out_grad / in2, - out_grad * in1 / (in2 ** 2))

        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return a / self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad / self.scalar,)

        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        # self.axes are the two axes to be transposed (reversing order).
        # np.transpose specifies all axes' order.
        dim = len(a.shape)
        axis1, axis2 = self.axes if self.axes is not None else (dim - 2, dim - 1)
        new_axes = list(range(dim))
        new_axes[axis1] = axis2
        new_axes[axis2] = axis1
        #new_axes = tuple(new_axes)

        return array_api.transpose(a, axes=new_axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad.transpose(self.axes))

        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return a.reshape(self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad.reshape(node.inputs[0].shape),)

        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: numpy.ndarray):
        ### BEGIN YOUR SOLUTION

        return array_api.broadcast_to(a, self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        in1 = node.inputs[0]
        #print(in1.shape, out_grad.shape)
        extended_shape = in1.shape + (1,) * (len(out_grad.shape) - len(in1.shape))
        reduced_axes = tuple(i for i in range(len(out_grad.shape)) if extended_shape[i] == 1 and out_grad.shape[i] != 1)

        #print(reduced_axes)
        return out_grad.sum(reduced_axes)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


# don't need to keep dimensions
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inter_out_grad = out_grad
        if self.axes:
            shape = list(node.inputs[0].shape)
            for axis in self.axes:
                shape[axis] = 1
            inter_out_grad = inter_out_grad.reshape(shape)
        return broadcast_to(inter_out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION

# class Summation(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, a):
#         ### BEGIN YOUR SOLUTION

#         return array_api.sum(a, axis=self.axes, keepdims=True)

#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION

#         in1 = node.inputs[0]
#         target_shape = in1.shape
#         #print(target_shape, out_grad.shape, self.axes)
#         if self.axes is None:
#             return (out_grad.broadcast_to(target_shape),)
#         med_shape = tuple(1 if i in self.axes else target_shape[i] for i in range(len(target_shape)))
#         reshaped_g = out_grad.reshape(med_shape)
#         return (reshaped_g.broadcast_to(target_shape),)

#         ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION

        return a @ b

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        in1, in2 = node.inputs
        #print(in1.shape, in2.shape, out_grad.shape)
        res1 = out_grad @ in2.transpose()
        res2 = in1.transpose() @ out_grad
        if len(res1.shape) > len(in1.shape):
            axes = tuple(range(len(res1.shape) - len(in1.shape)))
            res1 = res1.sum(axes)
        if len(res2.shape) > len(in2.shape):
            axes = tuple(range(len(res2.shape) - len(in2.shape)))
            res2 = res2.sum(axes)
        #print(res1.shape, res2.shape)
        return (res1, res2)

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return -a

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (-out_grad,)

        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.log(a)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad / node.inputs[0],)

        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.exp(a)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad * node,)

        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)

class GTZero(TensorOp):
    # an indicator for >0
    def compute(self, a):
        return a > 0

    def gradient(self, out_grad, node):
        raise NotImplementedError()


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.maximum(a, 0)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        in1 = node.inputs[0]
        indicator = GTZero()(in1)
        return (out_grad * indicator,)

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Max(TensorOp):
    def __init__(self, axes):
        self.axes = axes

    def compute(self, a):
        return array_api.max(a, axis=self.axes, keepdims=True)

def max(a, axes):
    return Max(axes)(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        arg0 = args[0]
        a = array_api.empty((len(args), arg0.size), device=arg0.device)

        # stack
        for i, arg in enumerate(args):
            a[i, :] = arg.compact().reshape((arg0.size,))

        new_axes = list(range(1, len(args) + 1))
        new_axes.insert(self.axis, 0)
        return Tensor(a.compact().reshape((len(args), *arg0.shape)).permute(new_axes))
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes = list(range(len(a.shape)))
        new_axes.pop(self.axis)
        new_axes.insert(0, self.axis)

        new_shape = list(a.shape).pop(self.axis)
        new_size = a.size / a.shape[self.axis]
        new_a = a.permute(new_axes).compact().reshape((a.shape[self.axis], new_size))
        args = [    # create reference instead of copy
            array_api.NDArray(new_a[i, :].compact().reshape(new_shape)) for i in range(a.shape[self.axis])
        ]
        return args
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
