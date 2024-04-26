from __future__ import annotations  # to support delayed annotations
from .device import *

import numpy as np
from math import prod   # may not exist in python 3.7

class NDArray(np.ndarray):
    """
    A generic N-D array class supporting CPU & GPU computations.
    Based on numpy array.
    Only supports float32 for now.
    """

    _device: Device
    _device_operators = [   # contains operators to be implemented
        "fill", "rand_uniform", "rand_normal",
        "ewise_add", "scalar_add", "ewise_sub", "scalar_sub", "ewise_rsub", "scalar_rsub",
        "ewise_mul", "scalar_mul", "ewise_div", "scalar_div",
        "scalar_pow", "ewise_maximum", "scalar_maximum",
        "ewise_eq", "scalar_eq", "ewise_ge", "scalar_ge", "ewise_ne", "scalar_ne", "ewise_gt", "scalar_gt", "ewise_lt", "scalar_lt", "ewise_le", "scalar_le",
        "log", "exp", "tanh", "neg",
        "matmul",
        "reduce_sum", "reduce_max"
    ]

    def __new__(cls, a, dtype=np.float32, device=cpu(), **kwargs):
        # copy by default
        return np.array(a, dtype, **kwargs).view(cls)

    def __init__(self, a, dtype=np.float32, device=cpu()):
        self._device = device

    @staticmethod
    def make(shape, dtype=np.float32, device=cpu()) -> NDArray:
        """
        Make an empty NDArray
        """
        new_arr = np.empty(shape, dtype=dtype).view(NDArray)
        new_arr._device = device
        return new_arr

    @property
    def device(self):
        return self._device

    def __repr__(self):
        return "NDArray(" + super().__str__() + f", dtype={self.dtype}, device={self.device})"

    """
    Utility functions.
    Same as numpy.
    """
    def reshape(self, *args, **kwargs) -> NDArray:
        new_arr = super().reshape(*args, **kwargs).view(NDArray)
        new_arr._device = self.device
        return new_arr

    def flatten(self, *args, **kwargs) -> NDArray:
        new_arr = super().flatten(*args, **kwargs).view(NDArray)
        new_arr._device = self.device
        return new_arr

    def transpose(self, *args, **kwargs) -> NDArray:
        new_arr = super().transpose(*args, **kwargs).view(NDArray)
        new_arr._device = self.device
        return new_arr

    def broadcast_to(self, *args, **kwargs) -> NDArray:
        new_arr = np.broadcast_to(self, *args, **kwargs).view(NDArray)
        new_arr._device = self.device
        return new_arr

    def fill(self, value):
        """
        Fill (in place) with a constant value.
        GPU supported.
        """
        self._device.fill(self, value)

    def to(self, device) -> NDArray:
        """
        Convert between devices, using to/from numpy calls as the unifying bridge.
        """
        if device == self.device:
            return self
        else:
            return NDArray(self, device=device)

    def ascontiguousarray(self) -> NDArray:
        """
        Useful when GPU computing
        """
        new_arr = np.ascontiguousarray(self).view(NDArray)
        new_arr._device = self.device
        return new_arr

    def rand_uniform(self, low: float, high: float):
        self._device.rand_uniform(self, low, high)

    def rand_normal(self, mean: float, std: float):
        self._device.rand_normal(self, mean, std)

    def numpy(self) -> np.ndarray:
        return self.view(np.ndarray)

    # Collection of elementwise and scalar function: add, multiply, boolean, etc.

    def ewise_or_scalar(self, other, ewise_func: function, scalar_func: function) -> NDArray:
        """
        Run either an elementwise or scalar version of a function.
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape
            ewise_func(self, other, out)
        else:
            scalar_func(self, other, out)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_add, self._device.scalar_add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_sub, self._device.scalar_sub)

    def __rsub__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_rsub, self._device.scalar_rsub)

    def __mul__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_mul, self._device.scalar_mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_div, self._device.scalar_div)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self._device.scalar_power(self, other, out)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_maximum, self._device.scalar_maximum)

    def __eq__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_eq, self._device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_ge, self._device.scalar_ge)

    def __ne__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_ne, self._device.scalar_ne)

    def __gt__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_gt, self._device.scalar_gt)

    def __lt__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_lt, self._device.scalar_lt)

    def __le__(self, other):
        return self.ewise_or_scalar(other, self._device.ewise_le, self._device.scalar_le)

    # Elementwise functions
    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self._device.log(self, out)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self._device.exp(self, out)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self._device.tanh(self, out)
        return out

    def __neg__(self):
        out = NDArray.make(self.shape, device=self.device)
        self._device.neg(self, out)
        return out

    # Matrix multiplication
    def __matmul__(self, other):
        """
        Matrix multplication of two arrays.
        This requires both arrays be 2-D.
        """
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]
        m, n, p = self.shape[0], self.shape[1], other.shape[1]
        out = NDArray.make((m, p), device=self.device)
        self._device.matmul(self, other, out, m, n, p)
        return out

    # Reductions
    def reduce_view_out(self, axis, keepdims=False):
        """
        Return a view to the array set up for reduction functions and output array.
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty Reduction Axis")
        if axis is None:
            view = self.reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            # out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), dtype=self.dtype, device=self.device)
        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support single axis."
                axis = axis[0]
            view = self.transpose(tuple([a for a in range(self.ndim) if a != axis]) + (axis,))
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)]) if keepdims else tuple([s for i, s in enumerate(self.shape) if i != axis]),
                dtype=self.dtype,
                device=self.device
            )
        return view, out

    def sum(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self._device.reduce_sum(view.ascontiguousarray(), out, view.shape[-1])
        return out

    def max(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self._device.reduce_max(view.ascontiguousarray(), out, view.shape[-1])
        return out