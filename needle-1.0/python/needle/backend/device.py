import numpy as np
from types import ModuleType

class Device:
    pass


class CPU(Device):
    """
    Same as numpy
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "CPU"

    # device operators
    # directly call numpy
    def fill(self, a, value):
        np.ndarray.fill(a, value)

    def ewise_add(self, a, other, out):
        np.add(a, other, out)

    scalar_add = ewise_add

    def ewise_sub(self, a, other, out):
        np.subtract(a, other, out)

    scalar_sub = ewise_sub

    def ewise_rsub(self, a, other, out):
        np.subtract(other, a, out)

    scalar_rsub = ewise_rsub

    def ewise_mul(self, a, other, out):
        np.multiply(a, other, out)

    scalar_mul = ewise_mul

    def ewise_div(self, a, other, out):
        np.divide(a, other, out)

    scalar_div = ewise_div

    def scalar_pow(self, a, other, out):
        np.power(a, other, out)

    def ewise_maximum(self, a, other, out):
        np.maximum(a, other, out)

    scalar_maximum = ewise_maximum

    def ewise_eq(self, a, other, out):
        bool_out = np.equal(other)
        out[:] = bool_out.astype(np.float32)

    scalar_eq = ewise_eq

    def ewise_ge(self, a, other, out):
        bool_out = np.greater_equal(a, other)
        out[:] = bool_out.astype(np.float32)

    scalar_ge = ewise_ge

    def ewise_ne(self, a, other, out):
        bool_out = np.not_equal(a, other)
        out[:] = bool_out.astype(np.float32)

    scalar_ne = ewise_ne

    def ewise_gt(self, a, other, out):
        bool_out = np.greater(a, other)
        out[:] = bool_out.astype(np.float32)

    scalar_gt = ewise_gt

    def ewise_lt(self, a, other, out):
        bool_out = np.less(a, other)
        out[:] = bool_out.astype(np.float32)

    scalar_lt = ewise_lt

    def ewise_le(self, a, other, out):
        bool_out = np.less_equal(a, other)
        out[:] = bool_out.astype(np.float32)

    scalar_le = ewise_le

    def log(self, a, out):
        np.log(a, out)

    def exp(self, a, out):
        np.exp(a, out)

    def tanh(self, a, out):
        np.tanh(a, out)

    def neg(self, a, out):
        np.negative(a, out)

    def matmul(self, a, other, out):
        np.matmul(a, other, out)

    def reduce_sum(self, a, out, reduce_size):
        # a is already compact
        # just need to reduce over last axis
        np.sum(a.view(np.ndarray), a.ndim - 1, None, out)

    def reduce_max(self, a, out, reduce_size):
        np.max(a.view(np.ndarray), a.ndim - 1, out)

    def rand_uniform(self, a, low: float, high: float):
        a[:] = np.random.uniform(low, high, a.shape)

    def rand_normal(self, a, mean: float, std: float):
        a[:] = np.random.normal(mean, std, a.shape)


class GPU(Device):
    """
    CUDA
    """

    # singleton
    _instance = None
    # fields
    _module: ModuleType

    def __new__(cls):
        if cls._instance is None:   # first load
            cls._instance = super().__new__(cls)
            try:
                from . import backend_ndarray_cuda
                cls._instance._module = backend_ndarray_cuda
            except ImportError:
                print("CUDA Not Found!")
        return cls._instance

    def __repr__(self):
        return "GPU"

    def ewise_add(self, a, other, out):
        

def cpu():
    return CPU()

def gpu():
    return GPU()

def default_device():
    return CPU()