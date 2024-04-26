"""
Generate random Tensor from NDArray.
"""

from ..backend import NDArray, Device, default_device
from ..auto_grad import Tensor

import numpy as np


def rand_uniform(shape, low=0.0, high=1.0, device=None, dtype=np.float32, requires_grad=False) -> Tensor:
    device = default_device() if device is None else device
    array = NDArray.make(shape, dtype, device)
    array.rand_uniform(low, high)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def rand_normal(shape, mean=0.0, std=1.0, device=None, dtype=np.float32, requires_grad=False) -> Tensor:
    device = default_device() if device is None else device
    array = NDArray.make(shape, dtype, device)
    array.rand_normal(mean, std)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def constant(shape, c=1.0, device=None, dtype=np.float32, requires_grad=False) -> Tensor:
    device = default_device() if device is None else device
    array = NDArray.make(shape, dtype, device)
    array.fill(c)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def rand_binary(shape, p=0.5, device=None, dtype=np.float32, requires_grad=False):
    device = default_device() if device is None else device
    array = NDArray.make(shape, np.float32, device)
    array.rand_uniform(0.0, 1.0)
    array = array < p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
