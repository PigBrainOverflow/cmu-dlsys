"""
For initial parameters of nn
"""

import math
from .initializer_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs) -> Tensor:
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    shape = kwargs.pop("shape")
    return rand_uniform(shape, low=-a, high=a, **kwargs)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs) -> Tensor:
    a = gain * math.sqrt(2.0 / (fan_in + fan_out))
    shape = kwargs.pop("shape")
    return rand_normal(shape, std=a, **kwargs)

def kaiming_uniform(fan_in, fan_out, **kwargs) -> Tensor:
    """
    Only support relu activation function.
    """
    # fan_in and fan_out are not equal to shape!!!
    bound = math.sqrt(6.0 / fan_in)
    shape = kwargs.pop("shape")
    return rand_uniform(shape, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, **kwargs) -> Tensor:
    """
    Only support relu activation function.
    """
    std = math.sqrt(2.0 / fan_in)
    shape = kwargs.pop("shape")
    return rand_normal(shape, std=std, **kwargs)
