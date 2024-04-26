import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION

    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(low=-a, high=a, **kwargs)

    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION

    a = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(std=a, **kwargs)

    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION

    # fan_in and fan_out are not equal to shape !!!
    bound = math.sqrt(6 / fan_in)
    shape = kwargs.pop("shape")
    return rand(*shape, low=-bound, high=bound, **kwargs)

    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION

    std = math.sqrt(2 / fan_in)
    return randn(std=std, **kwargs)

    ### END YOUR SOLUTION
