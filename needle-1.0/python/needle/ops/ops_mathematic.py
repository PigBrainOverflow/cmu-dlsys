from ..auto_grad import NDArray, Tensor, Op

from typing import Tuple

class EWiseAdd(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)

class AddScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)

class EWiseSub(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a - b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        return out_grad, -out_grad

def sub(a, b):
    return EWiseSub()(a, b)

class EWiseMul(Op):
    def compute(self, a: NDArray, b: NDArray) ->NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs

def multiply(a, b):
    return EWiseMul()(a, b)

class MulScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)

class PowerScalar(Op):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return self.scalar * out_grad * (node.inputs[0] ** (self.scalar - 1))
        # return out_grad * self.scalar * node / node.inputs[0]   # unsure about its correctness

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWisePow(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        if not isinstance(lhs, NDArray) or not isinstance(lhs, NDArray):
            raise ValueError("Both inputs must be tensors (NDArray).")
        grad_lhs = out_grad * rhs * (lhs ** (rhs - 1))
        grad_rhs = out_grad * (lhs ** rhs) * log(lhs)
        return grad_lhs, grad_rhs

def power(a, b):
    return EWisePow()(a, b)

class EWiseDiv(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad / rhs, -lhs / (rhs ** 2)

def divide(a, b):
    return EWiseDiv()(a, b)

class DivScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class MatMul(Op):
    """
    Only support 2-D @ 2-D due to CUDA operators
    """
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad @ rhs.transpose(), lhs.transpose() @ out_grad

def matmul(a, b):
    return MatMul()(a, b)

class Negate(Op):
    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return -out_grad

def negate(a):
    return Negate()(a)

class Log(Op):
    def compute(self, a: NDArray) -> NDArray:
        return a.log()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / node.inputs[0]

def log(a):
    return Log()(a)

class Exp(Op):
    def compute(self, a: NDArray) -> NDArray:
        return a.exp()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * node

def exp(a):
    return Exp()(a)

class ReLU(Op):
    def compute(self, a: NDArray) -> NDArray:
        return a.maximum(0)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * (node.inputs[0] > 0)

def relu(a):
    return ReLU()(a)

class GreaterScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a > self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        """
        No grad for this op.
        """
        raise NotImplementedError()

def greater_scalar(a, scalar):
    return GreaterScalar(scalar)(a)