"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is not None:
                if param not in self.u:
                    self.u[param] = ndl.init.zeros_like(param)
                m = self.momentum
                lr = self.lr
                wd = self.weight_decay
                self.u[param] = m * self.u[param] + (1.0 - m) * (param.grad.data + wd * param.data)
                param.data = ndl.Tensor((param.data - lr * self.u[param]), dtype=param.dtype, requires_grad=False).data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        # print(ndl.autograd.TENSOR_COUNTER)
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                if param not in self.m:
                    self.m[param] = ndl.init.zeros_like(param)
                grad_and_decay = param.grad.data + self.weight_decay * param.data
                self.m[param] = self.beta1 * self.m[param] + (1.0 - self.beta1) * grad_and_decay.data
                if param not in self.v:
                    self.v[param] = ndl.init.zeros_like(param)
                self.v[param] = self.beta2 * self.v[param] + (1.0 - self.beta2) * grad_and_decay.data ** 2
                mc = self.m[param] / (1.0 - self.beta1 ** self.t)
                vc = self.v[param] / (1.0 - self.beta2 ** self.t)
                updated_param = ndl.Tensor((param.data - self.lr * mc.data / (vc.data ** 0.5 + self.eps)), dtype=param.dtype, requires_grad=False)
                param.data = updated_param.data
        # print(ndl.autograd.TENSOR_COUNTER)
        # print("-------------------------")

        ### END YOUR SOLUTION
