"""
basic data structures like Tensor & Op, supporting auto differentiation
"""

from __future__ import annotations  # to support delayed annotations

# True: compute after construct
# False: compute while construct
LAZY_MODE = False

from .backend import NDArray, Device, cpu, gpu, default_device
# from . import initializer

import numpy as np
from typing import Tuple, List, Dict, Set

class Op:
    """
    A node in computational graph.
    Base class for ops.
    """

    def __call__(self, *args) -> Tensor:
        return Tensor.make_from_op(self, args)

    def compute(self, *args: Tuple[NDArray]) -> NDArray:
        """
        Calculate forward pass of operator.
        Parameters
        ----------
        input:
            a list of input arrays to the function
        Returns
        -------
        output:
            output array of the operation
        """
        raise NotImplementedError()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor | Tuple[Tensor]:
        """
        Compute partial adjoint for each input tensor for a given output adjoint.
        Parameters
        ----------
        out_grad:
            adjoint to the output tensor
        node:
            value node of forward evaluation
        Returns
        -------
        input_grads:
            a list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        """
        Convenience method to always return a tuple from gradient call.
        """
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class Tensor:
    """
    An edge in computational graph.
    """

    # trace of computational graph
    op: Op | None
    inputs: List[Tensor]
    # The following fields are cached fields for dynamic computation.
    cached_data: NDArray
    requires_grad: bool
    grad: Tensor

    def realize_cached_data(self) -> NDArray:
        """
        Run compute to realize the cached data.
        """
        # avoid recomputation
        if self.cached_data is None:
            self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def _init(
        self,
        op: Op | None,
        inputs: List[Tensor],
        *,
        num_outputs: int =1,
        cached_data: NDArray =None,
        requires_grad: bool | None =None
    ):
        """
        A helper function for initialization.
        """
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def __init__(
        self,
        array,
        *,
        device: Device | None =None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(array.numpy(), dtype=dtype, device=device)
        else:   # from NDArray / numpy
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, dtype=dtype, device=device)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        """
        Convert numpy array to NDArray.
        """
        return NDArray(numpy_array, dtype=dtype, device=device)

    @staticmethod
    def make_from_op(op: Op, inputs: List[Tensor]) -> Tensor:
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False) -> Tensor:
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self) -> Tensor:
        """
        Same as detach().
        """
        return self.detach()

    @data.setter
    def data(self, value):
        """
        Share data with value.
        """
        assert isinstance(value, Tensor) and value.dtype == self.dtype
        self.cached_data = value.realize_cached_data()

    def detach(self) -> Tensor:
        """
        Create a new tensor that shares the data but detaches from the graph.
        """
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return self.realize_cached_data().device

    def backward(self, out_grad=None):
        from .initializer import constant
        out_grad = out_grad if out_grad else constant(*self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self) -> np.ndarray:
        data = self.realize_cached_data()
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            from .ops import EWiseAdd
            return EWiseAdd()(self, other)
        else:
            from .ops import AddScalar
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            from .ops import EWiseMul
            return EWiseMul()(self, other)
        else:
            from .ops import MulScalar
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            from .ops import EWisePow
            return EWisePow()(self, other)
        else:
            from .ops import PowerScalar
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            from .ops import EWiseSub
            return EWiseSub()(self, other)
        else:
            from .ops import AddScalar
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            from .ops import EWiseDiv
            return EWiseDiv()(self, other)
        else:
            from .ops import DivScalar
            return DivScalar(other)(self)

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul()(self, other)

    def matmul(self, other):
        return self @ other

    def sum(self, axes=None):
        from .ops import Summation
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        from .ops import BroadcastTo
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        from .ops import Reshape
        return Reshape(shape)(self)

    def __neg__(self):
        from .ops import Negate
        return Negate()(self)

    def transpose(self, axes=None):
        from .ops import Transpose
        return Transpose(axes)(self)

    def __gt__(self, scalar):
        from .ops import GreaterScalar
        return GreaterScalar(scalar)(self)

    def relu(self):
        from .ops import ReLU
        return ReLU()(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


##############################
####### Helper Methods #######
##############################

def compute_gradient_of_variables(output_tensor, out_grad):
    """
    Take gradient of output node with respect to each node in node_list.
    Store the computed result in the grad field of each Variable.
    """
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    # Traverse graph in reverse topological order given the output_node that we are taking gradient.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        if node.requires_grad:
            node.grad = sum_node_list(node_to_output_grads_list[node])
            # node.op is None means this node is a source node.
            if node.op is not None:
                partial_adjoints = node.op.gradient_as_tuple(node.grad, node)
                for input, partial_adjoint in zip(node.inputs, partial_adjoints):
                    if input not in node_to_output_grads_list:
                        node_to_output_grads_list[input] = []
                    node_to_output_grads_list[input].append(partial_adjoint)

def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    topo_order: List[Tensor] = []
    visited: Set[Tensor] = set()
    for cur_node in node_list:
        topo_sort_dfs(cur_node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """
    Post-order DFS.
    """
    if node not in visited:
        visited.add(node)
        for input in node.inputs:
            topo_sort_dfs(input, visited, topo_order)
        topo_order.append(node)

def sum_node_list(node_list):
    """
    Custom sum function in order to avoid creating redundant nodes in Python sum implementation.
    Python sum needs an initial value which can be redundant.
    """
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
