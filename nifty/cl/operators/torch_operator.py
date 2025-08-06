# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
#
# Author: Philipp Arras

from functools import partial
from types import SimpleNamespace
from warnings import warn

import numpy as np

from ..any_array import AnyArray, device_available
from ..domain_tuple import DomainTuple
from ..multi_domain import MultiDomain
from .energy_operators import LikelihoodEnergyOperator
from .linear_operator import LinearOperator
from .operator import Operator, is_linearization
from .simple_linear_operators import VdotOperator

__all__ = ["TorchOperator"]


def _torch2anyarray(obj):
    if isinstance(obj, dict):
        return {kk: _torch2anyarray(vv) for kk, vv in obj.items()}
    elif device_available():
        import cupy
        cu = cupy.from_dlpack(obj)
        return AnyArray(cu)
    else:
        return AnyArray(np.array(obj))


def _anyarray2torch(obj):
    import torch
    if isinstance(obj, AnyArray):
        if obj.device_id == -1:
            return torch.from_numpy(obj._val)
        else:
            return torch.from_dlpack(obj._val) # .toDlpack()
    elif isinstance(obj, dict):
        return {kk: _anyarray2torch(vv) for kk, vv in obj.items()}


class TorchOperator(Operator):
    """Wrap a pytorch function as nifty operator.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        Domain of the operator.

    target : DomainTuple or MultiDomain
        Target of the operator.

    func : callable
        The torch function that is evaluated by the operator. If `domain` is a
        `MultiDomain`, `func` takes a `dict` as argument and like-wise for the
        target.
    """
    def __init__(self, domain, target, func):
        from torch import compile
        from torch.autograd.functional import vjp, jvp

        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._func = compile(func)
        self._bwd = compile(lambda x, cot: vjp(func, x, v=cot)[1])
        self._fwd = lambda x, tan: jvp(func, x, tan)[1]

    def apply(self, x):
        from ..sugar import makeField
        self._check_input(x)
        if is_linearization(x):
            # TODO: Adapt the Linearization class to handle value_and_grad
            # calls. Computing the pass through the function thrice (once now
            # and twice when differentiating) is redundant and inefficient.
            res = self._func(x.val.asnumpy())
            bwd = partial(self._bwd, _anyarray2torch(x.val.val))
            fwd = partial(self._fwd, _anyarray2torch(x.val.val))
            jac = TorchLinearOperator(self._domain, self._target, fwd, func_T=bwd)
            return x.new(makeField(self._target, _torch2anyarray(res)), jac)
        res = _torch2anyarray(self._func(_anyarray2torch(x.val)))
        if isinstance(res, dict):
            if not isinstance(self._target, MultiDomain):
                raise TypeError(("Torch function returns a dictionary although the "
                                 "target of the operator is a DomainTuple."))
            if set(res.keys()) != set(self._target.keys()):
                raise ValueError(("Keys do not match:\n"
                                  f"Target keys: {self._target.keys()}\n"
                                  f"Torch function returns: {res.keys()}"))
            for kk in res.keys():
                self._check_shape(self._target[kk].shape, res[kk].shape)
        else:
            if isinstance(self._target, MultiDomain):
                raise TypeError(("Torch function does not return a dictionary "
                                 "although the target of the operator is a "
                                 "MultiDomain."))
            self._check_shape(self._target.shape, res.shape)
        return makeField(self._target, res)

    @staticmethod
    def _check_shape(shp_tgt, shp_torch):
        if shp_tgt != shp_torch:
            raise ValueError(("Output shapes do not match:\n"
                              f"Target shape is\t\t{shp_tgt}\n"
                              f"Torch function returns\t{shp_torch}"))

    # def _simplify_for_constant_input_nontrivial(self, c_inp):
    #     func2 = lambda x: self._func({**x, **c_inp.asnumpy()})
    #     dom = {kk: vv for kk, vv in self._domain.items() if kk not in c_inp.keys()}
    #     return None, TorchOperator(dom, self._target, func2)


class TorchLinearOperator(LinearOperator):
    """Wrap a torch function as nifty linear operator.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        Domain of the operator.

    target : DomainTuple or MultiDomain
        Target of the operator.

    func : callable
        The torch function that is evaluated by the operator. It has to be
        implemented in terms of `torch.numpy` calls. If `domain` is a
        `MultiDomain`, `func` takes a `dict` as argument and like-wise for the
        target.

    func_T : callable
        The torch function that implements the transposed action of the operator.
        If None, torch computes the adjoint. Note that this is *not* the adjoint
        action. Default: None.

    domain_dtype:
        Needs to be set if `func_transposed` is None. Otherwise it does not have
        an effect. Dtype of the domain. If `domain` is a `MultiDomain`,
        `domain_dtype` is supposed to be a dictionary. Default: None.

    Note
    ----
    It is the user's responsibility that func is actually a linear function. The
    user can double check this with the help of
    `nifty.cl.extra.check_linear_operator`.
    """
    def __init__(self, domain, target, func, domain_dtype=None, func_T=None):
        from torch import compile as jit

        from ..sugar import makeDomain
        domain = makeDomain(domain)
        if domain_dtype is not None and func_T is None:
            if isinstance(domain, DomainTuple):
                inp = SimpleNamespace(shape=domain.shape, dtype=domain_dtype)
            else:
                inp = {kk: SimpleNamespace(shape=domain[kk].shape, dtype=domain_dtype[kk])
                       for kk in domain.keys()}
            # TODO: compute func_T with pytorch
            # func_T = jit(lambda x: jax.linear_transpose(func, inp)(x)[0])
            raise NotImplementedError()
        elif domain_dtype is None and func_T is not None:
            pass
        else:
            raise ValueError("Either domain_dtype or func_T have to be not None.")
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._func = func
        self._func_T = func_T
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        from ..sugar import makeField
        self._check_input(x, mode)
        if mode == self.TIMES:
            fx = self._func(_anyarray2torch(x.val))
            return makeField(self._target, _torch2anyarray(fx))
        # TODO: make conjugate part of the torch operation
        fx = self._func_T(_anyarray2torch(x.conjugate().val))
        return makeField(self._domain, _torch2anyarray(fx)).conjugate()
