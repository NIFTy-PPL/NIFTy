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
from .energy_operators import LikelihoodEnergyOperator
from .linear_operator import LinearOperator
from .operator import Operator

__all__ = ["JaxOperator", "JaxLikelihoodEnergyOperator", "JaxLinearOperator"]


def _jax2anyarray(obj):
    if isinstance(obj, dict):
        return {kk: _jax2anyarray(vv) for kk, vv in obj.items()}
    elif device_available():
        import cupy
        cu = cupy.from_dlpack(obj)
        return AnyArray(cu)
    else:
        return AnyArray(np.array(obj))


def _anyarray2jax(obj):
    import jax
    if isinstance(obj, AnyArray):
        if obj.device_id == -1:
            return jax.numpy.array(obj._val)
        else:
            return jax.dlpack.from_dlpack(obj._val)
    elif isinstance(obj, dict):
        return {kk: _anyarray2jax(vv) for kk, vv in obj.items()}


class JaxOperator(Operator):
    """Wrap a jax function as nifty operator.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        Domain of the operator.

    target : DomainTuple or MultiDomain
        Target of the operator.

    func : callable
        The jax function that is evaluated by the operator. It has to be
        implemented in terms of `jax.numpy` calls. If `domain` is a
        `MultiDomain`, `func` takes a `dict` as argument and like-wise for the
        target.

    Note
    ----
    Contrary to the convention in the rest of nifty, this operator returns
    Fields on device if a device is available. Normally, nifty operators return
    on the same device_id as their input has been.
    """
    def __init__(self, domain, target, func):
        import jax

        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._func = jax.jit(func)
        self._bwd = jax.jit(lambda x, y: jax.vjp(func, x)[1](y)[0])
        self._fwd = jax.jit(lambda x, y: jax.jvp(self._func, (x,), (y,))[1])

    def apply(self, x):
        from ..multi_domain import MultiDomain
        from ..sugar import is_linearization, makeField
        self._check_input(x)
        if is_linearization(x):
            # TODO: Adapt the Linearization class to handle value_and_grad
            # calls. Computing the pass through the function thrice (once now
            # and twice when differentiating) is redundant and inefficient.
            res = self._func(x.val.asnumpy())
            bwd = partial(self._bwd, _anyarray2jax(x.val.val))
            fwd = partial(self._fwd, _anyarray2jax(x.val.val))
            jac = JaxLinearOperator(self._domain, self._target, fwd, func_T=bwd)
            return x.new(makeField(self._target, _jax2anyarray(res)), jac)
        res = _jax2anyarray(self._func(_anyarray2jax(x.val)))
        if isinstance(res, dict):
            if not isinstance(self._target, MultiDomain):
                raise TypeError(("Jax function returns a dictionary although the "
                                 "target of the operator is a DomainTuple."))
            if set(res.keys()) != set(self._target.keys()):
                raise ValueError(("Keys do not match:\n"
                                  f"Target keys: {self._target.keys()}\n"
                                  f"Jax function returns: {res.keys()}"))
            for kk in res.keys():
                self._check_shape(self._target[kk].shape, res[kk].shape)
        else:
            if isinstance(self._target, MultiDomain):
                raise TypeError(("Jax function does not return a dictionary "
                                 "although the target of the operator is a "
                                 "MultiDomain."))
            self._check_shape(self._target.shape, res.shape)
        return makeField(self._target, res)

    @staticmethod
    def _check_shape(shp_tgt, shp_jax):
        if shp_tgt != shp_jax:
            raise ValueError(("Output shapes do not match:\n"
                              f"Target shape is\t\t{shp_tgt}\n"
                              f"Jax function returns\t{shp_jax}"))

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        func2 = lambda x: self._func({**x, **c_inp.asnumpy()})
        dom = {kk: vv for kk, vv in self._domain.items() if kk not in c_inp.keys()}
        return None, JaxOperator(dom, self._target, func2)


class JaxLinearOperator(LinearOperator):
    """Wrap a jax function as nifty linear operator.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        Domain of the operator.

    target : DomainTuple or MultiDomain
        Target of the operator.

    func : callable
        The jax function that is evaluated by the operator. It has to be
        implemented in terms of `jax.numpy` calls. If `domain` is a
        `MultiDomain`, `func` takes a `dict` as argument and like-wise for the
        target.

    func_T : callable
        The jax function that implements the transposed action of the operator.
        If None, jax computes the adjoint. Note that this is *not* the adjoint
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
        import jax

        from ..domain_tuple import DomainTuple
        from ..sugar import makeDomain
        domain = makeDomain(domain)
        if domain_dtype is not None and func_T is None:
            if isinstance(domain, DomainTuple):
                inp = SimpleNamespace(shape=domain.shape, dtype=domain_dtype)
            else:
                inp = {kk: SimpleNamespace(shape=domain[kk].shape, dtype=domain_dtype[kk])
                       for kk in domain.keys()}
            func_T = jax.jit(lambda x: jax.linear_transpose(func, inp)(x)[0])
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
            fx = self._func(_anyarray2jax(x.val))
            return makeField(self._target, _jax2anyarray(fx))
        # TODO: make conjugate part of the jax operation
        fx = self._func_T(_anyarray2jax(x.conjugate().val))
        return makeField(self._domain, _jax2anyarray(fx)).conjugate()


class JaxLikelihoodEnergyOperator(LikelihoodEnergyOperator):
    """Wrap a jax function as nifty likelihood energy operator.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        Domain of the operator.

    func : callable
        The jax function that is evaluated by the operator. It has to be
        implemented in terms of `jax.numpy` calls. If `domain` is a
        `MultiDomain`, `func` takes a `dict` as argument and like-wise for the
        target. It needs to map to a scalar.

    transformation : Operator, optional
        Coordinate transformation to Euclidean space.

    sampling_dtype : dtype, optional
        The dtype that shall be used for drawing samples from the metric of the
        likelihood.
    """
    def __init__(self, domain, func, transformation=None, sampling_dtype=None):
        import jax

        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._func = jax.jit(func)
        self._val_and_grad = jax.jit(jax.value_and_grad(func))
        self._dt = sampling_dtype
        self._trafo = transformation
        warn("JaxLikelihoodEnergyOperator does not support normalized residuals "
             "yet. This mean that the inference works but the data residuals do "
             "not show up in e.g. `ift.minisanity`.")
        super(JaxLikelihoodEnergyOperator, self).__init__(None, None)

    def get_transformation(self):
        if self._trafo is None:
            s = self.__name__ + " was instantiated without `transformation`"
            raise RuntimeError(s)
        return self._dt, self._trafo

    def apply(self, x):
        from ..sugar import is_linearization, makeField
        from .simple_linear_operators import VdotOperator

        self._check_input(x)
        lin = is_linearization(x)
        val = x.val.val if lin else x.val
        if not lin:
            return makeField(self._target, _jax2anyarray(self._func(_anyarray2jax(val))))
        res, grad = self._val_and_grad(_anyarray2jax(val))
        jac = VdotOperator(makeField(self._domain, _jax2anyarray(grad)))
        res = x.new(makeField(self._target, _jax2anyarray(res)), jac)
        if not x.want_metric:
            return res
        # TODO: Add jax version of metric
        return res.add_metric(self.get_metric_at(x.val))

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        func2 = lambda x: self._func({**x, **_anyarray2jax(c_inp.val)})
        dom = {kk: vv for kk, vv in self._domain.items() if kk not in c_inp.keys()}
        _, trafo = self._trafo.simplify_for_constant_input(c_inp)
        if isinstance(self._dt, dict):
            dt = {kk: self._dt[kk] for kk in dom.keys()}
        else:
            dt = self._dt
        return None, JaxLikelihoodEnergyOperator(dom, func2, trafo, dt)
