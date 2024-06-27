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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..field import Field
from ..logger import logger
from ..multi_field import MultiField
from ..utilities import iscomplextype
from .iteration_controllers import IterationController
from .minimizer import Minimizer


def _multiToArray(fld):
    szall = sum(2*v.size if iscomplextype(v.dtype) else v.size
                for v in fld.values())
    res = np.empty(szall, dtype=np.float64)
    ofs = 0
    for val in fld.values():
        sz2 = 2*val.size if iscomplextype(val.dtype) else val.size
        locdat = val.val.reshape(-1)
        if iscomplextype(val.dtype):
            locdat = locdat.view(locdat.real.dtype)
        res[ofs:ofs+sz2] = locdat
        ofs += sz2
    return res


def _toArray(fld):
    if isinstance(fld, Field):
        return fld.val.reshape(-1)
    return _multiToArray(fld)


def _toArray_rw(fld):
    if isinstance(fld, Field):
        return fld.val_rw().reshape(-1)
    return _multiToArray(fld)


def _toField(arr, template):
    if isinstance(template, Field):
        return Field(template.domain, arr.reshape(template.shape).copy())
    ofs = 0
    res = []
    for v in template.values():
        sz2 = 2*v.size if iscomplextype(v.dtype) else v.size
        locdat = arr[ofs:ofs+sz2].copy()
        if iscomplextype(v.dtype):
            locdat = locdat.view(np.complex128)
        res.append(Field(v.domain, locdat.reshape(v.shape)))
        ofs += sz2
    return MultiField(template.domain, tuple(res))


class _MinHelper:
    def __init__(self, energy):
        self._energy = energy
        self._domain = energy.position.domain

    def _update(self, x):
        pos = _toField(x, self._energy.position)
        if (pos != self._energy.position).s_any():
            self._energy = self._energy.at(pos)

    def fun(self, x):
        self._update(x)
        return self._energy.value

    def jac(self, x):
        self._update(x)
        return _toArray_rw(self._energy.gradient)

    def hessp(self, x, p):
        self._update(x)
        res = self._energy.apply_metric(_toField(p, self._energy.position))
        return _toArray_rw(res)


class _ScipyMinimizer(Minimizer):
    """Scipy-based minimizer

    Parameters
    ----------
    method     : str
        The selected Scipy minimization method.
    options    : dictionary
        A set of custom options for the selected minimizer.
    """

    def __init__(self, method, options, need_hessp, bounds):
        self._method = method
        self._options = options
        self._need_hessp = need_hessp
        self._bounds = bounds

    def __call__(self, energy):
        import scipy.optimize as opt
        hlp = _MinHelper(energy)
        energy = None  # drop handle, since we don't need it any more
        bounds = None
        if self._bounds is not None:
            if len(self._bounds) == 2:
                lo = self._bounds[0]
                hi = self._bounds[1]
                bounds = [(lo, hi)]*hlp._energy.position.size
            else:
                raise ValueError("unrecognized bounds")

        x = _toArray_rw(hlp._energy.position)
        hessp = hlp.hessp if self._need_hessp else None
        r = opt.minimize(hlp.fun, x, method=self._method, jac=hlp.jac,
                         hessp=hessp, options=self._options, bounds=bounds)
        if not r.success:
            logger.error("Problem in Scipy minimization: {}".format(r.message))
            return hlp._energy, IterationController.ERROR
        return hlp._energy, IterationController.CONVERGED


def L_BFGS_B(ftol, gtol, maxiter, maxcor=10, disp=False, bounds=None):
    """Returns a _ScipyMinimizer object carrying out the L-BFGS-B algorithm.

    See Also
    --------
    _ScipyMinimizer
    """
    options = {"ftol": ftol, "gtol": gtol, "maxiter": maxiter,
               "maxcor": maxcor, "disp": disp}
    return _ScipyMinimizer("L-BFGS-B", options, False, bounds)


class _ScipyCG(Minimizer):
    """Returns a _ScipyMinimizer object carrying out the conjugate gradient
    algorithm as implemented by SciPy.

    This class is only intended for double-checking NIFTy's own conjugate
    gradient implementation and should not be used otherwise.
    """
    def __init__(self, tol, maxiter):
        self._tol = tol
        self._maxiter = maxiter

    def __call__(self, energy, preconditioner=None):
        from scipy.sparse.linalg import LinearOperator as scipy_linop
        from scipy.sparse.linalg import cg

        from .quadratic_energy import QuadraticEnergy
        if not isinstance(energy, QuadraticEnergy):
            raise ValueError("need a quadratic energy for CG")

        class mymatvec:
            def __init__(self, op):
                self._op = op

            def __call__(self, inp):
                return _toArray(self._op(_toField(inp, energy.position)))

        op = energy._A
        b = _toArray(energy._b)
        sx = _toArray(energy.position)
        sci_op = scipy_linop(shape=(op.domain.size, op.target.size),
                             matvec=mymatvec(op))
        prec_op = None
        if preconditioner is not None:
            prec_op = scipy_linop(shape=(op.domain.size, op.target.size),
                                  matvec=mymatvec(preconditioner))
        res, stat = cg(sci_op, b, x0=sx, rtol=self._tol, M=prec_op,
                       maxiter=self._maxiter, atol=0.)
        stat = (IterationController.CONVERGED if stat >= 0 else
                IterationController.ERROR)
        return energy.at(_toField(res, energy.position)), stat
