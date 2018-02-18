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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division, print_function
from .minimizer import Minimizer
from ..field import Field
from .. import dobj


class ScipyMinimizer(Minimizer):
    """Scipy-based minimizer

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    method     : str
        The selected Scipy minimization method.
    options    : dictionary
        A set of custom options for the selected minimizer.
    """

    def __init__(self, controller, method, options, need_hessp):
        super(ScipyMinimizer, self).__init__()
        if not dobj.is_numpy():
            raise NotImplementedError
        self._controller = controller
        self._method = method
        self._options = options
        self._need_hessp = need_hessp

    def __call__(self, energy):
        class _MinimizationDone(BaseException):
            pass

        class _MinHelper(object):
            def __init__(self, controller, energy):
                self._controller = controller
                self._energy = energy
                self._domain = energy.position.domain

            def _update(self, x):
                pos = Field(self._domain, x.reshape(self._domain.shape))
                if (pos.val != self._energy.position.val).any():
                    self._energy = self._energy.at(pos.locked_copy())
                    status = self._controller.check(self._energy)
                    if status != self._controller.CONTINUE:
                        raise _MinimizationDone

            def fun(self, x):
                self._update(x)
                return self._energy.value

            def jac(self, x):
                self._update(x)
                return self._energy.gradient.val.flatten()

            def hessp(self, x, p):
                self._update(x)
                vec = Field(self._domain, p.reshape(self._domain.shape))
                res = self._energy.curvature(vec)
                return res.val.flatten()

        import scipy.optimize as opt
        hlp = _MinHelper(self._controller, energy)
        energy = None
        status = self._controller.start(hlp._energy)
        if status != self._controller.CONTINUE:
            return hlp._energy, status
        x = hlp._energy.position.val.flatten()
        try:
            if self._need_hessp:
                r = opt.minimize(hlp.fun, x,
                                 method=self._method, jac=hlp.jac,
                                 hessp=hlp.hessp,
                                 options=self._options)
            else:
                r = opt.minimize(hlp.fun, x,
                                 method=self._method, jac=hlp.jac,
                                 options=self._options)
        except _MinimizationDone:
            status = self._controller.check(hlp._energy)
            return hlp._energy, self._controller.check(hlp._energy)
        if not r.success:
            print("Problem in Scipy minimization:", r.message)
        else:
            print("Problem in Scipy minimization")
        return hlp._energy, self._controller.ERROR


def NewtonCG(controller):
    return ScipyMinimizer(controller, "Newton-CG",
                          {"xtol": 1e-20, "maxiter": None}, True)


def L_BFGS_B(controller, maxcor=10):
    return ScipyMinimizer(controller, "L-BFGS-B",
                          {"ftol": 1e-20, "gtol": 1e-20, "maxcor": maxcor},
                          False)
