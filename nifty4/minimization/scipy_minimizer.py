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

from __future__ import division
from .minimizer import Minimizer
from ..field import Field
from .. import dobj
from ..logger import logger
from .iteration_controller import IterationController


class _MinHelper(object):
    def __init__(self, energy):
        self._energy = energy
        self._domain = energy.position.domain

    def _update(self, x):
        pos = Field(self._domain, x.reshape(self._domain.shape))
        if (pos.val != self._energy.position.val).any():
            self._energy = self._energy.at(pos.locked_copy())

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


class ScipyMinimizer(Minimizer):
    """Scipy-based minimizer

    Parameters
    ----------
    method     : str
        The selected Scipy minimization method.
    options    : dictionary
        A set of custom options for the selected minimizer.
    """

    def __init__(self, method, options, need_hessp, bounds):
        super(ScipyMinimizer, self).__init__()
        if not dobj.is_numpy():
            raise NotImplementedError
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

        x = hlp._energy.position.val.flatten()
        hessp = hlp.hessp if self._need_hessp else None
        r = opt.minimize(hlp.fun, x, method=self._method, jac=hlp.jac,
                         hessp=hessp, options=self._options, bounds=bounds)
        if not r.success:
            logger.error("Problem in Scipy minimization:", r.message)
            return hlp._energy, IterationController.ERROR
        return hlp._energy, IterationController.CONVERGED


def NewtonCG(xtol, maxiter, disp=False):
    """Returns a ScipyMinimizer object carrying out the Newton-CG algorithm.

    See Also
    --------
    ScipyMinimizer
    """
    options = {"xtol": xtol, "maxiter": maxiter, "disp": disp}
    return ScipyMinimizer("Newton-CG", options, True, None)


def L_BFGS_B(ftol, gtol, maxiter, maxcor=10, disp=False, bounds=None):
    """Returns a ScipyMinimizer object carrying out the L-BFGS-B algorithm.

    See Also
    --------
    ScipyMinimizer
    """
    options = {"ftol": ftol, "gtol": gtol, "maxiter": maxiter,
               "maxcor": maxcor, "disp": disp}
    return ScipyMinimizer("L-BFGS-B", options, False, bounds)
