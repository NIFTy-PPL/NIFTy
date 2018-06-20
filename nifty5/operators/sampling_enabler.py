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

import numpy as np

from ..minimization.conjugate_gradient import ConjugateGradient
from ..minimization.quadratic_energy import QuadraticEnergy
from .endomorphic_operator import EndomorphicOperator


class SamplingEnabler(EndomorphicOperator):
    """Class which augments the capability of another operator object via
    numerical inversion.

    Parameters
    ----------
    op : :class:`EndomorphicOperator`
        The operator to be enhanced.
        The InversionEnabler object will support the same operation modes as
        `op`, and additionally the inverse set. The newly-added modes will
        be computed by iterative inversion.
    iteration_controller : :class:`IterationController`
        The iteration controller to use for the iterative numerical inversion
        done by a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this operator should be an approximation to `op`, which
        supports the operation modes that `op` doesn't have. It is used as a
        preconditioner during the iterative inversion, to accelerate
        convergence.
    """

    def __init__(self, likelihood, prior, iteration_controller,
                 approximation=None):
        self._op = likelihood + prior
        super(SamplingEnabler, self).__init__()
        self._likelihood = likelihood
        self._prior = prior
        self._ic = iteration_controller
        self._approximation = approximation

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        try:
            return self._op.draw_sample(from_inverse, dtype)
        except NotImplementedError:
            # MR FIXME: I think there is a silent assumption that
            # from_inverse==True when we arrive here.
            # Can we make this explicit?
            s = self._prior.draw_sample(from_inverse=True)
            sp = self._prior(s)
            nj = self._likelihood.draw_sample()
            energy = QuadraticEnergy(s, self._op, sp + nj,
                                     _grad=self._likelihood(s) - nj)
            inverter = ConjugateGradient(self._ic)
            if self._approximation is not None:
                energy, convergence = inverter(
                    energy, preconditioner=self._approximation.inverse)
            else:
                energy, convergence = inverter(energy)
            return energy.position

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)
