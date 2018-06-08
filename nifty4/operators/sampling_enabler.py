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

from ..minimization.quadratic_energy import QuadraticEnergy
from ..minimization.iteration_controller import IterationController
from ..logger import logger
from .endomorphic_operator import EndomorphicOperator
import numpy as np


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
    inverter : :class:`Minimizer`
        The minimizer to use for the iterative numerical inversion.
        Typically, this is a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this operator should be an approximation to `op`, which
        supports the operation modes that `op` doesn't have. It is used as a
        preconditioner during the iterative inversion, to accelerate
        convergence.
    """

    def __init__(self, likelihood, prior, sampling_inverter,
                 approximation=None):
        self._op = likelihood + prior
        super(SamplingEnabler, self).__init__()
        self._likelihood = likelihood
        self._prior = prior
        self._sampling_inverter = sampling_inverter

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        try:
            return self._op.draw_sample(from_inverse, dtype)
        except NotImplementedError:
            s = self._prior.draw_sample()
            sp = self._prior.inverse_times(s)
            nj = self._likelihood.draw_sample()
            energy = QuadraticEnergy(s, self._op, sp + nj,
                                     _grad=self._likelihood(s) - nj)
            energy, convergence = self._sampling_inverter(energy)
            return energy.position

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)
