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

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from ..minimization.conjugate_gradient import ConjugateGradient
from ..minimization.quadratic_energy import QuadraticEnergy
from .endomorphic_operator import EndomorphicOperator


class SamplingEnabler(EndomorphicOperator):
    """Class which acts as the operator object (likelihood + prior)
    and enables sampling from its inverse even if the operator object
    itself does not support it.


    Parameters
    ----------
    likelihood : :class:`EndomorphicOperator`
        Metric of the likelihood
    prior : :class:`EndomorphicOperator`
        Inverse metric of the prior
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
        self._likelihood = likelihood
        self._prior = prior
        self._ic = iteration_controller
        self._approximation = approximation
        self._domain = self._op.domain
        self._capability = self._op.capability

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        try:
            return self._op.draw_sample(from_inverse, dtype)
        except NotImplementedError:
            if not from_inverse:
                raise ValueError("from_inverse must be True here")
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

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            "SamplingEnabler:",
            indent("\n".join((
                "Likelihood:", self._likelihood.__repr__(),
                "Prior:", self._prior.__repr__())))))
