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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..minimization.conjugate_gradient import ConjugateGradient
from ..minimization.quadratic_energy import QuadraticEnergy
from .endomorphic_operator import EndomorphicOperator
from .operator import Operator


class SamplingEnabler(EndomorphicOperator):
    """Class which acts as a operator object built of (`likelihood` + `prior`)
    and enables sampling from its inverse even if the operator object
    itself does not support it.


    Parameters
    ----------
    likelihood : :class:`EndomorphicOperator`
        Metric of the likelihood
    prior : :class:`EndomorphicOperator`
        Metric of the prior
    iteration_controller : :class:`IterationController`
        The iteration controller to use for the iterative numerical inversion
        done by a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this linear operator should be an approximation to the
        operator, which supports the operation modes that the operator doesn't
        have. It is used as a preconditioner during the iterative inversion,
        to accelerate convergence.
    start_from_zero : boolean
        If true, the conjugate gradient algorithm starts from a field filled
        with zeros. Otherwise, it starts from a prior samples. Default is
        False.
    """

    def __init__(self, likelihood, prior, iteration_controller,
                 approximation=None, start_from_zero=False):
        if not isinstance(likelihood, Operator) or not isinstance(prior, Operator):
            raise TypeError
        self._likelihood = likelihood
        self._prior = prior
        self._ic = iteration_controller
        self._approximation = approximation
        self._start_from_zero = bool(start_from_zero)
        self._op = likelihood + prior
        self._domain = self._op.domain
        self._capability = self._op.capability
        self.apply = self._op.apply

    def special_draw_sample(self, from_inverse=False):
        try:
            res = self._op.draw_sample(from_inverse)
            return self._op(res), res
        except NotImplementedError:
            if not from_inverse:
                raise ValueError("from_inverse must be True here")
            if self._start_from_zero:
                b = self._op.draw_sample()
                energy = QuadraticEnergy(0*b, self._op, b)
            else:
                s = self._prior.draw_sample(from_inverse=True)
                nj = self._likelihood.draw_sample()
                b = self._prior(s) + nj
                energy = QuadraticEnergy(s, self._op, b,
                                         _grad=self._likelihood(s) - nj)
            inverter = ConjugateGradient(self._ic)
            if self._approximation is not None:
                energy, convergence = inverter(
                    energy, preconditioner=self._approximation.inverse)
            else:
                energy, convergence = inverter(energy)
            return b, energy.position

    def draw_sample(self, from_inverse=False):
        return self.special_draw_sample(from_inverse=from_inverse)[1]

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            "SamplingEnabler:",
            indent("\n".join((
                "Likelihood:", self._likelihood.__repr__(),
                "Prior:", self._prior.__repr__())))))
