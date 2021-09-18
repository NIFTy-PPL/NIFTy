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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..logger import logger
from ..minimization.conjugate_gradient import ConjugateGradient
from ..minimization.iteration_controllers import IterationController
from ..minimization.quadratic_energy import QuadraticEnergy
from ..sugar import full
from .endomorphic_operator import EndomorphicOperator
from .linear_operator import LinearOperator


class InversionEnabler(EndomorphicOperator):
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

    def __init__(self, op, iteration_controller, approximation=None):
        # isinstance(op, EndomorphicOperator) does not suffice since op can be
        # a ChainOperator
        if not isinstance(op, LinearOperator):
            raise TypeError('Operator needs to be linear.')
        if op.domain is not op.target:
            raise TypeError('Operator needs to be endomorphic.')
        self._op = op
        self._ic = iteration_controller
        self._approximation = approximation
        self._domain = op.domain
        self._capability = self._addInverse[self._op.capability]

    def apply(self, x, mode):
        self._check_mode(mode)
        if self._op.capability & mode:
            return self._op.apply(x, mode)

        x0 = full(x.domain, 0.)
        invmode = self._modeTable[self.INVERSE_BIT][self._ilog[mode]]
        invop = self._op._flip_modes(self._ilog[invmode])
        prec = self._approximation
        if prec is not None:
            prec = prec._flip_modes(self._ilog[mode])
        energy = QuadraticEnergy(x0, invop, x)
        inverter = ConjugateGradient(self._ic)
        r, stat = inverter(energy, preconditioner=prec)
        if stat != IterationController.CONVERGED:
            logger.warning("Error detected during operator inversion")
        return r.position

    def draw_sample(self, from_inverse=False):
        return self._op.draw_sample(from_inverse)
