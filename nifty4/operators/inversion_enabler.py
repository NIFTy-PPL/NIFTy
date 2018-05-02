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
from ..field import Field
from ..logger import logger
from .endomorphic_operator import EndomorphicOperator
import numpy as np


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
    inverter : :class:`Minimizer`
        The minimizer to use for the iterative numerical inversion.
        Typically, this is a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this operator should be an approximation to `op`, which
        supports the operation modes that `op` doesn't have. It is used as a
        preconditioner during the iterative inversion, to accelerate
        convergence.
    """

    def __init__(self, op, inverter, approximation=None):
        super(InversionEnabler, self).__init__()
        self._op = op
        self._inverter = inverter
        self._approximation = approximation

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def capability(self):
        return self._addInverse[self._op.capability]

    def apply(self, x, mode):
        self._check_mode(mode)
        if self._op.capability & mode:
            return self._op.apply(x, mode)

        x0 = Field.zeros(self._tgt(mode), dtype=x.dtype)
        invmode = self._modeTable[self.INVERSE_BIT][self._ilog[mode]]
        invop = self._op._flip_modes(self._ilog[invmode])
        prec = self._approximation
        if prec is not None:
            prec = prec._flip_modes(self._ilog[mode])
        energy = QuadraticEnergy(x0, invop, x)
        r, stat = self._inverter(energy, preconditioner=prec)
        if stat != IterationController.CONVERGED:
            logger.warning("Error detected during operator inversion")
        return r.position

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        try:
            return self._op.draw_sample(from_inverse, dtype)
        except NotImplementedError:
            samp = self._op.draw_sample(not from_inverse, dtype)
            return self.inverse_times(samp) if from_inverse else self(samp)
