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
from ..compat import *
import numpy as np
from ..field import Field
from ..utilities import NiftyMetaBase, memo


class Energy(NiftyMetaBase()):
    """ Provides the functional used by minimization schemes.

   The Energy object is an implementation of a scalar function including its
   gradient and metric at some position.

    Parameters
    ----------
    position : Field
        The input parameter of the scalar function.

    Notes
    -----
    An instance of the Energy class is defined at a certain location. If one
    is interested in the value, gradient or metric of the abstract energy
    functional one has to 'jump' to the new position using the `at` method.
    This method returns a new energy instance residing at the new position. By
    this approach, intermediate results from computing e.g. the gradient can
    safely be reused for e.g. the value or the metric.

    Memorizing the evaluations of some quantities (using the memo decorator)
    minimizes the computational effort for multiple calls.

    See Also
    --------
    memo

    """

    def __init__(self, position):
        super(Energy, self).__init__()
        self._position = position

    def at(self, position):
        """ Returns a new Energy object, initialized at `position`.

        Parameters
        ----------
        position : Field
            Location in parameter space for the new Energy object.

        Returns
        -------
        Energy
            Energy object at new position.
        """
        return self.__class__(position)

    @property
    def position(self):
        """
        Field : selected location in parameter space.

        The Field location in parameter space where value, gradient and
        metric are evaluated.
        """
        return self._position

    @property
    def value(self):
        """
        float : value of the functional.

            The value of the energy functional at given `position`.
        """
        raise NotImplementedError

    @property
    def gradient(self):
        """
        Field : The gradient at given `position`.
        """
        raise NotImplementedError

    @property
    @memo
    def gradient_norm(self):
        """
        float : L2-norm of the gradient at given `position`.
        """
        return self.gradient.norm()

    @property
    def metric(self):
        """
        LinearOperator : implicitly defined metric.
            A positive semi-definite operator or function describing the
            metric of the potential at the given `position`.
        """
        raise NotImplementedError

    def longest_step(self, dir):
        """Returns the longest allowed step size along `dir`

        Parameters
        ----------
        dir : Field
            the search direction

        Returns
        -------
        float or None
            the longest allowed step when starting from `self.position` along
            `dir`. If None, the step size is not limited.
        """
        return None

    def makeInvertible(self, controller, preconditioner=None):
        from .iteration_controller import IterationController
        if not isinstance(controller, IterationController):
            raise TypeError
        return MetricInversionEnabler(self, controller, preconditioner)

    def __mul__(self, factor):
        from .energy_sum import EnergySum
        if isinstance(factor, (float, int)):
            return EnergySum.make([self], [factor])
        return NotImplemented

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __add__(self, other):
        from .energy_sum import EnergySum
        if isinstance(other, Energy):
            return EnergySum.make([self, other])
        return NotImplemented

    def __sub__(self, other):
        from .energy_sum import EnergySum
        if isinstance(other, Energy):
            return EnergySum.make([self, other], [1., -1.])
        return NotImplemented

    def __neg__(self):
        from .energy_sum import EnergySum
        return EnergySum.make([self], [-1.])


class MetricInversionEnabler(Energy):
    def __init__(self, ene, controller, preconditioner):
        super(MetricInversionEnabler, self).__init__(ene.position)
        self._energy = ene
        self._controller = controller
        self._preconditioner = preconditioner

    def at(self, position):
        if self._position.isSubsetOf(position):
            return self
        return MetricInversionEnabler(
            self._energy.at(position), self._controller, self._preconditioner)

    @property
    def position(self):
        return self._energy.position

    @property
    def value(self):
        return self._energy.value

    @property
    def gradient(self):
        return self._energy.gradient

    @property
    def metric(self):
        from ..operators.linear_operator import LinearOperator
        from ..operators.inversion_enabler import InversionEnabler
        curv = self._energy.metric
        if self._preconditioner is None:
            precond = None
        elif isinstance(self._preconditioner, LinearOperator):
            precond = self._preconditioner
        elif isinstance(self._preconditioner, Energy):
            precond = self._preconditioner.at(self.position).metric
        return InversionEnabler(curv, self._controller, precond)

    def longest_step(self, dir):
        return self._energy.longest_step(dir)
