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

from .energy import Energy


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its metric must be
    position-independent.
    """

    def __init__(self, position, A, b, _grad=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        if _grad is not None:
            self._grad = _grad
            Ax = _grad if b is None else _grad + b
        else:
            Ax = self._A(self._position)
            self._grad = Ax if b is None else Ax - b
        self._value = 0.5*self._position.s_vdot(Ax).real
        if b is not None:
            self._value -= b.s_vdot(self._position).real

    def at(self, position):
        return QuadraticEnergy(position, self._A, self._b)

    def at_with_grad(self, position, grad):
        """Specialized version of `at`, taking also a gradient.

        This custom method is meant for use within :class:ConjugateGradient`
        minimizers, which already have the gradient available. It saves time
        by not recomputing it.

        Parameters
        ----------
        position : :class:`nifty8.field.Field`
            Location in parameter space for the new Energy object.
        grad : :class:`nifty8.field.Field`
            Energy gradient at the new position.

        Returns
        -------
        Energy
            Energy object at new position.
        """
        return QuadraticEnergy(position, self._A, self._b, grad)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def metric(self):
        return self._A

    def apply_metric(self, x):
        return self._A(x)
