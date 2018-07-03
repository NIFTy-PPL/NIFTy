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
            Ax = self._A(self.position)
            self._grad = Ax if b is None else Ax - b
        self._grad.lock()
        self._value = 0.5*self.position.vdot(Ax)
        if b is not None:
            self._value -= b.vdot(self.position)

    def at(self, position):
        return QuadraticEnergy(position=position, A=self._A, b=self._b)

    def at_with_grad(self, position, grad):
        """ Specialized version of `at`, taking also a gradient.

        This custom method is meant for use within :class:ConjugateGradient`
        minimizers, which already have the gradient available. It saves time
        by not recomputing it.

        Parameters
        ----------
        position : Field
            Location in parameter space for the new Energy object.
        grad : Field
            Energy gradient at the new position.

        Returns
        -------
        Energy
            Energy object at new position.
        """
        return QuadraticEnergy(position=position, A=self._A, b=self._b,
                               _grad=grad)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def metric(self):
        return self._A
