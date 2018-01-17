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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from .energy import Energy


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its curvature must be
    position-independent.
    """

    def __init__(self, position, A, b, _grad=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        if _grad is not None:
            self._grad = _grad
            Ax = _grad + self._b
        else:
            Ax = self._A(self.position)
            self._grad = Ax - self._b
        self._value = 0.5*self.position.vdot(Ax) - b.vdot(self.position)

    def at(self, position):
        return QuadraticEnergy(position=position, A=self._A, b=self._b)

    def at_with_grad(self, position, grad):
        return QuadraticEnergy(position=position, A=self._A, b=self._b,
                               _grad=grad)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def curvature(self):
        return self._A
