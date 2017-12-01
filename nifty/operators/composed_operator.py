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

from builtins import range
from .linear_operator import LinearOperator


class ComposedOperator(LinearOperator):
    """ NIFTY class for composed operators.

    The  NIFTY composed operator class combines multiple linear operators.

    Parameters
    ----------
    operators : tuple of NIFTy Operators
        The tuple of LinearOperators.


    Attributes
    ----------
    domain : DomainTuple
        The NIFTy.space in which the operator is defined.
    target : DomainTuple
        The NIFTy.space in which the outcome of the operator lives
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    """

    def __init__(self, operators):
        super(ComposedOperator, self).__init__()

        self._operator_store = ()
        old_op = None
        for op in operators:
            if not isinstance(op, LinearOperator):
                raise TypeError("The elements of the operator list must be"
                                "instances of the LinearOperator base class")
            if old_op is not None and op.domain != old_op.target:
                raise ValueError("incompatible domains")
            self._operator_store += (op,)
            old_op = op

    @property
    def domain(self):
        return self._operator_store[0].domain

    @property
    def target(self):
        return self._operator_store[-1].target

    @property
    def unitary(self):
        return False

    def _times(self, x):
        return self._times_helper(x, func='times')

    def _adjoint_times(self, x):
        return self._inverse_times_helper(x, func='adjoint_times')

    def _inverse_times(self, x):
        return self._inverse_times_helper(x, func='inverse_times')

    def _adjoint_inverse_times(self, x):
        return self._times_helper(x, func='adjoint_inverse_times')

    def _times_helper(self, x, func):
        for op in self._operator_store:
            x = getattr(op, func)(x)
        return x

    def _inverse_times_helper(self, x, func):
        for op in reversed(self._operator_store):
            x = getattr(op, func)(x)
        return x
