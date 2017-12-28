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
    """

    def __init__(self, operators):
        super(ComposedOperator, self).__init__()

        self._operator_store = ()
        old_op = None
        self._capability = operators[0].capability
        for op in operators:
            if not isinstance(op, LinearOperator):
                raise TypeError("The elements of the operator list must be"
                                "instances of the LinearOperator base class")
            if old_op is not None and op.domain != old_op.target:
                raise ValueError("incompatible domains")
            self._operator_store += (op,)
            self._capability &= op.capability
            old_op = op
        if self._capability == 0:
            raise ValueError("composed operator does not support any mode")

    @property
    def domain(self):
        return self._operator_store[0].domain

    @property
    def target(self):
        return self._operator_store[-1].target

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            for op in self._operator_store:
                x = op.apply(x, mode)
        else:
            for op in reversed(self._operator_store):
                x = op.apply(x, mode)
        return x
