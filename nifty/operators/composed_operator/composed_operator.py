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
from ..linear_operator import LinearOperator
from ... import DomainTuple

class ComposedOperator(LinearOperator):
    """ NIFTY class for composed operators.

    The  NIFTY composed operator class combines multiple linear operators.

    Parameters
    ----------
    operators : tuple of NIFTy Operators
        The tuple of LinearOperators.


    Attributes
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The NIFTy.space in which the operator is defined.
    target : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The NIFTy.space in which the outcome of the operator lives
    unitary : boolean
        Indicates whether the Operator is unitary or not.

    Raises
    ------
    TypeError
        Raised if
            * an element of the operator list is not an instance of the
              LinearOperator base class.

    Notes
    -----
    Very useful in case one has to transform a Field living over a product
    space (see example below).

    Examples
    --------
    Minimal example of transforming a Field living on two domains into its
    harmonic space.

    >>> x1 = RGSpace(5)
    >>> x2 = RGSpace(10)
    >>> k1 = RGRGTransformation.get_codomain(x1)
    >>> k2 = RGRGTransformation.get_codomain(x2)
    >>> FFT1 = FFTOperator(domain=(x1,x2), target=(k1,x2), space=0)
    >>> FFT2 = FFTOperator(domain=(k1,x2), target=(k1,k2), space=1)
    >>> FFT = ComposedOperator((FFT1, FFT2)
    >>> f = Field.from_random('normal', domain=(x1,x2))
    >>> FFT.times(f)

    """

    # ---Overwritten properties and methods---
    def __init__(self, operators):
        super(ComposedOperator, self).__init__()

        for i in range(1, len(operators)):
            if operators[i].domain != operators[i-1].target:
                raise ValueError("incompatible domains")
        self._operator_store = ()
        for op in operators:
            if not isinstance(op, LinearOperator):
                raise TypeError("The elements of the operator list must be"
                                "instances of the LinearOperator base class")
            self._operator_store += (op,)

    # ---Mandatory properties and methods---
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
