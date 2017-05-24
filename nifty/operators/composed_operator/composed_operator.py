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

from nifty.operators.linear_operator import LinearOperator


class ComposedOperator(LinearOperator):
    """ NIFTY class for composed operators.

    The  NIFTY composed operator class combines multiple linear operators.

    Parameters
    ----------
    operators : tuple of NIFTy Operators
        The tuple of LinearOperators.
    default_spaces : tuple of ints *optional*
        Defines on which space(s) of a given field the Operator acts by
        default (default: None)


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
              LinearOperator-baseclass.

    Notes
    -----
    Very usefull in case one has to transform a Field living over a product
    space (see example below).

    Examples
    --------
    Minimal example of transforming a Field living on two domains into its
    harmonic space.

    >>> x1 = RGSpace(5)
    >>> x2 = RGSpace(10)
    >>> k1 = RGRGTransformation.get_codomain(x1)
    >>> k2 = RGRGTransformation.get_codomain(x2)
    >>> FFT1 = FFTOperator(domain=x1, target=k1,
                           domain_dtype=np.float64, target_dtype=np.complex128)
    >>> FFT2 = FFTOperator(domain=x2, target=k2,
                           domain_dtype=np.float64, target_dtype=np.complex128)
    >>> FFT = ComposedOperator((FFT1, FFT2)
    >>> f = Field.from_random('normal', domain=(x1,x2))
    >>> FFT.times(f)

    See Also
    --------
    EndomorphicOperator, ProjectionOperator,
    DiagonalOperator, SmoothingOperator, ResponseOperator,
    PropagatorOperator, ComposedOperator

    """

    # ---Overwritten properties and methods---
    def __init__(self, operators, default_spaces=None):
        super(ComposedOperator, self).__init__(default_spaces)

        self._operator_store = ()
        for op in operators:
            if not isinstance(op, LinearOperator):
                raise TypeError("The elements of the operator list must be"
                                "instances of the LinearOperator-baseclass")
            self._operator_store += (op,)

    def _check_input_compatibility(self, x, spaces, inverse=False):
        """
        The input check must be disabled for the ComposedOperator, since it
        is not easily forecasteable what the output of an operator-call
        will look like.
        """
        return spaces

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        if not hasattr(self, '_domain'):
            self._domain = ()
            for op in self._operator_store:
                self._domain += op.domain
        return self._domain

    @property
    def target(self):
        if not hasattr(self, '_target'):
            self._target = ()
            for op in self._operator_store:
                self._target += op.target
        return self._target

    @property
    def unitary(self):
        return False

    def _times(self, x, spaces):
        return self._times_helper(x, spaces, func='times')

    def _adjoint_times(self, x, spaces):
        return self._inverse_times_helper(x, spaces, func='adjoint_times')

    def _inverse_times(self, x, spaces):
        return self._inverse_times_helper(x, spaces, func='inverse_times')

    def _adjoint_inverse_times(self, x, spaces):
        return self._times_helper(x, spaces, func='adjoint_inverse_times')

    def _times_helper(self, x, spaces, func):
        space_index = 0
        if spaces is None:
            spaces = range(len(self.domain))
        for op in self._operator_store:
            active_spaces = spaces[space_index:space_index+len(op.domain)]
            space_index += len(op.domain)

            x = getattr(op, func)(x, spaces=active_spaces)
        return x

    def _inverse_times_helper(self, x, spaces, func):
        space_index = 0
        if spaces is None:
            spaces = range(len(self.target))[::-1]
        for op in reversed(self._operator_store):
            active_spaces = spaces[space_index:space_index+len(op.target)]
            space_index += len(op.target)

            x = getattr(op, func)(x, spaces=active_spaces)
        return x
