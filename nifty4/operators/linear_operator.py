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

import abc
from ..utilities import NiftyMetaBase
from ..field import Field
import numpy as np


class LinearOperator(NiftyMetaBase()):
    """NIFTY base class for linear operators.

    The base NIFTY operator class is an abstract class from which
    other specific operator subclasses are derived.

    Attributes
    ----------
    TIMES : int
        Symbolic constant representing normal operator application
    ADJOINT_TIMES : int
        Symbolic constant representing adjoint operator application
    INVERSE_TIMES : int
        Symbolic constant representing inverse operator application
    ADJOINT_INVERSE_TIMES : int
        Symbolic constant representing adjoint inverse operator application
    INVERSE_ADJOINT_TIMES : int
        same as ADJOINT_INVERSE_TIMES

    Notes
    -----
    The symbolic constants for the operation modes can be combined by the
    "bitwise-or" operator "|", for expressing the capability of the operator
    by means of a single integer number.
    """

    # Field Operator Modes
    TIMES = 1
    ADJOINT_TIMES = 2
    INVERSE_TIMES = 4
    ADJOINT_INVERSE_TIMES = 8
    INVERSE_ADJOINT_TIMES = 8

    # Operator Transform Flags
    ADJOINT_BIT = 1
    INVERSE_BIT = 2

    _ilog = (-1, 0, 1, -1, 2, -1, -1, -1, 3)
    _validMode = (False, True, True, False, True, False, False, False, True)
    _modeTable = ((1, 2, 4, 8),
                  (2, 1, 8, 4),
                  (4, 8, 1, 2),
                  (8, 4, 2, 1))
    _capTable = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                 (0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15),
                 (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15),
                 (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15))
    _addInverse = (0, 5, 10, 15, 5, 5, 15, 15, 10, 15, 10, 15, 15, 15, 15, 15)
    _backwards = 6
    _all_ops = 15

    def _dom(self, mode):
        return self.domain if (mode & 9) else self.target

    def _tgt(self, mode):
        return self.domain if (mode & 6) else self.target

    def __init__(self):
        pass

    @abc.abstractproperty
    def domain(self):
        """DomainTuple : the operator's input domain

            The domain on which the Operator's input Field lives."""
        raise NotImplementedError

    @abc.abstractproperty
    def target(self):
        """DomainTuple : the operator's output domain

            The domain on which the Operator's output Field lives."""
        raise NotImplementedError

    def _flip_modes(self, trafo):
        from .operator_adapter import OperatorAdapter
        return self if trafo == 0 else OperatorAdapter(self, trafo)

    @property
    def inverse(self):
        """LinearOperator : the inverse of `self`

        Returns a LinearOperator object which behaves as if it were
        the inverse of this operator."""
        return self._flip_modes(self.INVERSE_BIT)

    @property
    def adjoint(self):
        """LinearOperator : the adjoint of `self`

        Returns a LinearOperator object which behaves as if it were
        the adjoint of this operator."""
        return self._flip_modes(self.ADJOINT_BIT)

    @staticmethod
    def _toOperator(thing, dom):
        from .diagonal_operator import DiagonalOperator
        from .scaling_operator import ScalingOperator
        if isinstance(thing, LinearOperator):
            return thing
        if isinstance(thing, Field):
            return DiagonalOperator(thing)
        if np.isscalar(thing):
            return ScalingOperator(thing, dom)
        return NotImplemented

    def __mul__(self, other):
        from .chain_operator import ChainOperator
        other = self._toOperator(other, self.domain)
        return ChainOperator.make([self, other])

    def __rmul__(self, other):
        from .chain_operator import ChainOperator
        other = self._toOperator(other, self.target)
        return ChainOperator.make([other, self])

    def __add__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator.make([self, other], [False, False])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator.make([self, other], [False, True])

    def __rsub__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator.make([other, self], [False, True])

    @abc.abstractproperty
    def capability(self):
        """int : the supported operation modes

        Returns the supported subset of :attr:`TIMES`, :attr:`ADJOINT_TIMES`,
        :attr:`INVERSE_TIMES`, and :attr:`ADJOINT_INVERSE_TIMES`,
        joined together by the "|" operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x, mode):
        """ Applies the Operator to a given `x`, in a specified `mode`.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's domain or target,
            depending on mode.

        mode : int
            - :attr:`TIMES`: normal application
            - :attr:`ADJOINT_TIMES`: adjoint application
            - :attr:`INVERSE_TIMES`: inverse application
            - :attr:`ADJOINT_INVERSE_TIMES` or
              :attr:`INVERSE_ADJOINT_TIMES`: adjoint inverse application

        Returns
        -------
        Field
            The processed Field living on the Operator's target or domain,
            depending on mode.
        """
        raise NotImplementedError

    def __call__(self, x):
        """Same as :meth:`times`"""
        return self.apply(x, self.TIMES)

    def times(self, x):
        """ Applies the Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's domain.

        Returns
        -------
        Field
            The processed Field living on the Operator's target domain.
        """
        return self.apply(x, self.TIMES)

    def inverse_times(self, x):
        """Applies the inverse Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's target domain

        Returns
        -------
        Field
            The processed Field living on the Operator's domain.
        """
        return self.apply(x, self.INVERSE_TIMES)

    def adjoint_times(self, x):
        """Applies the adjoint-Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's target domain

        Returns
        -------
        Field
            The processed Field living on the Operator's domain.
        """
        return self.apply(x, self.ADJOINT_TIMES)

    def adjoint_inverse_times(self, x):
        """ Applies the adjoint-inverse Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's domain.

        Returns
        -------
        Field
            The processed Field living on the Operator's target domain.

        Notes
        -----
        If the operator has an `inverse` then the inverse adjoint is identical
        to the adjoint inverse. We provide both names for convenience.
        """
        return self.apply(x, self.ADJOINT_INVERSE_TIMES)

    def inverse_adjoint_times(self, x):
        """Same as :meth:`adjoint_inverse_times`"""
        return self.apply(x, self.ADJOINT_INVERSE_TIMES)

    def _check_mode(self, mode):
        if not self._validMode[mode]:
            raise ValueError("invalid operator mode specified")
        if mode & self.capability == 0:
            raise ValueError("requested operator mode is not supported")

    def _check_input(self, x, mode):
        if not isinstance(x, Field):
            raise ValueError("supplied object is not a `Field`.")

        self._check_mode(mode)
        if x.domain != self._dom(mode):
            raise ValueError("The operator's and field's domains don't match.")
