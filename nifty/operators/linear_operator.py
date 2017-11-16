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

from builtins import str
import abc
from ..utilities import NiftyMeta
from ..field import Field
from future.utils import with_metaclass


class LinearOperator(with_metaclass(
        NiftyMeta, type('NewBase', (object,), {}))):
    """NIFTY base class for linear operators.

    The base NIFTY operator class is an abstract class from which
    other specific operator subclasses are derived.


    Attributes
    ----------
    domain : DomainTuple
        The domain on which the Operator's input Field lives.
    target : DomainTuple
        The domain in which the Operators result lives.
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    """

    def __init__(self):
        pass

    @abc.abstractproperty
    def domain(self):
        """
        domain : DomainTuple
            The domain on which the Operator's input Field lives.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def target(self):
        """
        target : DomainTuple
            The domain on which the Operator's output Field lives.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def unitary(self):
        """
        unitary : boolean
            States whether the Operator is unitary or not.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.times(x)

    def times(self, x):
        """ Applies the Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's domain.

        Returns
        -------
        out : Field
            The processed Field living on the Operator's target domain.
        """
        self._check_input_compatibility(x)
        return self._times(x)

    def inverse_times(self, x):
        """Applies the inverse Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's target domain

        Returns
        -------
        out : Field
            The processed Field living on the Operator's domain.
        """
        self._check_input_compatibility(x, inverse=True)
        try:
            y = self._inverse_times(x)
        except NotImplementedError:
            if self.unitary:
                y = self._adjoint_times(x)
            else:
                raise
        return y

    def adjoint_times(self, x):
        """Applies the adjoint-Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's target domain

        Returns
        -------
        out : Field
            The processed Field living on the Operator's domain.
        """
        if self.unitary:
            return self.inverse_times(x)

        self._check_input_compatibility(x, inverse=True)
        try:
            y = self._adjoint_times(x)
        except NotImplementedError:
            if self.unitary:
                y = self._inverse_times(x)
            else:
                raise
        return y

    def adjoint_inverse_times(self, x):
        """ Applies the adjoint-inverse Operator to a given Field.

        Parameters
        ----------
        x : Field
            The input Field, living on the Operator's domain.

        Returns
        -------
        out : Field
            The processed Field living on the Operator's target domain.

        Notes
        -----
        If the operator has an `inverse` then the inverse adjoint is identical
        to the adjoint inverse. We provide both names for convenience.
        """
        self._check_input_compatibility(x)
        try:
            y = self._adjoint_inverse_times(x)
        except NotImplementedError:
            if self.unitary:
                y = self._times(x)
            else:
                raise
        return y

    def inverse_adjoint_times(self, x):
        return self.adjoint_inverse_times(x)

    def _times(self, x):
        raise NotImplementedError(
            "no generic instance method 'times'.")

    def _adjoint_times(self, x):
        raise NotImplementedError(
            "no generic instance method 'adjoint_times'.")

    def _inverse_times(self, x):
        raise NotImplementedError(
            "no generic instance method 'inverse_times'.")

    def _adjoint_inverse_times(self, x):
        raise NotImplementedError(
            "no generic instance method 'adjoint_inverse_times'.")

    def _check_input_compatibility(self, x, inverse=False):
        if not isinstance(x, Field):
            raise ValueError("supplied object is not a `Field`.")

        if x.domain != (self.target if inverse else self.domain):
            raise ValueError("The operator's and and field's domains "
                             "don't match.")

    def __repr__(self):
        return str(self.__class__)
