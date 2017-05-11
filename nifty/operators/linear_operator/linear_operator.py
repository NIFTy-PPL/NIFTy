# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import abc
from nifty.nifty_meta import NiftyMeta

from keepers import Loggable
from nifty.field import Field
import nifty.nifty_utilities as utilities


class LinearOperator(Loggable, object):

    """NIFTY base class for linear operators.
    The base NIFTY operator class is an abstract class from which
    other specific operator subclasses, including those preimplemented
    in NIFTY (e.g. the EndomorphicOperator, ProjectionOperator,
    DiagonalOperator, SmoothingOperator, ResponseOperator,
    PropagatorOperator, ComposedOperator) must be derived.

    Parameters
    ----------


    Attributes
    ----------
    domain : NIFTy.space
        The NIFTy.space in which the operator is defined.
    target : NIFTy.space
        The NIFTy.space in which the outcome of the operator lives
    unitary : boolean
        Indicates whether the operator is unitary or not


    Raises
    ------
    NotImplementedError
        Raised if
            * domain is not defined
            * target is not defined
            * unitary is not set to (True/False)

    Notes
    -----
    All Operators wihtin NIFTy are linear and must therefore be a subclasses of the
    LinearOperator. A LinearOperator must have the attributes domain, target
    and unitary to be properly defined.

    Examples
    --------


    See Also
    --------
    EndomorphicOperator, ProjectionOperator,
    DiagonalOperator, SmoothingOperator, ResponseOperator,
    PropagatorOperator, ComposedOperator

    """

    __metaclass__ = NiftyMeta

    def __init__(self, default_spaces=None):
        self.default_spaces = default_spaces

    def _parse_domain(self, domain):
        return utilities.parse_domain(domain)

    @abc.abstractproperty
    def domain(self):
        """
        domain : NIFTy.space
            The NIFTy.space in which the operator is defined.
            Every operator which inherits from the abstract LinearOperator
            base class must have this attribute.

        Notes :
             is an abstractbaseclass.abstractproperty
             (https://docs.python.org/2/library/abc.html)
        """
        raise NotImplementedError

    @abc.abstractproperty
    def target(self):
        """
        target : NIFTy.space
            The NIFTy.space on which the processed (output) Field lives.
            Every operator which inherits from the abstract LinearOperator
            base class must have this attribute.

        Notes :
             is an abstractbaseclass.abstractproperty
             (https://docs.python.org/2/library/abc.html)
        """
        raise NotImplementedError

    @abc.abstractproperty
    def unitary(self):
        """
        unitary : boolean
            States whether the Operator is unitary or not.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.

        Notes :
             is an abstractbaseclass.abstractproperty
             (https://docs.python.org/2/library/abc.html)

        """
        raise NotImplementedError

    @property
    def default_spaces(self):
        return self._default_spaces

    @default_spaces.setter
    def default_spaces(self, spaces):
        self._default_spaces = utilities.cast_axis_to_tuple(spaces)

    def __call__(self, *args, **kwargs):
        return self.times(*args, **kwargs)

    def times(self, x, spaces=None, **kwargs):
        """ Applies the Operator to a given Field.

        Operator and Field have to live over the same domain.

        Parameters
        ----------
        x : NIFTY.Field
            applies the Operator to the given Field
        spaces : integer (default: None)
            defines on which space of the given Field the Operator acts
        **kwargs
           Additional keyword arguments get passed to the used copy_empty
           routine.

        Returns
        -------
        out : NIFTy.Field
            the processed Field living on the target space

        See Also
       --------

        """

        spaces = self._check_input_compatibility(x, spaces)

        y = self._times(x, spaces, **kwargs)
        return y

    def inverse_times(self, x, spaces=None, **kwargs):
        """ Applies the inverse-Operator to a given Field.

        Operator and Field have to live over the same domain.

        Parameters
        ----------
        x : NIFTY.Field
            applies the Operator to the given Field
        spaces : integer (default: None)
            defines on which space of the given Field the Operator acts
        **kwargs
           Additional keyword arguments get passed to the used copy_empty
           routine.

        Returns
        -------
        out : NIFTy.Field
            the processed Field living on the target space

        See Also
       --------

        """

        spaces = self._check_input_compatibility(x, spaces, inverse=True)

        try:
            y = self._inverse_times(x, spaces, **kwargs)
        except(NotImplementedError):
            if (self.unitary):
                y = self._adjoint_times(x, spaces, **kwargs)
            else:
                raise
        return y

    def adjoint_times(self, x, spaces=None, **kwargs):
        """ Applies the adjoint-Operator to a given Field.

        Operator and Field have to live over the same domain.

        Parameters
        ----------
        x : NIFTY.Field
            applies the Operator to the given Field
        spaces : integer (default: None)
            defines on which space of the given Field the Operator acts
        **kwargs
           Additional keyword arguments get passed to the used copy_empty
           routine.

        Returns
        -------
        out : NIFTy.Field
            the processed Field living on the target space

        See Also
       --------

        """
        if self.unitary:
            return self.inverse_times(x, spaces)

        spaces = self._check_input_compatibility(x, spaces, inverse=True)

        try:
            y = self._adjoint_times(x, spaces, **kwargs)
        except(NotImplementedError):
            if (self.unitary):
                y = self._inverse_times(x, spaces, **kwargs)
            else:
                raise
        return y

    # If the operator supports inverse() then the inverse adjoint is identical
    # to the adjoint inverse. We provide both names for convenience.
    def adjoint_inverse_times(self, x, spaces=None, **kwargs):
        """ Applies the adjoint-inverse Operator to a given Field.

        Operator and Field have to live over the same domain.

        Parameters
        ----------
        x : NIFTY.Field
            applies the Operator to the given Field
        spaces : integer (default: None)
            defines on which space of the given Field the Operator acts
        **kwargs
           Additional keyword arguments get passed to the used copy_empty
           routine.

        Returns
        -------
        out : NIFTy.Field
            the processed Field living on the target space

        See Also
       --------

        """
        if self.unitary:
            return self.times(x, spaces)

        spaces = self._check_input_compatibility(x, spaces)

        try:
            y = self._adjoint_inverse_times(x, spaces, **kwargs)
        except(NotImplementedError):

    def inverse_adjoint_times(self, x, spaces=None, **kwargs):
        """ Applies the inverse-adjoint Operator to a given Field.

        Operator and Field have to live over the same domain.

        Parameters
        ----------
        x : NIFTY.Field
            applies the Operator to the given Field
        spaces : integer (default: None)
            defines on which space of the given Field the Operator acts
        **kwargs
           Additional keyword arguments get passed to the used copy_empty
           routine.

        Returns
        -------
        out : NIFTy.Field
            the processed Field living on the target space

        See Also
       --------

        """
        spaces = self._check_input_compatibility(x, spaces)

        try:
            y = self._inverse_adjoint_times(x, spaces, **kwargs)
        except(NotImplementedError):
            if self.unitary:
                y = self._times(x, spaces, **kwargs)
            else:
                raise
            try:
                y = self._inverse_adjoint_times(x, spaces, **kwargs)
            except(NotImplementedError):
                if self.unitary:
                    y = self._times(x, spaces, **kwargs)
                else:
                    raise
        return y

    def inverse_adjoint_times(self, x, spaces=None, **kwargs):
        return adjoint_inverse_times(x, spaces, **kwargs)

    def _times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'times'.")

    def _adjoint_times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'adjoint_times'.")

    def _inverse_times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'inverse_times'.")

    def _adjoint_inverse_times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'adjoint_inverse_times'.")

    def _inverse_adjoint_times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'inverse_adjoint_times'.")

    def _check_input_compatibility(self, x, spaces, inverse=False):
        if not isinstance(x, Field):
            raise ValueError(
                "supplied object is not a `nifty.Field`.")

        if spaces is None:
            spaces = self.default_spaces

        # sanitize the `spaces` and `types` input
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        # if the operator's domain is set to something, there are two valid
        # cases:
        # 1. Case:
        #   The user specifies with `spaces` that the operators domain should
        #   be applied to certain spaces in the domain-tuple of x.
        # 2. Case:
        #   The domains of self and x match completely.

        if not inverse:
            self_domain = self.domain
        else:
            self_domain = self.target

        if spaces is None:
            if self_domain != x.domain:
                raise ValueError(
                    "The operator's and and field's domains don't "
                    "match.")
        else:
            for i, space_index in enumerate(spaces):
                if x.domain[space_index] != self_domain[i]:
                    raise ValueError(
                        "The operator's and and field's domains don't "
                        "match.")

        return spaces

    def __repr__(self):
        return str(self.__class__)
