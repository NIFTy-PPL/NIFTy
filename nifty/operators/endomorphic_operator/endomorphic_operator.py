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

from nifty.operators.linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):

    """NIFTY class for endomorphic operators.
    The  NIFTY EndomorphicOperator class is a class derived from the
    LinearOperator. Domain and target are the same in any EndomorphicOperator.
    Prominent other specific operator subclasses, in NIFTy are
    (e.g. DiagonalOperator, SmoothingOperator,
    PropagatorOperator, ProjectionOperator).

    Parameters
    ----------

    Attributes
    ----------
    domain : NIFTy.space
        The NIFTy.space in which the operator is defined.
    target : NIFTy.space
        The NIFTy.space in which the outcome of the operator lives.
        As the Operator is endomorphic this is the same as its domain.
    self_adjoint : boolean
        Indicates whether the operator is self_adjoint or not.
    unitary: boolean
        Indicates whether the operator is unitary or not.

    Raises
    ------
    NotImplementedError
        Raised if
            * self_adjoint is not defined

    Notes
    -----

    Examples
    --------


    See Also
    --------
    DiagonalOperator, SmoothingOperator,
    PropagatorOperator, ProjectionOperator

    """


    # ---Overwritten properties and methods---

    def inverse_times(self, x, spaces=None):
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
            the processed Field living on the domain space

        See Also
       --------

        """
        if self.self_adjoint and self.unitary:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_times(
                                                              x=x,
                                                              spaces=spaces)

    def adjoint_times(self, x, spaces=None):
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
            the processed Field living on the domain space

        See Also
       --------

        """
        if self.self_adjoint:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    def adjoint_inverse_times(self, x, spaces=None):
        """ Applies the adjoint-inverse-Operator to a given Field.

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
            the processed Field living on the domain space

        See Also
       --------

        """
        if self.self_adjoint:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_inverse_times(
                                                                x=x,
                                                                spaces=spaces)

    def inverse_adjoint_times(self, x, spaces=None):
        """ Applies the inverse-adjoint-Operator to a given Field.

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
            the processed Field living on the domain space

        See Also
       --------

        """
        if self.self_adjoint:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self.domain

    # ---Added properties and methods---

    @abc.abstractproperty
    def self_adjoint(self):
        raise NotImplementedError
