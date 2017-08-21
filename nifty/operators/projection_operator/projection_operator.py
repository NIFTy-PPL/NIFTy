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

import numpy as np

from nifty.field import Field

from nifty.operators.endomorphic_operator import EndomorphicOperator


class ProjectionOperator(EndomorphicOperator):
    """ NIFTY class for projection operators.

    The NIFTY ProjectionOperator class is a class derived from the
    EndomorphicOperator.

    Parameters
    ----------
    projection_field : Field
        Field on which the operator projects
    default_spaces : tuple of ints *optional*
        Defines on which space(s) of a given field the Operator acts by
        default (default: None)

    Attributes
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain on which the Operator's input Field lives.
    target : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain in which the outcome of the operator lives. As the Operator
        is endomorphic this is the same as its domain.
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    self_adjoint : boolean
        Indicates whether the operator is self_adjoint or not.

    Raises
    ------
    TypeError
        Raised if
            * if projection_field is not a Field

    Notes
    -----


    Examples
    --------
    >>> x_space = RGSpace(5)
    >>> f1 = Field(x_space, val=3.)
    >>> f2 = Field(x_space, val=5.)
    >>> P = ProjectionOperator(f1)
    >>> res = P.times(f2)
    >>> res.val
    <distributed_data_object>
    array([ 225.,  225.,  225.,  225.,  225.])

    See Also
    --------
    EndomorphicOperator

    """

    # ---Overwritten properties and methods---

    def __init__(self, projection_field, default_spaces=None):
        super(ProjectionOperator, self).__init__(default_spaces)

        if not isinstance(projection_field, Field):
            raise TypeError("The projection_field must be a NIFTy-Field"
                            "instance.")
        self._projection_field = projection_field
        self._unitary = None

    def _times(self, x, spaces):
        # if the domain matches directly
        # -> multiply the fields directly
        if x.domain == self.domain:
            # here the actual multiplication takes place
            dotted = (self._projection_field * x).sum()
            return self._projection_field * dotted

        # if the distribution_strategy of self is sub-slice compatible to
        # the one of x, reshape the local data of self and apply it directly
        active_axes = []
        if spaces is None:
            active_axes = range(len(x.shape))
        else:
            for space_index in spaces:
                active_axes += x.domain_axes[space_index]

        axes_local_distribution_strategy = \
            x.val.get_axes_local_distribution_strategy(active_axes)
        if axes_local_distribution_strategy == \
           self._projection_field.distribution_strategy:
            local_projection_vector = \
                self._projection_field.val.get_local_data(copy=False)
        else:
            # create an array that is sub-slice compatible
            self.logger.warn("The input field is not sub-slice compatible to "
                             "the distribution strategy of the operator. "
                             "Performing an probably expensive "
                             "redistribution.")
            redistr_projection_val = self._projection_field.val.copy(
                distribution_strategy=axes_local_distribution_strategy)
            local_projection_vector = \
                redistr_projection_val.get_local_data(copy=False)

        local_x = x.val.get_local_data(copy=False)

        l = len(local_projection_vector.shape)
        sublist_projector = range(l)
        sublist_x = np.arange(len(local_x.shape)) + l

        for i in xrange(l):
            a = active_axes[i]
            sublist_x[a] = i

        dotted = np.einsum(local_projection_vector, sublist_projector,
                           local_x, sublist_x)

        # get those elements from sublist_x that haven't got contracted
        sublist_dotted = sublist_x[sublist_x >= l]

        remultiplied = np.einsum(local_projection_vector, sublist_projector,
                                 dotted, sublist_dotted,
                                 sublist_x)
        result_field = x.copy_empty(dtype=remultiplied.dtype)
        result_field.val.set_local_data(remultiplied, copy=False)
        return result_field

    def _inverse_times(self, x, spaces):
        raise NotImplementedError("The ProjectionOperator is a singular "
                                  "operator and therefore has no inverse.")

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._projection_field.domain

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = (self._projection_field.val == 1).all()
        return self._unitary

    @property
    def self_adjoint(self):
        return True

    # ---Added properties and methods---

    @property
    def projection_field(self):
        return self._projection_field
