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

import numpy as np

from d2o import distributed_data_object,\
                STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.basic_arithmetics import log as nifty_log
from nifty.config import nifty_configuration as gc
from nifty.field import Field
from nifty.operators.endomorphic_operator import EndomorphicOperator


class DiagonalOperator(EndomorphicOperator):
    """ NIFTY class for diagonal operators.

    The NIFTY DiagonalOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field pixel-wise with its
    diagonal.


    Parameters
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain on which the Operator's input Field lives.
    diagonal : {scalar, list, array, Field, d2o-object}
        The diagonal entries of the operator.
    bare : boolean
        Indicates whether the input for the diagonal is bare or not
        (default: False).
    copy : boolean
        Internal copy of the diagonal (default: True)
    distribution_strategy : string
        setting the prober distribution_strategy of the
        diagonal (default : None). In case diagonal is d2o-object or Field,
        their distribution_strategy is used as a fallback.
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
    distribution_strategy : string
        Defines the distribution_strategy of the distributed_data_object
        in which the diagonal entries are stored in.

    Raises
    ------

    Notes
    -----
    The ambiguity of bare or non-bare diagonal entries is based on the choice
    of a matrix representation of the operator in question. The naive choice
    of absorbing the volume weights into the matrix leads to a matrix-vector
    calculus with the non-bare entries which seems intuitive, though.
    The choice of keeping matrix entries and volume weights separate
    deals with the bare entries that allow for correct interpretation
    of the matrix entries; e.g., as variance in case of an covariance operator.

    Examples
    --------
    >>> x_space = RGSpace(5)
    >>> D = DiagonalOperator(x_space, diagonal=[1., 3., 2., 4., 6.])
    >>> f = Field(x_space, val=2.)
    >>> res = D.times(f)
    >>> res.val
    <distributed_data_object>
    array([ 2.,  6.,  4.,  8.,  12.])

    See Also
    --------
    EndomorphicOperator

    """

    # ---Overwritten properties and methods---

    def __init__(self, domain=(), diagonal=None, bare=False, copy=True,
                 distribution_strategy=None, default_spaces=None):
        super(DiagonalOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)

        if distribution_strategy is None:
            if isinstance(diagonal, distributed_data_object):
                distribution_strategy = diagonal.distribution_strategy
            elif isinstance(diagonal, Field):
                distribution_strategy = diagonal.distribution_strategy

        self._distribution_strategy = self._parse_distribution_strategy(
                               distribution_strategy=distribution_strategy,
                               val=diagonal)

        self.set_diagonal(diagonal=diagonal, bare=bare, copy=copy)

    def _times(self, x, spaces):
        return self._times_helper(x, spaces, operation=lambda z: z.__mul__)

    def _adjoint_times(self, x, spaces):
        return self._times_helper(x, spaces,
                                  operation=lambda z: z.adjoint().__mul__)

    def _inverse_times(self, x, spaces):
        return self._times_helper(x, spaces, operation=lambda z: z.__rdiv__)

    def _adjoint_inverse_times(self, x, spaces):
        return self._times_helper(x, spaces,
                                  operation=lambda z: z.adjoint().__rdiv__)

    def diagonal(self, bare=False, copy=True):
        """ Returns the diagonal of the Operator.

        Parameters
        ----------
        bare : boolean
            Whether the returned Field values should be bare or not.
        copy : boolean
            Whether the returned Field should be copied or not.

        Returns
        -------
        out : Field
            The diagonal of the Operator.

        """
        if bare:
            diagonal = self._diagonal.weight(power=-1)
        elif copy:
            diagonal = self._diagonal.copy()
        else:
            diagonal = self._diagonal
        return diagonal

    def inverse_diagonal(self, bare=False):
        """ Returns the inverse-diagonal of the operator.

        Parameters
        ----------
        bare : boolean
            Whether the returned Field values should be bare or not.

        Returns
        -------
        out : Field
            The inverse of the diagonal of the Operator.

        """        
        return 1./self.diagonal(bare=bare, copy=False)

    def trace(self, bare=False):
        """ Returns the trace the operator.

        Parameters
        ----------
        bare : boolean
            Whether the returned Field values should be bare or not.

        Returns
        -------
        out : scalar
            The trace of the Operator.

        """
        return self.diagonal(bare=bare, copy=False).sum()

    def inverse_trace(self, bare=False):
        """ Returns the inverse-trace of the operator.

        Parameters
        ----------
        bare : boolean
            Whether the returned Field values should be bare or not.

        Returns
        -------
        out : scalar
            The inverse of the trace of the Operator.

        """
        return self.inverse_diagonal(bare=bare).sum()

    def trace_log(self):
        """ Returns the trave-log of the operator.

        Returns
        -------
        out : scalar
            the trace of the logarithm of the Operator.

        """
        log_diagonal = nifty_log(self.diagonal(copy=False))
        return log_diagonal.sum()

    def determinant(self):
        """ Returns the determinant of the operator.

        Returns
        -------
        out : scalar
        out : scalar
            the determinant of the Operator

        """

        return self.diagonal(copy=False).val.prod()

    def inverse_determinant(self):
        """ Returns the inverse-determinant of the operator.

        Returns
        -------
        out : scalar
            the inverse-determinant of the Operator

        """

        return 1/self.determinant()

    def log_determinant(self):
        """ Returns the log-eterminant of the operator.

        Returns
        -------
        out : scalar
            the log-determinant of the Operator

        """

        return np.log(self.determinant())

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        if self._self_adjoint is None:
            self._self_adjoint = (self._diagonal.val.imag == 0).all()
        return self._self_adjoint

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = (self._diagonal.val *
                             self._diagonal.val.conjugate() == 1).all()
        return self._unitary

    # ---Added properties and methods---

    @property
    def distribution_strategy(self):
        """
        distribution_strategy : string
            Defines the way how the diagonal operator is distributed
            among the nodes. Available distribution_strategies are:
            'fftw', 'equal' and 'not'.

        Notes :
            https://arxiv.org/abs/1606.05385

        """

        return self._distribution_strategy

    def _parse_distribution_strategy(self, distribution_strategy, val):
        if distribution_strategy is None:
            if isinstance(val, distributed_data_object):
                distribution_strategy = val.distribution_strategy
            elif isinstance(val, Field):
                distribution_strategy = val.distribution_strategy
            else:
                self.logger.info("Datamodel set to default!")
                distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['all']:
            raise ValueError(
                    "Invalid distribution_strategy!")
        return distribution_strategy

    def set_diagonal(self, diagonal, bare=False, copy=True):
        """ Sets the diagonal of the Operator.

        Parameters
        ----------
        diagonal : {scalar, list, array, Field, d2o-object}
            The diagonal entries of the operator.
        bare : boolean
            Indicates whether the input for the diagonal is bare or not
            (default: False).
        copy : boolean
            Specifies if a copy of the input shall be made (default: True).

        """

        # use the casting functionality from Field to process `diagonal`
        f = Field(domain=self.domain,
                  val=diagonal,
                  distribution_strategy=self.distribution_strategy,
                  copy=copy)

        # weight if the given values were `bare` is True
        # do inverse weightening if the other way around
        if bare:
            # If `copy` is True, we won't change external data by weightening
            # Otherwise, inplace weightening would change the external field
            f.weight(inplace=copy)

        # Reset the self_adjoint property:
        self._self_adjoint = None

        # Reset the unitarity property
        self._unitary = None

        # store the diagonal-field
        self._diagonal = f

    def _times_helper(self, x, spaces, operation):
        # if the domain matches directly
        # -> multiply the fields directly
        if x.domain == self.domain:
            # here the actual multiplication takes place
            return operation(self.diagonal(copy=False))(x)

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
        if axes_local_distribution_strategy == self.distribution_strategy:
            local_diagonal = self._diagonal.val.get_local_data(copy=False)
        else:
            # create an array that is sub-slice compatible
            self.logger.warn("The input field is not sub-slice compatible to "
                             "the distribution strategy of the operator.")
            redistr_diagonal_val = self._diagonal.val.copy(
                distribution_strategy=axes_local_distribution_strategy)
            local_diagonal = redistr_diagonal_val.get_local_data(copy=False)

        reshaper = [x.shape[i] if i in active_axes else 1
                    for i in xrange(len(x.shape))]
        reshaped_local_diagonal = np.reshape(local_diagonal, reshaper)

        # here the actual multiplication takes place
        local_result = operation(reshaped_local_diagonal)(
                           x.val.get_local_data(copy=False))

        result_field = x.copy_empty(dtype=local_result.dtype)
        result_field.val.set_local_data(local_result, copy=False)
        return result_field
