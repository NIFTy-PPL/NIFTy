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

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from .linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):
    """ NIFTy class for endomorphic operators.

    The  NIFTy EndomorphicOperator class is a class derived from the
    LinearOperator. By definition, domain and target are the same in
    EndomorphicOperator.
    """

    @property
    def target(self):
        """DomainTuple : returns :attr:`domain`

        Returns `self.domain`, because this is also the target domain
        for endomorphic operators."""
        return self._domain

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        """Generate a zero-mean sample

        Generates a sample from a Gaussian distribution with zero mean and
        covariance given by the operator.

        Parameters
        ----------
        from_inverse : bool (default : False)
            if True, the sample is drawn from the inverse of the operator
        dtype : numpy datatype (default : numpy.float64)
            the data type to be used for the sample

        Returns
        -------
        Field
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError
