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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):
    """Represents a :class:`LinearOperator` which is endomorphic, i.e. one
    which has identical domain and target.
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

    def _dom(self, mode):
        return self._domain

    def _tgt(self, mode):
        return self._domain

    def _check_input(self, x, mode):
        self._check_mode(mode)
        if self.domain != x.domain:
            raise ValueError("The operator's and field's domains don't match.")
