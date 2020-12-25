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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

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

    def draw_sample(self, from_inverse=False):
        """Generates a sample from a Gaussian distribution with zero mean and
        covariance given by the operator.

        May or may not be implemented. Only optional.

        Parameters
        ----------
        from_inverse : bool (default : False)
            if True, the sample is drawn from the inverse of the operator

        Returns
        -------
        Field or MultiField
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError

    def draw_sample_with_dtype(self, dtype, from_inverse=False):
        """Generates a sample from a Gaussian distribution with zero mean,
        covariance given by the operator and specified data type.

        This method is implemented only for operators which actually draw
        samples (e.g. `DiagonalOperator`). Operators which process the sample
        (like `SandwichOperator`) implement only `draw_sample()`.

        May or may not be implemented. Only optional.

        Parameters
        ----------
        dtype : numpy.dtype or dict of numpy.dtype
            Dtype used for sampling from this operator. If the domain of `op`
            is a `MultiDomain`, the dtype can either be specified as one value
            for all components of the `MultiDomain` or in form of a dictionary
            whose keys need to conincide the with keys of the `MultiDomain`.
        from_inverse : bool (default : False)
            if True, the sample is drawn from the inverse of the operator

        Returns
        -------
        Field or MultiField
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError

    def get_sqrt(self):
        """Return operator op which obeys `self == op.adjoint @ op`.

        Note that this function is only implemented for operators with real
        spectrum.

        Returns
        -------
        EndomorphicOperator
            Operator which is the square root of `self`

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
