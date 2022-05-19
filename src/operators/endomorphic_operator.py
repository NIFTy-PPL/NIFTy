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
        :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError

    @property
    def sampling_dtype(self):
        """Sampling dtype if operator is used as covariance operator."""
        if hasattr(self, "_dtype"):
            return self._dtype
        return None

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
