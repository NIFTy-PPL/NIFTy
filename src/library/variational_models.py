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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..linearization import Linearization
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..operators.einsum import MultiLinearEinsum
from ..operators.energy_operators import EnergyOperator
from ..operators.linear_operator import LinearOperator
from ..operators.multifield2vector import Multifield2Vector
from ..operators.sandwich_operator import SandwichOperator
from ..operators.simple_linear_operators import FieldAdapter, PartialExtractor
from ..sugar import domain_union, from_random, full, makeField


class MeanfieldModel():
    """Collect the operators required for Gaussian mean-field variational
    inference.

    Parameters
    ----------
    domain: MultiDomain
        The domain of the model parameters.
    """

    def __init__(self, domain):
        self.domain = MultiDomain.make(domain)
        self.Flat = Multifield2Vector(self.domain)

        self.std = FieldAdapter(self.Flat.target,'var').absolute()
        self.latent = FieldAdapter(self.Flat.target,'latent')
        self.mean = FieldAdapter(self.Flat.target,'mean')
        self.generator = self.Flat.adjoint(self.mean + self.std * self.latent)
        self.entropy = GaussianEntropy(self.std.target) @ self.std

    def get_initial_pos(self, initial_mean=None, initial_sig = 1):
        """Provide an initial position for a given mean parameter vector and an
        initial standard deviation.

        Parameters
        ----------
        initial_mean: MultiField
            The initial mean of the variational approximation. If not None, a
            Gaussian sample with mean zero and standard deviation of 0.1 is
            used. Default: None
        initial_sig: positive float
            The initial standard deviation shared by all parameters. Default: 1
        """

        initial_pos = from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = full(self.generator.domain['latent'], 0.)
        initial_pos['var'] = full(self.generator.domain['var'], initial_sig)

        if initial_mean is None:
            initial_mean = 0.1*from_random(self.generator.target)

        initial_pos['mean'] = self.Flat(initial_mean)
        return MultiField.from_dict(initial_pos)


class FullCovarianceModel():
    """Collect the operators required for Gaussian full-covariance variational
    inference.

    Parameters
    ----------
    domain: MultiDomain
        The domain of the model parameters.
    """

    def __init__(self, domain):
        self.domain = MultiDomain.make(domain)
        self.Flat = Multifield2Vector(self.domain)
        one_space = UnstructuredDomain(1)
        self.flat_domain = self.Flat.target[0]
        N_tri = self.flat_domain.shape[0]*(self.flat_domain.shape[0]+1)//2
        triangular_space = DomainTuple.make(UnstructuredDomain(N_tri))
        tri = FieldAdapter(triangular_space, 'cov')
        mat_space = DomainTuple.make((self.flat_domain,self.flat_domain))
        lat_mat_space = DomainTuple.make((one_space,self.flat_domain))
        lat = FieldAdapter(lat_mat_space,'latent')
        LT = LowerTriangularProjector(triangular_space,mat_space)
        mean = FieldAdapter(self.flat_domain,'mean')
        cov = LT @ tri
        co = FieldAdapter(cov.target, 'co')

        matmul_setup_dom = domain_union((co.domain,lat.domain))
        co_part = PartialExtractor(matmul_setup_dom, co.domain)
        lat_part = PartialExtractor(matmul_setup_dom, lat.domain)
        matmul_setup = lat_part.adjoint @ lat.adjoint @ lat + co_part.adjoint @ co.adjoint @ cov
        MatMult = MultiLinearEinsum(matmul_setup.target,'ij,ki->jk', key_order=('co','latent'))

        Resp = Respacer(MatMult.target, mean.target)
        self.generator = self.Flat.adjoint @ (mean + Resp @ MatMult @ matmul_setup)

        Diag = DiagonalSelector(cov.target, self.Flat.target)
        diag_cov = Diag(cov).absolute()
        self.entropy = GaussianEntropy(diag_cov.target) @ diag_cov

    def get_initial_pos(self, initial_mean=None, initial_sig=1):
        """Provide an initial position for a given mean parameter vector and a
        diagonal covariance with an initial standard deviation.

        Parameters
        ----------
        initial_mean: MultiField
            The initial mean of the variational approximation. If not None, a
            Gaussian sample with mean zero and standard deviation of 0.1 is
            used. Default: None
        initial_sig: positive float
            The initial standard deviation shared by all parameters. Default: 1
        """
        initial_pos = from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = full(self.generator.domain['latent'], 0.)
        diag_tri = np.diag(np.full(self.flat_domain.shape[0], initial_sig))[np.tril_indices(self.flat_domain.shape[0])]
        initial_pos['cov'] = makeField(self.generator.domain['cov'], diag_tri)
        if initial_mean is None:
            initial_mean = 0.1*from_random(self.generator.target)
        initial_pos['mean'] = self.Flat(initial_mean)
        return MultiField.from_dict(initial_pos)


class GaussianEntropy(EnergyOperator):
    """Calculate the entropy of a Gaussian distribution given the diagonal of a
    triangular decomposition of the covariance.

    Parameters
    ----------
    domain: Domain
        The domain of the diagonal.
    """

    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        res = -0.5*(2*np.pi*np.e*x**2).log().sum()
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        if not x.want_metric:
            return res
        # FIXME not sure about metric
        return res.add_metric(SandwichOperator.make(res.jac))


class LowerTriangularProjector(LinearOperator):
    """Project the DOFs of a triangular matrix into the matrix form.

    Parameters
    ----------
    domain: Domain
        A one-dimensional domain containing N(N+1)/2 DOFs of a triangular
        matrix.
    target: Domain
        A two-dimensional domain with NxN entries.
    """

    def __init__(self, domain, target):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._indices = np.tril_indices(target.shape[0])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = np.zeros(self._target.shape)
            res[self._indices] = x
        else:
            res = x[self._indices].reshape(self._domain.shape)
        return makeField(self._tgt(mode), res)


class DiagonalSelector(LinearOperator):
    """Extract the diagonal of a two-dimensional field.

    Parameters
    ----------
    domain: Domain
        The two-dimensional domain of the input field
    target: Domain
        The one-dimensional domain on which the diagonal of the input field is
        defined.
    """

    def __init__(self, domain, target):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = np.diag(x.val)
        if mode == self.ADJOINT_TIMES:
            x = x.reshape(self._domain.shape)
        return makeField(self._tgt(mode), x)


class Respacer(LinearOperator):
    """Re-map a field from one domain to another one with the same amounts of
    DOFs. Wrapps the numpy.reshape method.

    Parameters
    ----------
    domain: Domain
        The domain of the input field.
    target: Domain
        The domain of the output field.
    """

    def __init__(self, domain, target):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return makeField(self._tgt(mode), x.val.reshape(self._tgt(mode).shape))
