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
from ..multi_field import MultiField
from ..operators.einsum import MultiLinearEinsum
from ..operators.energy_operators import EnergyOperator
from ..operators.linear_operator import LinearOperator
from ..operators.multifield2vector import Multifield2Vector
from ..operators.sandwich_operator import SandwichOperator
from ..operators.simple_linear_operators import FieldAdapter
from ..sugar import full, makeField, makeDomain, from_random, is_fieldlike
from ..minimization.energy_adapter import StochasticEnergyAdapter
from ..utilities import myassert


def _eval(op, position):
    return op(position.extract(op.domain))


class MeanFieldVI:
    def __init__(self, initial_position, hamiltonian, n_samples, mirror_samples,
                 initial_sig=1, comm=None, nanisinf=False):
        """Collect the operators required for Gaussian mean-field variational
        inference.
        """
        Flat = Multifield2Vector(initial_position.domain)
        self._std = FieldAdapter(Flat.target, 'std').absolute()
        latent = FieldAdapter(Flat.target,'latent')
        self._mean = FieldAdapter(Flat.target, 'mean')
        self._generator = Flat.adjoint(self._mean + self._std * latent)
        self._entropy = GaussianEntropy(self._std.target) @ self._std
        self._mean = Flat.adjoint @ self._mean
        self._std = Flat.adjoint @ self._std
        pos = {'mean': Flat(initial_position)}
        if is_fieldlike(initial_sig):
            pos['std'] = Flat(initial_sig)
        else:
            pos['std'] = full(Flat.target, initial_sig)
        pos = MultiField.from_dict(pos)
        op = hamiltonian(self._generator) + self._entropy
        self._KL = StochasticEnergyAdapter.make(pos, op, ['latent',], n_samples,
                                    mirror_samples, nanisinf=nanisinf, comm=comm)
        self._samdom = latent.domain

    @property
    def mean(self):
        return _eval(self._mean,self._KL.position)

    @property
    def std(self):
        return _eval(self._std,self._KL.position)

    @property
    def entropy(self):
        return _eval(self._entropy,self._KL.position)

    def draw_sample(self):
        _, op = self._generator.simplify_for_constant_input(
                from_random(self._samdom))
        return op(self._KL.position)

    def minimize(self, minimizer):
        self._KL, _ = minimizer(self._KL)

class FullCovarianceVI:
    def __init__(self, position, hamiltonian, n_samples, mirror_samples,
                initial_sig=1, comm=None, nanisinf=False):
        """Collect the operators required for Gaussian full-covariance variational
        inference.
        """
        Flat = Multifield2Vector(position.domain)
        flat_domain = Flat.target[0]
        mat_space = DomainTuple.make((flat_domain,flat_domain))
        lat = FieldAdapter(Flat.target,'latent')
        LT = LowerTriangularInserter(mat_space)
        tri = FieldAdapter(LT.domain, 'cov')
        mean = FieldAdapter(flat_domain,'mean')
        cov = LT @ tri
        matmul_setup = lat.adjoint @ lat + cov.ducktape_left('co')
        MatMult = MultiLinearEinsum(matmul_setup.target,'ij,j->i',
                                    key_order=('co','latent'))

        self._generator = Flat.adjoint @ (mean + MatMult @ matmul_setup)

        diag_cov = (DiagonalSelector(cov.target) @ cov).absolute()
        self._entropy = GaussianEntropy(diag_cov.target) @ diag_cov
        diag_tri = np.diag(np.full(flat_domain.shape[0], initial_sig))
        pos = MultiField.from_dict(
                {'mean': Flat(position),
                 'cov': LT.adjoint(makeField(mat_space, diag_tri))})
        op = hamiltonian(self._generator) + self._entropy
        self._KL = StochasticEnergyAdapter.make(pos, op, ['latent',], n_samples,
                                    mirror_samples, nanisinf=nanisinf, comm=comm)
        self._mean = Flat.adjoint @ mean
        self._samdom = lat.domain

    @property
    def mean(self):
        return _eval(self._mean,self._KL.position)

    @property
    def entropy(self):
        return _eval(self._entropy,self._KL.position)

    def draw_sample(self):
        _, op = self._generator.simplify_for_constant_input(
                from_random(self._samdom))
        return op(self._KL.position)

    def minimize(self, minimizer):
        self._KL, _ = minimizer(self._KL)


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


class LowerTriangularInserter(LinearOperator):
    """Inserts the DOFs of a lower triangular matrix into a matrix.

    Parameters
    ----------
    target: Domain
        A two-dimensional domain with NxN entries.
    """

    def __init__(self, target):
        myassert(len(target.shape) == 2)
        myassert(target.shape[0] == target.shape[1])
        self._target = makeDomain(target)
        ndof = (target.shape[0]*(target.shape[0]+1))//2
        self._domain = makeDomain(UnstructuredDomain(ndof))
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
        The two-dimensional domain of the input field. Must be of shape NxN.
    """

    def __init__(self, domain):
        myassert(len(domain.shape) == 2)
        myassert(domain.shape[0] == domain.shape[1])
        self._domain = makeDomain(domain)
        self._target = makeDomain(UnstructuredDomain(domain.shape[0]))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return makeField(self._tgt(mode), np.diag(x.val))
