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
from ..minimization.energy_adapter import StochasticEnergyAdapter
from ..multi_field import MultiField
from ..operators.einsum import MultiLinearEinsum
from ..operators.energy_operators import EnergyOperator
from ..operators.linear_operator import LinearOperator
from ..operators.multifield2vector import Multifield2Vector
from ..operators.sandwich_operator import SandwichOperator
from ..operators.simple_linear_operators import FieldAdapter
from ..sugar import from_random, full, is_fieldlike, makeDomain, makeField
from ..utilities import myassert


class MeanFieldVI:
    """Collect the operators required for Gaussian meanfield variational
    inference.

    Gaussian meanfield variational inference approximates some target
    distribution with a Gaussian distribution with a diagonal covariance
    matrix. The parameters of the approximation, in this case the mean and
    standard deviation, are obtained by minimizing a stochastic estimate of the
    Kullback-Leibler divergence between the target and the approximation.  In
    order to obtain gradients w.r.t the parameters, the reparametrization trick
    is employed, which separates the stochastic part of the approximation from
    a deterministic function, the generator. Samples from the approximation are
    drawn by processing samples from a standard Gaussian through this
    generator.

    Parameters
    ----------
    position : :class:`nifty8.field.Field`
        The initial estimate of the approximate mean parameter.
    hamiltonian : Energy
        Hamiltonian of the approximated probability distribution.
    n_samples : int
        Number of samples used to stochastically estimate the KL.
    mirror_samples : bool
        Whether the negative of the drawn samples are also used, as they are
        equally legitimate samples. If true, the number of used samples
        doubles. Mirroring samples stabilizes the KL estimate as extreme sample
        variation is counterbalanced. Since it improves stability in many
        cases, it is recommended to set `mirror_samples` to `True`.
    initial_sig : positive :class:`nifty8.field.Field` or positive float
        The initial estimate of the standard deviation.
    comm : MPI communicator or None
        If not None, samples will be distributed as evenly as possible across
        this communicator. If `mirror_samples` is set, then a sample and its
        mirror image will always reside on the same task.
    nanisinf : bool
        If true, nan energies which can happen due to overflows in the forward
        model are interpreted as inf. Thereby, the code does not crash on these
        occasions but rather the minimizer is told that the position it has
        tried is not sensible.
    """
    def __init__(self, position, hamiltonian, n_samples, mirror_samples,
                 initial_sig=1, comm=None, nanisinf=False):
        Flat = Multifield2Vector(position.domain)
        self._std = FieldAdapter(Flat.target, 'std').absolute()
        latent = FieldAdapter(Flat.target,'latent')
        self._mean = FieldAdapter(Flat.target, 'mean')
        self._generator = Flat.adjoint(self._mean + self._std * latent)
        self._entropy = GaussianEntropy(self._std.target) @ self._std
        self._mean = Flat.adjoint @ self._mean
        self._std = Flat.adjoint @ self._std
        pos = {'mean': Flat(position)}
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
        return self._mean.force(self._KL.position)

    @property
    def std(self):
        return self._std.force(self._KL.position)

    @property
    def entropy(self):
        return self._entropy.force(self._KL.position)

    @property
    def KL(self):
        return self._KL

    def draw_sample(self):
        _, op = self._generator.simplify_for_constant_input(
                from_random(self._samdom))
        return op(self._KL.position)

    def minimize(self, minimizer):
        self._KL, _ = minimizer(self._KL)


class FullCovarianceVI:
    """Collect the operators required for Gaussian full-covariance variational

    Gaussian meanfield variational inference approximates some target
    distribution with a Gaussian distribution with a diagonal covariance
    matrix. The parameters of the approximation, in this case the mean and a
    lower triangular matrix corresponding to a Cholesky decomposition of the
    covariance, are obtained by minimizing a stochastic estimate of the
    Kullback-Leibler divergence between the target and the approximation.  In
    order to obtain gradients w.r.t the parameters, the reparametrization trick
    is employed, which separates the stochastic part of the approximation from
    a deterministic function, the generator. Samples from the approximation are
    drawn by processing samples from a standard Gaussian through this
    generator.

    Note that the size of the covariance scales quadratically with the number
    of model parameters.

    Parameters
    ----------
    position : :class:`nifty8.field.Field`
        The initial estimate of the approximate mean parameter.
    hamiltonian : Energy
        Hamiltonian of the approximated probability distribution.
    n_samples : int
        Number of samples used to stochastically estimate the KL.
    mirror_samples : bool
        Whether the negative of the drawn samples are also used, as they are
        equally legitimate samples. If true, the number of used samples
        doubles. Mirroring samples stabilizes the KL estimate as extreme sample
        variation is counterbalanced. Since it improves stability in many
        cases, it is recommended to set `mirror_samples` to `True`.
    initial_sig : positive float
        The initial estimate for the standard deviation. Initially no
        correlation between the parameters is assumed.
    comm : MPI communicator or None
        If not None, samples will be distributed as evenly as possible across
        this communicator. If `mirror_samples` is set, then a sample and its
        mirror image will always reside on the same task.
    nanisinf : bool
        If true, nan energies which can happen due to overflows in the forward
        model are interpreted as inf. Thereby, the code does not crash on these
        occasions but rather the minimizer is told that the position it has
        tried is not sensible.
    """
    def __init__(self, position, hamiltonian, n_samples, mirror_samples,
                initial_sig=1, comm=None, nanisinf=False):
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
        return self._mean.force(self._KL.position)

    @property
    def entropy(self):
        return self._entropy.force(self._KL.position)

    @property
    def KL(self):
        return self._KL

    def draw_sample(self):
        _, op = self._generator.simplify_for_constant_input(
                from_random(self._samdom))
        return op(self._KL.position)

    def minimize(self, minimizer):
        self._KL, _ = minimizer(self._KL)


class GaussianEntropy(EnergyOperator):
    """Entropy of a Gaussian distribution given the diagonal of a triangular
    decomposition of the covariance.

    As metric a `SandwichOperator` of the Jacobian is used. This is not a
    proper Fisher metric but may be useful for second order minimization.

    Parameters
    ----------
    domain: Domain, DomainTuple, list of Domain
        The domain of the diagonal.
    """

    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)

    def apply(self, x):
        self._check_input(x)
        if isinstance(x, Field):
             if not np.issubdtype(x.dtype, np.floating):
                 raise NotImplementedError("only real fields are allowed")
        if isinstance(x, MultiField):
             for key in x.keys():
                 if not np.issubdtype(x[key].dtype, np.floating):
                     raise NotImplementedError("only real fields are allowed")
        res = (x*x).scale(2*np.pi*np.e).log().sum().scale(-0.5)
        if not isinstance(x, Linearization):
            return res
        if not x.want_metric:
            return res
        return res.add_metric(SandwichOperator.make(res.jac))


class LowerTriangularInserter(LinearOperator):
    """Insert the entries of a lower triangular matrix into a matrix.

    Parameters
    ----------
    target: Domain, DomainTuple, list of Domain
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
    domain: Domain, DomainTuple, list of Domain
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
