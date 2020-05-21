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

import numpy as np

from ..minimization.conjugate_gradient import ConjugateGradient
from ..minimization.quadratic_energy import QuadraticEnergy
from ..multi_domain import MultiDomain
from .endomorphic_operator import EndomorphicOperator
from .operator import Operator


class SamplingEnabler(EndomorphicOperator):
    """Class which acts as a operator object built of (`likelihood` + `prior`)
    and enables sampling from its inverse even if the operator object
    itself does not support it.


    Parameters
    ----------
    likelihood : :class:`EndomorphicOperator`
        Metric of the likelihood
    prior : :class:`EndomorphicOperator`
        Metric of the prior
    iteration_controller : :class:`IterationController`
        The iteration controller to use for the iterative numerical inversion
        done by a :class:`ConjugateGradient` object.
    approximation : :class:`LinearOperator`, optional
        if not None, this linear operator should be an approximation to the
        operator, which supports the operation modes that the operator doesn't
        have. It is used as a preconditioner during the iterative inversion,
        to accelerate convergence.
    start_from_zero : boolean
        If true, the conjugate gradient algorithm starts from a field filled
        with zeros. Otherwise, it starts from a prior samples. Default is
        False.
    """

    def __init__(self, likelihood, prior, iteration_controller,
                 approximation=None, start_from_zero=False):
        if not isinstance(likelihood, Operator) or not isinstance(prior, Operator):
            raise TypeError
        self._likelihood = likelihood
        self._prior = prior
        self._ic = iteration_controller
        self._approximation = approximation
        self._start_from_zero = bool(start_from_zero)
        self._op = likelihood + prior
        self._domain = self._op.domain
        self._capability = self._op.capability
        self.apply = self._op.apply

    def draw_sample(self, from_inverse=False):
        try:
            return self._op.draw_sample(from_inverse)
        except NotImplementedError:
            if not from_inverse:
                raise ValueError("from_inverse must be True here")
            if self._start_from_zero:
                b = self._op.draw_sample()
                energy = QuadraticEnergy(0*b, self._op, b)
            else:
                s = self._prior.draw_sample(from_inverse=True)
                sp = self._prior(s)
                nj = self._likelihood.draw_sample()
                energy = QuadraticEnergy(s, self._op, sp + nj,
                                         _grad=self._likelihood(s) - nj)
            inverter = ConjugateGradient(self._ic)
            if self._approximation is not None:
                energy, convergence = inverter(
                    energy, preconditioner=self._approximation.inverse)
            else:
                energy, convergence = inverter(energy)
            return energy.position

    def draw_sample_with_dtype(self, dtype, from_inverse=False):
        return self._op.draw_sample_with_dtype(dtype, from_inverse)

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            "SamplingEnabler:",
            indent("\n".join((
                "Likelihood:", self._likelihood.__repr__(),
                "Prior:", self._prior.__repr__())))))


class SamplingDtypeSetter(EndomorphicOperator):
    """Class that adds the information whether the operator at hand is the
    covariance of a real-valued Gaussian or a complex-valued Gaussian
    probability distribution.

    This wrapper class shall address the following ambiguity which arises when
    drawing a sampling from a Gaussian distribution with zero mean and given
    covariance. E.g. a `ScalingOperator` with `1.` on its diagonal can be
    viewed as the covariance operator of both a real-valued and complex-valued
    Gaussian distribution. `SamplingDtypeSetter` specifies this data type.

    Parameters
    ----------
    op : EndomorphicOperator
        Operator which shall be supplemented with a dtype for sampling. Needs
        to be positive definite, hermitian and needs to implement the method
        `draw_sample_with_dtype()`. Note that these three properties are not
        checked in the constructor.
    dtype : numpy.dtype or dict of numpy.dtype
        Dtype used for sampling from this operator. If the domain of `op` is a
        `MultiDomain`, the dtype can either be specified as one value for all
        components of the `MultiDomain` or in form of a dictionary whose keys
        need to conincide the with keys of the `MultiDomain`.
    """
    def __init__(self, op, dtype):
        if isinstance(op, SamplingDtypeSetter):
            if op._dtype != dtype:
                raise ValueError('Dtype for sampling already set to another dtype.')
            op = op._op
        if not isinstance(op, EndomorphicOperator):
            raise TypeError
        if not hasattr(op, 'draw_sample_with_dtype'):
            raise TypeError
        if isinstance(dtype, dict):
            dtype = {kk: np.dtype(vv) for kk, vv in dtype.items()}
        else:
            dtype = np.dtype(dtype)
        if isinstance(op.domain, MultiDomain):
            if isinstance(dtype, np.dtype):
                dtype = {kk: dtype for kk in op.domain.keys()}
            if set(dtype.keys()) != set(op.domain.keys()):
                raise TypeError
        self._dtype = dtype
        self._domain = op.domain
        self._capability = op.capability
        self.apply = op.apply
        self._op = op

    def draw_sample(self, from_inverse=False):
        return self._op.draw_sample_with_dtype(self._dtype,
                                               from_inverse=from_inverse)

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            f"SamplingDtypeSetter {self._dtype}:",
            indent(self._op.__repr__())))
