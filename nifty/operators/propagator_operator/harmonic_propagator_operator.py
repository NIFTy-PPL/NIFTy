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

from nifty.operators import EndomorphicOperator,\
                            FFTOperator,\
                            InvertibleOperatorMixin


class HarmonicPropagatorOperator(InvertibleOperatorMixin, EndomorphicOperator):
    """ NIFTY Harmonic Propagator Operator D.

    The propagator operator D, is known from the Wiener Filter.
    Its inverse functional form might look like:
    D = (S^(-1) + M)^(-1)
    D = (S^(-1) + N^(-1))^(-1)
    D = (S^(-1) + R^(\dagger) N^(-1) R)^(-1)
    In contrast to the PropagatorOperator the inference is done in the
    harmonic space.

    Parameters
    ----------
        S : LinearOperator
            Covariance of the signal prior.
        M : LinearOperator
            Likelihood contribution.
        R : LinearOperator
            Response operator translating signal to (noiseless) data.
        N : LinearOperator
            Covariance of the noise prior or the likelihood, respectively.
        inverter : class to invert explicitly defined operators
            (default:ConjugateGradient)
        preconditioner : Field
            numerical preconditioner to speed up convergence
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
    ValueError
        is raised if
            * neither N nor M is given

    Notes
    -----

    Examples
    --------

    See Also
    --------
    Scientific reference
    https://arxiv.org/abs/0806.3474

    """

    # ---Overwritten properties and methods---

    def __init__(self, S=None, M=None, R=None, N=None, inverter=None,
                 preconditioner=None):
        """
            Sets the standard operator properties and `codomain`, `_A1`, `_A2`,
            and `RN` if required.

            Parameters
            ----------
            S : operator
                Covariance of the signal prior.
            M : operator
                Likelihood contribution.
            R : operator
                Response operator translating signal to (noiseless) data.
            N : operator
                Covariance of the noise prior or the likelihood, respectively.

        """
        # infer domain, and target
        # infer domain, and target
        if M is not None:
            self._codomain = M.domain
            self._likelihood = M.times

        elif N is None:
            raise ValueError("Either M or N must be given!")

        elif R is not None:
            self._codomain = R.domain
            self._likelihood = \
                lambda z: R.adjoint_times(N.inverse_times(R.times(z)))
        else:
            self._codomain = N.domain
            self._likelihood = lambda z: N.inverse_times(z)

        self._domain = S.domain
        self._S = S
        self._fft_S = FFTOperator(self._domain, target=self._codomain)

        super(HarmonicPropagatorOperator, self).__init__(inverter=inverter,
                                                 preconditioner=preconditioner)

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---
    def _likelihood_times(self, x, spaces=None):
        transformed_x = self._fft_S.times(x, spaces=spaces)
        y = self._likelihood(transformed_x)
        transformed_y = self._fft_S.adjoint_times(y, spaces=spaces)
        result = x.copy_empty()
        result.set_val(transformed_y, copy=False)
        return result

    def _inverse_times(self, x, spaces):
        pre_result = self._S.inverse_times(x, spaces)
        pre_result += self._likelihood_times(x)
        result = x.copy_empty()
        result.set_val(pre_result, copy=False)
        return result
