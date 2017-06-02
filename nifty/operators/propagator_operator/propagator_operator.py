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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from nifty.operators import EndomorphicOperator,\
                            FFTOperator,\
                            InvertibleOperatorMixin


class PropagatorOperator(InvertibleOperatorMixin, EndomorphicOperator):
    """ NIFTY Propagator Operator D.

    The propagator operator D, is known from the Wiener Filter.
    Its inverse functional form might look like:
    D = (S^(-1) + M)^(-1)
    D = (S^(-1) + N^(-1))^(-1)
    D = (S^(-1) + R^(\dagger) N^(-1) R)^(-1)

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
    >>> x_space = RGSpace(4)
    >>> k_space = RGRGTransformation.get_codomain(x_space)
    >>> f = Field(x_space, val=[2, 4, 6, 8])
    >>> S = create_power_operator(k_space, spec=1)
    >>> N = DiagonalOperaor(f.domain, diag=1)
    >>> D = PropagatorOperator(S=S, N=N) # D^{-1} = S^{-1} + N^{-1}
    >>> D(f).val
    <distributed_data_object>
    array([ 1.,  2.,  3.,  4.]

    See Also
    --------
    Scientific reference
    https://arxiv.org/abs/0806.3474

    """

    # ---Overwritten properties and methods---

    def __init__(self, S=None, M=None, R=None, N=None, inverter=None,
                 preconditioner=None, default_spaces=None):
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
        if M is not None:
            self._domain = M.domain
            self._likelihood_times = M.times

        elif N is None:
            raise ValueError("Either M or N must be given!")

        elif R is not None:
            self._domain = R.domain
            self._likelihood_times = \
                lambda z: R.adjoint_times(N.inverse_times(R.times(z)))
        else:
            self._domain = N.domain
            self._likelihood_times = lambda z: N.inverse_times(z)

        self._S = S
        self._fft_S = FFTOperator(self._domain, target=self._S.domain)

        if preconditioner is None:
            preconditioner = self._S_times

        super(PropagatorOperator, self).__init__(inverter=inverter,
                                                 preconditioner=preconditioner,
                                                 default_spaces=default_spaces)

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

    def _S_times(self, x, spaces=None):
            transformed_x = self._fft_S(x, spaces=spaces)
            y = self._S(transformed_x, spaces=spaces)
            transformed_y = self._fft_S.inverse_times(y, spaces=spaces)
            result = x.copy_empty()
            result.set_val(transformed_y, copy=False)
            return result

    def _S_inverse_times(self, x, spaces=None):
            transformed_x = self._fft_S(x, spaces=spaces)
            y = self._S.inverse_times(transformed_x, spaces=spaces)
            transformed_y = self._fft_S.inverse_times(y, spaces=spaces)
            result = x.copy_empty()
            result.set_val(transformed_y, copy=False)
            return result

    def _inverse_times(self, x, spaces):
        pre_result = self._S_inverse_times(x, spaces)
        pre_result += self._likelihood_times(x)
        result = x.copy_empty()
        result.set_val(pre_result, copy=False)
        return result
