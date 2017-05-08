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


class PropagatorOperator(InvertibleOperatorMixin, EndomorphicOperator):

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
