# -*- coding: utf-8 -*-
from nifty.minimization import ConjugateGradient
from nifty.field import Field
from nifty.operators import EndomorphicOperator,\
                            FFTOperator


class PropagatorOperator(EndomorphicOperator):

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

        self.preconditioner = preconditioner

        if inverter is not None:
            self.inverter = inverter
        else:
            self.inverter = ConjugateGradient(
                                preconditioner=self.preconditioner)

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def field_type(self):
        return ()

    @property
    def implemented(self):
        return True

    @property
    def symmetric(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    def _S_times(self, x, spaces=None, types=None):
            transformed_x = self._fft_S(x,
                                        spaces=spaces,
                                        types=types)
            y = self._S(transformed_x, spaces=spaces, types=types)
            transformed_y = self._fft_S.inverse_times(y,
                                                      spaces=spaces,
                                                      types=types)
            result = x.copy_empty()
            result.set_val(transformed_y, copy=False)
            return result

    def _S_inverse_times(self, x, spaces=None, types=None):
            transformed_x = self._fft_S(x,
                                        spaces=spaces,
                                        types=types)
            y = self._S.inverse_times(transformed_x,
                                      spaces=spaces,
                                      types=types)
            transformed_y = self._fft_S.inverse_times(y,
                                                      spaces=spaces,
                                                      types=types)
            result = x.copy_empty()
            result.set_val(transformed_y, copy=False)
            return result

    def _times(self, x, spaces, types, x0=None):
        if x0 is None:
            x0 = Field(self.domain, val=0., dtype=x.dtype)

        (result, convergence) = self.inverter(A=self.inverse_times,
                                              b=x,
                                              x0=x0)
        return result

    def _inverse_times(self, x, spaces, types):
        pre_result = self._S_inverse_times(x, spaces, types)
        pre_result += self._likelihood_times(x)
        result = x.copy_empty()
        result.set_val(pre_result, copy=False)
        return result
