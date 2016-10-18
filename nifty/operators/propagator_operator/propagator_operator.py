# -*- coding: utf-8 -*-
import numpy as np
from nifty.minimization import ConjugateGradient
from nifty.nifty_utilities import get_default_codomain
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

        fft_S = FFTOperator(S.domain, target=self._domain)
        self._S_times = lambda z: fft_S(S(fft_S.inverse_times(z)))
        self._S_inverse_times = lambda z: fft_S(S.inverse_times(
                                          fft_S.inverse_times(z)))

        if preconditioner is None:
            preconditioner = self._S_times

        self.preconditioner = preconditioner

        if inverter is not None:
            self.inverter = inverter
        else:
            self.inverter = ConjugateGradient(
                                preconditioner=self.preconditioner)

        self.x0 = None

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

    def _times(self, x, spaces, types):
        if self.x0 is None:
            x0 = Field(self.domain, val=0., dtype=np.complex128)
        else:
            x0 = self.x0
        (result, convergence) = self.inverter(A=self.inverse_times,
                                              b=x,
                                              x0=x0)
        self.x0 = result
        return result

    def _inverse_times(self, x, spaces, types):
        result = self._S_inverse_times(x)
        result += self._likelihood_times(x)
        return result
