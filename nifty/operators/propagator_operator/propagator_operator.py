# -*- coding: utf-8 -*-
from nifty.minimization import ConjugateGradient
from nifty.nifty_utilities import get_default_codomain
from nifty.operators import EndomorphicOperator,\
                            FFTOperator

import logging
logger = logging.getLogger('NIFTy.PropagatorOperator')


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
            fft_RN = FFTOperator(self._domain, target=N.domain)
            self._likelihood_times = \
                lambda z: R.adjoint_times(
                            fft_RN.inverse_times(N.inverse_times(
                                fft_RN(R.times(z)))))
        else:
            self._domain = (get_default_codomain(N.domain[0]),)
            fft_RN = FFTOperator(self._domain, target=N.domain)
            self._likelihood_times = \
                lambda z: fft_RN.inverse_times(N.inverse_times(
                                fft_RN(z)))

        fft_S = FFTOperator(S.domain, self._domain)
        self._S_times = lambda z: fft_S.inverse_times(S(fft_S(z)))
        self._S_inverse_times = lambda z: fft_S.inverse_times(
                                            S.inverse_times(fft_S(z)))

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

    def _times(self, x, spaces, types):
        (result, convergence) = self.inverter(A=self._inverse_times, b=x)
        return result

    def _inverse_multiply(self, x, **kwargs):
        result = self._S_inverse_times(x)
        result += self._likelihood_times(x)
        return result
