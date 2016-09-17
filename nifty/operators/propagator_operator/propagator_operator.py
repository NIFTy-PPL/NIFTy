# -*- coding: utf-8 -*-
from nifty.config import about
from nifty.minimization import ConjugateGradient

from nifty.operators.linear_operator import LinearOperator


class PropagatorOperator(LinearOperator):
    """
        ..                                                                            __
        ..                                                                          /  /_
        ..      _______   _____   ______    ______    ____ __   ____ __   ____ __  /   _/  ______    _____
        ..    /   _   / /   __/ /   _   | /   _   | /   _   / /   _   / /   _   / /  /   /   _   | /   __/
        ..   /  /_/  / /  /    /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_  /  /_/  / /  /
        ..  /   ____/ /__/     \______/ /   ____/  \______|  \___   /  \______|  \___/  \______/ /__/     operator class
        .. /__/                        /__/                 /______/

        NIFTY subclass for propagator operators (of a certain family)

        The propagator operators :math:`D` implemented here have an inverse
        formulation like :math:`(S^{-1} + M)`, :math:`(S^{-1} + N^{-1})`, or
        :math:`(S^{-1} + R^\dagger N^{-1} R)` as appearing in Wiener filter
        theory.

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

        See Also
        --------
        conjugate_gradient

        Notes
        -----
        The propagator will puzzle the operators `S` and `M` or `R`, `N` or
        only `N` together in the predefined from, a domain is set
        automatically. The application of the inverse is done by invoking a
        conjugate gradient.
        Note that changes to `S`, `M`, `R` or `N` auto-update the propagator.

        Examples
        --------
        >>> f = field(rg_space(4), val=[2, 4, 6, 8])
        >>> S = power_operator(f.target, spec=1)
        >>> N = diagonal_operator(f.domain, diag=1)
        >>> D = propagator_operator(S=S, N=N) # D^{-1} = S^{-1} + N^{-1}
        >>> D(f).val
        array([ 1.,  2.,  3.,  4.])

        Attributes
        ----------
        domain : space
            A space wherein valid arguments live.
        codomain : space
            An alternative space wherein valid arguments live; commonly the
            codomain of the `domain` attribute.
        sym : bool
            Indicates that the operator is self-adjoint.
        uni : bool
            Indicates that the operator is not unitary.
        imp : bool
            Indicates that volume weights are implemented in the `multiply`
            instance methods.
        target : space
            The space wherein the operator output lives.
        _A1 : {operator, function}
            Application of :math:`S^{-1}` to a field.
        _A2 : {operator, function}
            Application of all operations not included in `A1` to a field.
        RN : {2-tuple of operators}, *optional*
            Contains `R` and `N` if given.

    """

    # ---Overwritten properties and methods---

    def __init__(self, S=None, M=None, R=None, N=None, inverter=None):
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

        self.S = S
        self.S_inverse_times = self.S.inverse_times

        # build up the likelihood contribution
        (self.M_times,
         M_domain,
         M_field_type,
         M_target,
         M_field_type_target) = self._build_likelihood_contribution(M, R, N)

        # assert that S and M have matching domains
        if not (self.domain == M_domain and
                self.field_type == M_target and
                self.target == M_target and
                self.field_type_target == M_field_type_target):
            raise ValueError(about._errors.cstring(
                "ERROR: The domains and targets of the prior " +
                "signal covariance and the likelihood contribution must be " +
                "the same in the sense of '=='."))

        if inverter is not None:
            self.inverter = inverter
        else:
            self.inverter = conjugate_gradient()

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self.S.domain

    @property
    def field_type(self):
        return self.S.field_type

    @property
    def target(self):
        return self.S.target

    @property
    def field_type_target(self):
        return self.S.field_type_target

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

    def _build_likelihood_contribution(self, M, R, N):
        # if a M is given, return its times method and its domains
        # supplier and discard R and N
        if M is not None:
            return (M.times, M.domain, M.field_type, M.target, M.cotarget)

        if N is not None:
            if R is not None:
                return (lambda z: R.adjoint_times(N.inverse_times(R.times(z))),
                        R.domain, R.field_type, R.domain, R.field_type)
            else:
                return (N.inverse_times,
                        N.domain, N.field_type, N.target, N.field_type_target)
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: At least M or N must be given."))

    def _multiply(self, x, W=None, spam=None, reset=None, note=False,
                  x0=None, tol=1E-4, clevel=1, limii=None, **kwargs):

        if W is None:
            W = self.S
        (result, convergence) = self.inverter(A=self._inverse_multiply,
                                              b=x,
                                              W=W,
                                              spam=spam,
                                              reset=reset,
                                              note=note,
                                              x0=x0,
                                              tol=tol,
                                              clevel=clevel,
                                              limii=limii)
        # evaluate
        if not convergence:
            about.warnings.cprint("WARNING: conjugate gradient failed.")

        return result

    def _inverse_multiply(self, x, **kwargs):
        result = self.S_inverse_times(x)
        result += self.M_times(x)
        return result
