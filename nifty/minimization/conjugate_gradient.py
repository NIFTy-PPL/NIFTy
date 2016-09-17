# -*- coding: utf-8 -*-

import numpy as np
from nifty.config import notification, about

class ConjugateGradient(object):
    """
        ..      _______       ____ __
        ..    /  _____/     /   _   /
        ..   /  /____  __  /  /_/  / __
        ..   \______//__/  \____  //__/  class
        ..                /______/

        NIFTY tool class for conjugate gradient

        This tool minimizes :math:`A x = b` with respect to `x` given `A` and
        `b` using a conjugate gradient; i.e., a step-by-step minimization
        relying on conjugated gradient directions. Further, `A` is assumed to
        be a positive definite and self-adjoint operator. The use of a
        preconditioner `W` that is roughly the inverse of `A` is optional.
        For details on the methodology refer to [#]_, for details on usage and
        output, see the notes below.

        Parameters
        ----------
        A : {operator, function}
            Operator `A` applicable to a field.
        b : field
            Resulting field of the operation `A(x)`.
        W : {operator, function}, *optional*
            Operator `W` that is a preconditioner on `A` and is applicable to a
            field (default: None).
        spam : function, *optional*
            Callback function which is given the current `x` and iteration
            counter each iteration (default: None).
        reset : integer, *optional*
            Number of iterations after which to restart; i.e., forget previous
            conjugated directions (default: sqrt(b.dim)).
        note : bool, *optional*
            Indicates whether notes are printed or not (default: False).

        See Also
        --------
        scipy.sparse.linalg.cg

        Notes
        -----
        After initialization by `__init__`, the minimizer is started by calling
        it using `__call__`, which takes additional parameters. Notifications,
        if enabled, will state the iteration number, current step widths
        `alpha` and `beta`, the current relative residual `delta` that is
        compared to the tolerance, and the convergence level if changed.
        The minimizer will exit in three states: DEAD if alpha becomes
        infinite, QUIT if the maximum number of iterations is reached, or DONE
        if convergence is achieved. Returned will be the latest `x` and the
        latest convergence level, which can evaluate ``True`` for the exit
        states QUIT and DONE.

        References
        ----------
        .. [#] J. R. Shewchuk, 1994, `"An Introduction to the Conjugate
            Gradient Method Without the Agonizing Pain"
            <http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf>`_

        Examples
        --------
        >>> b = field(point_space(2), val=[1, 9])
        >>> A = diagonal_operator(b.domain, diag=[4, 3])
        >>> x,convergence = conjugate_gradient(A, b, note=True)(tol=1E-4, clevel=3)
        iteration : 00000001   alpha = 3.3E-01   beta = 1.3E-03   delta = 3.6E-02
        iteration : 00000002   alpha = 2.5E-01   beta = 7.6E-04   delta = 1.0E-03
        iteration : 00000003   alpha = 3.3E-01   beta = 2.5E-04   delta = 1.6E-05   convergence level : 1
        iteration : 00000004   alpha = 2.5E-01   beta = 1.8E-06   delta = 2.1E-08   convergence level : 2
        iteration : 00000005   alpha = 2.5E-01   beta = 2.2E-03   delta = 1.0E-09   convergence level : 3
        ... done.
        >>> bool(convergence)
        True
        >>> x.val # yields 1/4 and 9/3
        array([ 0.25,  3.  ])

        Attributes
        ----------
        A : {operator, function}
            Operator `A` applicable to a field.
        x : field
            Current field.
        b : field
            Resulting field of the operation `A(x)`.
        W : {operator, function}
            Operator `W` that is a preconditioner on `A` and is applicable to a
            field; can be ``None``.
        spam : function
            Callback function which is given the current `x` and iteration
            counter each iteration; can be ``None``.
        reset : integer
            Number of iterations after which to restart; i.e., forget previous
            conjugated directions (default: sqrt(b.dim)).
        note : notification
            Notification instance.

    """
    def __init__(self):
        pass

    def __call__(self, A, b, x0=None, W=None, spam=None, reset=None,
                 note=False, **kwargs):
        """
            Runs the conjugate gradient minimization.

            Parameters
            ----------
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        if hasattr(A, "__call__"):
            self.A = A
        else:
            raise AttributeError(about._errors.cstring(
                "ERROR: A must be callable!"))

        self.b = b

        if W is None or hasattr(W, "__call__"):
            self.W = W
        else:
            raise AttributeError(about._errors.cstring(
                "ERROR: W must be None or callable!"))

        self.spam = spam

        if reset is None:
            self.reset = max(2, int(np.sqrt(b.domain.dim)))
        else:
            self.reset = max(2, int(reset))

        self.note = notification(default=bool(note))

        self.x = self.b.copy_empty()
        self.x.set_val(new_val=0)
        self.x.set_val(new_val=x0)

        if self.W is None:
            return self._calc_without(**kwargs)
        else:
            return self._calc_with(**kwargs)

    def _calc_without(self, tol=1E-4, clevel=1, limii=None):
        clevel = int(clevel)
        if limii is None:
            limii = 10*self.b.domain.dim
        else:
            limii = int(limii)

        r = self.b-self.A(self.x)
        print ('r', r.val)
        d = self.b.copy_empty()
        d.set_val(new_val = r.get_val())
        gamma = r.dot(d)
        if gamma==0:
            return self.x, clevel+1
        delta_ = np.absolute(gamma)**(-0.5)


        convergence = 0
        ii = 1
        while(True):

            # print ('gamma', gamma)
            q = self.A(d)
            # print ('q', q.val)
            alpha = gamma/d.dot(q) ## positive definite
            if np.isfinite(alpha) == False:
                self.note.cprint(
                    "\niteration : %08u   alpha = NAN\n... dead."%ii)
                return self.x, 0
            self.x += d * alpha
            # print ('x', self.x.val)
            if np.signbit(np.real(alpha)) == True:
                about.warnings.cprint(
                    "WARNING: positive definiteness of A violated.")
                r = self.b-self.A(self.x)
            elif (ii%self.reset) == 0:
                r = self.b-self.A(self.x)
            else:
                r -= q * alpha
            # print ('r', r.val)
            gamma_ = gamma
            gamma = r.dot(r)
            # print ('gamma', gamma)
            beta = max(0, gamma/gamma_) ## positive definite
            # print ('d*beta', beta, (d*beta).val)
            d = r + d*beta
            # print ('d', d.val)
            delta = delta_*np.absolute(gamma)**0.5
            self.note.cflush(
        "\niteration : %08u   alpha = %3.1E   beta = %3.1E   delta = %3.1E"\
        %(ii,np.real(alpha),np.real(beta),np.real(delta)))
            if gamma == 0:
                convergence = clevel+1
                self.note.cprint("   convergence level : INF\n... done.")
                break
            elif np.absolute(delta)<tol:
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if(convergence==clevel):
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0, convergence-1)
            if ii==limii:
                self.note.cprint("\n... quit.")
                break

            if (self.spam is not None):
                self.spam(self.x, ii)

            ii += 1

        if (self.spam is not None):
            self.spam(self.x,ii)

        return self.x, convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _calc_with(self, tol=1E-4, clevel=1, limii=None): ## > runs cg with preconditioner

        clevel = int(clevel)
        if(limii is None):
            limii = 10*self.b.domain.dim
        else:
            limii = int(limii)
        r = self.b-self.A(self.x)

        d = self.W(r)
        gamma = r.dot(d)
        if gamma==0:
            return self.x, clevel+1
        delta_ = np.absolute(gamma)**(-0.5)

        convergence = 0
        ii = 1
        while(True):
            q = self.A(d)
            alpha = gamma/d.dot(q) ## positive definite
            if np.isfinite(alpha) == False:
                self.note.cprint(
                    "\niteration : %08u   alpha = NAN\n... dead."%ii)
                return self.x, 0
            self.x += d * alpha ## update
            if np.signbit(np.real(alpha)) == True:
                about.warnings.cprint(
                "WARNING: positive definiteness of A violated.")
                r = self.b-self.A(self.x)
            elif (ii%self.reset) == 0:
                r = self.b-self.A(self.x)
            else:
                r -= q * alpha
            s = self.W(r)
            gamma_ = gamma
            gamma = r.dot(s)
            if np.signbit(np.real(gamma)) == True:
                about.warnings.cprint(
                "WARNING: positive definiteness of W violated.")
            beta = max(0, gamma/gamma_) ## positive definite
            d = s + d*beta ## conjugated gradient

            delta = delta_*np.absolute(gamma)**0.5
            self.note.cflush(
        "\niteration : %08u   alpha = %3.1E   beta = %3.1E   delta = %3.1E"\
        %(ii,np.real(alpha),np.real(beta),np.real(delta)))
            if gamma==0:
                convergence = clevel+1
                self.note.cprint("   convergence level : INF\n... done.")
                break
            elif np.absolute(delta)<tol:
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if convergence==clevel:
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0, convergence-1)
            if ii==limii:
                self.note.cprint("\n... quit.")
                break

            if (self.spam is not None):
                self.spam(self.x,ii)

            ii += 1

        if (self.spam is not None):
            self.spam(self.x,ii)
        return self.x, convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.conjugate_gradient>"

##=============================================================================

