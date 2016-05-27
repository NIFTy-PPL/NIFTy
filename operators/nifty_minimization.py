## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                     __   ____   __
    ..                   /__/ /   _/ /  /_
    ..         __ ___    __  /  /_  /   _/  __   __
    ..       /   _   | /  / /   _/ /  /   /  / /  /
    ..      /  / /  / /  / /  /   /  /_  /  /_/  /
    ..     /__/ /__/ /__/ /__/    \___/  \___   /  tools
    ..                                  /______/

    This module extends NIFTY with a nifty set of tools including further
    operators, namely the :py:class:`invertible_operator` and the
    :py:class:`propagator_operator`, and minimization schemes, namely
    :py:class:`steepest_descent` and :py:class:`conjugate_gradient`. Those
    tools are supposed to support the user in solving information field
    theoretical problems (almost) without numerical pain.

"""
from __future__ import division
#from nifty_core import *
import numpy as np
from nifty.config import notification, about
from nifty.nifty_field import field
from nifty.nifty_simple_math import vdot


#from nifty_core import space,                                                \
#                       field
#from operators import operator, \
#                      diagonal_operator



##=============================================================================

class conjugate_gradient(object):
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
            conjugated directions (default: sqrt(b.get_dim())).
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
            conjugated directions (default: sqrt(b.get_dim())).
        note : notification
            Notification instance.

    """
    def __init__(self, A, b, W=None, spam=None, reset=None, note=False):
        """
            Initializes the conjugate_gradient and sets the attributes (except
            for `x`).

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
                conjugated directions (default: sqrt(b.get_dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        if hasattr(A,"__call__") == True:
            self.A = A ## applies A
        else:
            raise AttributeError(about._errors.cstring(
                "ERROR: A must be callable!"))

        self.b = b

        if (W is None) or (hasattr(W,"__call__")==True):
            self.W = W ## applies W ~ A_inverse
        else:
            raise AttributeError(about._errors.cstring(
                "ERROR: W must be None or callable!"))

        self.spam = spam ## serves as callback given x and iteration number

        if reset is None: ## 2 < reset ~ sqrt(dim)
            self.reset = max(2,
                             int(np.sqrt(b.domain.get_dim())))
        else:
            self.reset = max(2,
                             int(reset))

        self.note = notification(default=bool(note))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self, x0=None, **kwargs): ## > runs cg with/without preconditioner
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
                Maximum number of iterations performed (default: 10 * b.get_dim()).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        self.x = self.b.copy_empty()
        self.x.set_val(new_val = 0)
        self.x.set_val(new_val = x0)

        if self.W is None:
            return self._calc_without(**kwargs)
        else:
            return self._calc_with(**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _calc_without(self, tol=1E-4, clevel=1, limii=None): ## > runs cg without preconditioner
        clevel = int(clevel)
        if limii is None:
            limii = 10*self.b.domain.get_dim()
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
            limii = 10*self.b.domain.get_dim()
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





##=============================================================================

class steepest_descent(object):
    """
        ..                          __
        ..                        /  /
        ..      _______      ____/  /
        ..    /  _____/    /   _   /
        ..   /_____  / __ /  /_/  / __
        ..  /_______//__/ \______|/__/  class

        NIFTY tool class for steepest descent minimization

        This tool minimizes a scalar energy-function by steepest descent using
        the functions gradient. Steps and step widths are choosen according to
        the Wolfe conditions [#]_. For details on usage and output, see the
        notes below.

        Parameters
        ----------
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function, *optional*
            Callback function which is given the current `x` and iteration
            counter each iteration (default: None).
        a : {4-tuple}, *optional*
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}, *optional*
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : bool, *optional*
            Indicates whether notes are printed or not (default: False).

        See Also
        --------
        scipy.optimize.fmin_cg, scipy.optimize.fmin_ncg,
        scipy.optimize.fmin_l_bfgs_b

        Notes
        -----
        After initialization by `__init__`, the minimizer is started by calling
        it using `__call__`, which takes additional parameters. Notifications,
        if enabled, will state the iteration number, current step width `alpha`,
        current maximal change `delta` that is compared to the tolerance, and
        the convergence level if changed. The minimizer will exit in three
        states: DEAD if no step width above 1E-13 is accepted, QUIT if the
        maximum number of iterations is reached, or DONE if convergence is
        achieved. Returned will be the latest `x` and the latest convergence
        level, which can evaluate ``True`` for all exit states.

        References
        ----------
        .. [#] J. Nocedal and S. J. Wright, Springer 2006, "Numerical
            Optimization", ISBN: 978-0-387-30303-1 (print) / 978-0-387-40065-5
            `(online) <http://link.springer.com/book/10.1007/978-0-387-40065-5/page/1>`_

        Examples
        --------
        >>> def egg(x):
        ...     E = 0.5*x.dot(x) # energy E(x) -- a two-dimensional parabola
        ...     g = x # gradient
        ...     return E,g
        >>> x = field(point_space(2), val=[1, 3])
        >>> x,convergence = steepest_descent(egg, note=True)(x0=x, tol=1E-4, clevel=3)
        iteration : 00000001   alpha = 1.0E+00   delta = 6.5E-01
        iteration : 00000002   alpha = 2.0E+00   delta = 1.4E-01
        iteration : 00000003   alpha = 1.6E-01   delta = 2.1E-03
        iteration : 00000004   alpha = 2.6E-03   delta = 3.0E-04
        iteration : 00000005   alpha = 2.0E-04   delta = 5.3E-05   convergence level : 1
        iteration : 00000006   alpha = 8.2E-05   delta = 4.4E-06   convergence level : 2
        iteration : 00000007   alpha = 6.6E-06   delta = 3.1E-06   convergence level : 3
        ... done.
        >>> bool(convergence)
        True
        >>> x.val # approximately zero
        array([ -6.87299426e-07  -2.06189828e-06])

        Attributes
        ----------
        x : field
            Current field.
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function
            Callback function which is given the current `x` and iteration
            counter each iteration; can be ``None``.
        a : {4-tuple}
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : notification
            Notification instance.

    """
    def __init__(self,eggs,spam=None,a=(0.2,0.5,1,2),c=(1E-4,0.9),note=False):
        """
            Initializes the steepest_descent and sets the attributes (except
            for `x`).

            Parameters
            ----------
            eggs : function
                Given the current `x` it returns the tuple of energy and gradient.
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            a : {4-tuple}, *optional*
                Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
                widths (default: (0.2,0.5,1,2)).
            c : {2-tuple}, *optional*
                Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
                (default: (1E-4,0.9)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        self.eggs = eggs ## returns energy and gradient

        self.spam = spam ## serves as callback given x and iteration number
        self.a = a ## 0 < a1 ~ a2 < 1 ~ a3 < a4
        self.c = c ## 0 < c1 < c2 < 1
        self.note = notification(default=bool(note))

        self._alpha = None ## last alpha

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self,x0,alpha=1,tol=1E-4,clevel=8,limii=100000):
        """
            Runs the steepest descent minimization.

            Parameters
            ----------
            x0 : field
                Starting guess for the minimization.
            alpha : scalar, *optional*
                Starting step width to be multiplied with normalized gradient
                (default: 1).
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by maximal change in
                `x` (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 8).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 100,000).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        if(not isinstance(x0,field)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.x = x0

        ## check for exsisting alpha
        if(alpha is None):
            if(self._alpha is not None):
                alpha = self._alpha
            else:
                alpha = 1

        clevel = max(1,int(clevel))
        limii = int(limii)

        E,g = self.eggs(self.x) ## energy and gradient
        norm = g.norm() ## gradient norm
        if(norm==0):
            self.note.cprint("\niteration : 00000000   alpha = 0.0E+00   delta = 0.0E+00\n... done.")
            return self.x,clevel+2

        convergence = 0
        ii = 1
        while(True):
            x_,E,g,alpha,a = self._get_alpha(E,g,norm,alpha) ## "news",alpha,a

            if(alpha is None):
                self.note.cprint("\niteration : %08u   alpha < 1.0E-13\n... dead."%ii)
                break
            else:
                delta = abs(g).max()*(alpha/norm)
                #delta = np.absolute(g.val).max()*(alpha/norm)
                self.note.cflush("\niteration : %08u   alpha = %3.1E   delta = %3.1E"%(ii,alpha,delta))
                ## update
                self.x = x_

                alpha *= a

            norm = g.norm() ## gradient norm
            if(delta==0):
                convergence = clevel+2
                self.note.cprint("   convergence level : %u\n... done."%convergence)
                break
            elif(delta<tol):
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if(convergence==clevel):
                    convergence += int(ii==clevel)
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0,convergence-1)
            if(ii==limii):
                self.note.cprint("\n... quit.")
                break

            if(self.spam is not None):
                self.spam(self.x,ii)

            ii += 1

        if(self.spam is not None):
            self.spam(self.x,ii)

        ## memorise last alpha
        if(alpha is not None):
            self._alpha = alpha/a ## undo update

        return self.x,convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_alpha(self,E,g,norm,alpha): ## > determines the new alpha
        while(True):
            ## Wolfe conditions
            wolfe,x_,E_,g_,a = self._check_wolfe(E,g,norm,alpha)
            #            wolfe,x_,E_,g_,a = self._check_strong_wolfe(E,g,norm,alpha)
            if(wolfe):
                return x_,E_,g_,alpha,a
            else:
                alpha *= a
                if(alpha<1E-13):
                    return None,None,None,None,None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _check_wolfe(self,E,g,norm,alpha): ## > checks the Wolfe conditions
        x_ = self._get_x(g,norm,alpha)
        pg = norm
        E_,g_ = self.eggs(x_)
        if(E_>E+self.c[0]*alpha*pg):
            if(E_<E):
                return True,x_,E_,g_,self.a[1]
            return False,None,None,None,self.a[0]
        pg_ = g.dot(g_)/norm
        if(pg_<self.c[1]*pg):
            return True,x_,E_,g_,self.a[3]
        return True,x_,E_,g_,self.a[2]

#    def _check_strong_wolfe(self,E,g,norm,alpha): ## > checks the strong Wolfe conditions
#        x_ = self._get_x(g,norm,alpha)
#        pg = norm
#        E_,g_ = self.eggs(x_)
#        if(E_>E+self.c[0]*alpha*pg):
#            if(E_<E):
#                return True,x_,E_,g_,self.a[1]
#            return False,None,None,None,self.a[0]
#        apg_ = np.absolute(g.dot(g_))/norm
#        if(apg_>self.c[1]*np.absolute(pg)):
#            return True,x_,E_,g_,self.a[3]
#        return True,x_,E_,g_,self.a[2]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_x(self,g,norm,alpha): ## > updates x
        return self.x-g*(alpha/norm)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.steepest_descent>"

##=============================================================================


##=============================================================================

class quasi_newton_minimizer(object):
    """
        ..                          __
        ..                        /  /
        ..      _______      ____/  /
        ..    /  _____/    /   _   /
        ..   /_____  / __ /  /_/  / __
        ..  /_______//__/ \______|/__/  class

        NIFTY tool class for steepest descent minimization

        This tool minimizes a scalar energy-function by steepest descent using
        the functions gradient. Steps and step widths are choosen according to
        the Wolfe conditions [#]_. For details on usage and output, see the
        notes below.

        Parameters
        ----------
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function, *optional*
            Callback function which is given the current `x` and iteration
            counter each iteration (default: None).
        a : {4-tuple}, *optional*
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}, *optional*
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : bool, *optional*
            Indicates whether notes are printed or not (default: False).

        See Also
        --------
        scipy.optimize.fmin_cg, scipy.optimize.fmin_ncg,
        scipy.optimize.fmin_l_bfgs_b

        Notes
        -----
        After initialization by `__init__`, the minimizer is started by calling
        it using `__call__`, which takes additional parameters. Notifications,
        if enabled, will state the iteration number, current step width `alpha`,
        current maximal change `delta` that is compared to the tolerance, and
        the convergence level if changed. The minimizer will exit in three
        states: DEAD if no step width above 1E-13 is accepted, QUIT if the
        maximum number of iterations is reached, or DONE if convergence is
        achieved. Returned will be the latest `x` and the latest convergence
        level, which can evaluate ``True`` for all exit states.

        References
        ----------
        .. [#] J. Nocedal and S. J. Wright, Springer 2006, "Numerical
            Optimization", ISBN: 978-0-387-30303-1 (print) / 978-0-387-40065-5
            `(online) <http://link.springer.com/book/10.1007/978-0-387-40065-5/page/1>`_

        Examples
        --------
        >>> def egg(x):
        ...     E = 0.5*x.dot(x) # energy E(x) -- a two-dimensional parabola
        ...     g = x # gradient
        ...     return E,g
        >>> x = field(point_space(2), val=[1, 3])
        >>> x,convergence = steepest_descent(egg, note=True)(x0=x, tol=1E-4, clevel=3)
        iteration : 00000001   alpha = 1.0E+00   delta = 6.5E-01
        iteration : 00000002   alpha = 2.0E+00   delta = 1.4E-01
        iteration : 00000003   alpha = 1.6E-01   delta = 2.1E-03
        iteration : 00000004   alpha = 2.6E-03   delta = 3.0E-04
        iteration : 00000005   alpha = 2.0E-04   delta = 5.3E-05   convergence level : 1
        iteration : 00000006   alpha = 8.2E-05   delta = 4.4E-06   convergence level : 2
        iteration : 00000007   alpha = 6.6E-06   delta = 3.1E-06   convergence level : 3
        ... done.
        >>> bool(convergence)
        True
        >>> x.val # approximately zero
        array([ -6.87299426e-07  -2.06189828e-06])

        Attributes
        ----------
        x : field
            Current field.
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function
            Callback function which is given the current `x` and iteration
            counter each iteration; can be ``None``.
        a : {4-tuple}
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : notification
            Notification instance.

    """
    def __init__(self, f, fprime, f_args=(), callback=None,
                 c1=1E-4, c2=0.9, note=False):
        """
            Initializes the steepest_descent and sets the attributes (except
            for `x`).

            Parameters
            ----------
            eggs : function
                Given the current `x` it returns the tuple of energy and gradient.
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            a : {4-tuple}, *optional*
                Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
                widths (default: (0.2,0.5,1,2)).
            c : {2-tuple}, *optional*
                Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
                (default: (1E-4,0.9)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        assert(callable(f))
        assert(callable(fprime))
        self.f = f
        self.fprime = fprime

        self.callback = callback

        self.line_searcher = line_search_strong_wolfe(f=self.f,
                                                      fprime=self.fprime,
                                                      f_args=f_args,
                                                      c1=c1, c2=c2)

        self.note = notification(default=bool(note))

        self.memory = {}

    def __call__(self, x0, tol=1E-4, clevel=8, limii=100000):
        """
            Runs the steepest descent minimization.

            Parameters
            ----------
            x0 : field
                Starting guess for the minimization.
            alpha : scalar, *optional*
                Starting step width to be multiplied with normalized gradient
                (default: 1).
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by maximal change in
                `x` (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 8).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 100,000).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        if not isinstance(x0, field):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.x = x0

        clevel = max(1, int(clevel))
        limii = int(limii)

        convergence = 0
        f_k_minus_1 = None
        f_k = self.f(self.x)
        step_length = 0
        for i in xrange(limii):
            if self.callback is not None:
                self.callback(self.x, i)

            # compute the the gradient for the current x
            gradient = self.fprime(self.x)
            gradient_norm = gradient.norm()

            # check if x is at a flat point
            if gradient_norm == 0:
                self.note.cprint(
                    "\niteration : 00000000   step_length = 0.0E+00   " +
                    "delta = 0.0E+00\n... done.")
                convergence = clevel+2
                break

            descend_direction = self._get_descend_direction(gradient,
                                                            gradient_norm)

            # compute the step length, which minimizes f_k along the
            # search direction = the gradient
            self.line_searcher.set_coordinates(xk=self.x,
                                               pk=descend_direction,
                                               f_k=f_k,
                                               fprime_k=gradient,
                                               f_k_minus_1=f_k_minus_1)
            f_k_minus_1 = f_k
            step_length, f_k = self.line_searcher.perform_line_search()

            if step_length < 1E-13:
                self.note.cprint(
                   "\niteration : %08u   step_length < 1.0E-13\n... dead." % i)
                break

            # update x
            self.x += descend_direction*step_length

            # check convergence
            delta = abs(gradient).max() * (step_length/gradient_norm)
            self.note.cflush(
                "\niteration : %08u   step_length = %3.1E   delta = %3.1E"
                % (i, step_length, delta))
            if delta == 0:
                convergence = clevel + 2
                self.note.cprint("   convergence level : %u\n... done." %
                                 convergence)
                break
            elif delta < tol:
                convergence += 1
                self.note.cflush("   convergence level : %u" % convergence)
                if convergence == clevel:
                    convergence += int(i == clevel)
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0, convergence-1)

        else:
            self.note.cprint("\n... quit.")

        return self.x, convergence

    def _get_descend_direction(self, gradient, gradient_norm):
        raise NotImplementedError

    def __repr__(self):
        return "<nifty_tools.steepest_descent>"


class steepest_descent_new(quasi_newton_minimizer):
    def _get_descend_direction(self, gradient, gradient_norm):
        return gradient/(-gradient_norm)

class lbfgs(quasi_newton_minimizer):
    def __init__(self, *args, **kwargs):
        super(lbfgs, self).__init__(*args, **kwargs)
        # self.memory...


    def _get_descend_direction(self, gradient, gradient_norm):
        pass



# -----------------------------------------------------------------------------
# Pure-Python Wolfe line and scalar searches
# Based on scipy implementation
# -----------------------------------------------------------------------------

class line_search_strong_wolfe(object):
    """
    Class for finding a step size that satisfies the strong Wolfe conditions.
    """

    def __init__(self, f, fprime, f_args=(), c1=1e-4, c2=0.9,
                 max_step_size=50):

        """
        Parameters
        ----------

        f : callable f(x, *args)
            Objective function.

        fprime : callable f'(x, *args)
            Objective functions gradient.

        f_args : tuple (optional)
            Additional arguments passed to objective function and its
            derivation.

        c1 : float (optional)
            Parameter for Armijo condition rule.

        c2 : float (optional)
            Parameter for curvature condition rule.

        max_step_size : float (optional)
            Maximum step size
        """

        assert(callable(f))
        assert(callable(fprime))
        self.c1 = np.float(c1)
        self.c2 = np.float(c2)
        self.max_step_size = max_step_size
        self.max_zoom_iterations = 100

        self.f = f
        self.fprime = fprime
        self.f_args = f_args

        self.xk = None
        self.pk = None

        self.f_k = None
        self.f_k_minus_1 = None
        self.fprime_k = None

        self._last_alpha_star = 1.

    def set_coordinates(self, xk, pk, f_k=None, fprime_k=None,
                        f_k_minus_1=None):
        """
        Set the coordinates for a new line search.

        Parameters
        ----------
        xk : ndarray, d2o, field
            Starting point.

        pk : ndarray, d2o, field
            Unit vector in search direction.

        f_k : float (optional)
            Function value f(x_k).

        fprime_k : ndarray, d2o, field (optional)
            Function value fprime(xk).

        f_k_minus_1 : None, float (optional)
            Function value f(x_k-1).

        """

        self.xk = xk
        self.pk = pk

        if f_k is None:
            self.f_k = self.f(xk)
        else:
            self.f_k = f_k

        if fprime_k is None:
            self.fprime_k = self.fprime(xk)
        else:
            self.fprime_k = fprime_k

        self.f_k_minus_1 = f_k_minus_1

    def _phi(self, alpha):
        if alpha == 0:
            value = self.f_k
        else:
            value = self.f(self.xk + self.pk*alpha, *self.f_args)
        return value

    def _phiprime(self, alpha):
        if alpha == 0:
            gradient = self.fprime_k
        else:
            gradient = self.fprime(self.xk + self.pk*alpha, *self.f_args)
        return vdot(gradient, self.pk)

    def perform_line_search(self, c1=None, c2=None, max_step_size=None,
                            max_iterations=10):
        if c1 is None:
            c1 = self.c1
        if c2 is None:
            c2 = self.c2
        if max_step_size is None:
            max_step_size = self.max_step_size

        # initialize the zero phis
        old_phi_0 = self.f_k_minus_1
        phi_0 = self._phi(0.)
        phiprime_0 = self._phiprime(0.)

        if phiprime_0 == 0:
            about.warnings.cprint(
                "WARNING: Flat gradient in search direction.")
            return 0., 0.

        # set alphas
        alpha0 = 0.
        if old_phi_0 is not None and phiprime_0 != 0:
            alpha1 = min(1.0, 1.01*2*(phi_0 - old_phi_0)/phiprime_0)
            if alpha1 < 0:
                alpha1 = 1.0
        else:
            alpha1 = 1.0

#        alpha1 = 1.
#        alpha1 = 1.01*self._last_alpha_star


        # give the alpha0 phis the right init value
        phi_alpha0 = phi_0
        phiprime_alpha0 = phiprime_0

        # start the minimization loop
        for i in xrange(max_iterations):
            phi_alpha1 = self._phi(alpha1)

            if alpha1 == 0:
                about.warnings.cprint("WARNING: Increment size became 0.")
                alpha_star = 0.
                phi_star = phi_0
                break

            if (phi_alpha1 > phi_0 + c1*alpha1*phiprime_0) or \
               ((phi_alpha1 >= phi_alpha0) and (i > 1)):
                (alpha_star, phi_star) = self._zoom(alpha0, alpha1,
                                                    phi_0, phiprime_0,
                                                    phi_alpha0, phiprime_alpha0,
                                                    phi_alpha1,
                                                    c1, c2)
                break

            phiprime_alpha1 = self._phiprime(alpha1)
            if abs(phiprime_alpha1) <= -c2*phiprime_0:
                alpha_star = alpha1
                phi_star = phi_alpha1
                break

            if phiprime_alpha1 >= 0:
                (alpha_star, phi_star) = self._zoom(alpha1, alpha0,
                                                    phi_0, phiprime_0,
                                                    phi_alpha1, phiprime_alpha1,
                                                    phi_alpha0,
                                                    c1, c2)
                break

            # update alphas
            alpha0, alpha1 = alpha1, min(2*alpha1, max_step_size)
            phi_alpha0 = phi_alpha1
            phiprime_alpha0 = phiprime_alpha1
            # phi_alpha1 = self._phi(alpha1)

        else:
            # max_iterations was reached
            alpha_star = alpha1
            phi_star = phi_alpha1
            about.warnings.cprint(
                "WARNING: The line search algorithm did not converge.")

        self._last_alpha_star = alpha_star
        return alpha_star, phi_star

    def _zoom(self, alpha_lo, alpha_hi, phi_0, phiprime_0,
              phi_lo, phiprime_lo, phi_hi, c1, c2):

        max_iterations = self.max_zoom_iterations
        # define the cubic and quadratic interpolant checks
        cubic_delta = 0.2  # cubic
        quad_delta = 0.1  # quadratic

        # initialize the most recent versions (j-1) of phi and alpha
        alpha_recent = 0
        phi_recent = phi_0

        for i in xrange(max_iterations):
            delta_alpha = alpha_hi - alpha_lo
            if delta_alpha < 0:
                a, b = alpha_hi, alpha_lo
            else:
                a, b = alpha_lo, alpha_hi

            # Try cubic interpolation
            if i > 0:
                cubic_check = cubic_delta * delta_alpha
                alpha_j = self._cubicmin(alpha_lo, phi_lo, phiprime_lo,
                                         alpha_hi, phi_hi,
                                         alpha_recent, phi_recent)
            # If cubic was not successful or not available, try quadratic
            if (i == 0) or (alpha_j is None) or (alpha_j > b - cubic_check) or\
               (alpha_j < a + cubic_check):
                quad_check = quad_delta * delta_alpha
                alpha_j = self._quadmin(alpha_lo, phi_lo, phiprime_lo,
                                        alpha_hi, phi_hi)
                # If quadratic was not successfull, try bisection
                if (alpha_j is None) or (alpha_j > b - quad_check) or \
                   (alpha_j < a + quad_check):
                    alpha_j = alpha_lo + 0.5*delta_alpha

            # Check if the current value of alpha_j is already sufficient
            phi_alphaj = self._phi(alpha_j)
            # If the first Wolfe condition is not met replace alpha_hi
            # by alpha_j
            if (phi_alphaj > phi_0 + c1*alpha_j*phiprime_0) or\
               (phi_alphaj >= phi_lo):
                alpha_recent, phi_recent = alpha_hi, phi_hi
                alpha_hi, phi_hi = alpha_j, phi_alphaj
            else:
                phiprime_alphaj = self._phiprime(alpha_j)
                # If the second Wolfe condition is met, return the result
                if abs(phiprime_alphaj) <= -c2*phiprime_0:
                    alpha_star = alpha_j
                    phi_star = phi_alphaj
                    break
                # If not, check the sign of the slope
                if phiprime_alphaj*delta_alpha >= 0:
                    alpha_recent, phi_recent = alpha_hi, phi_hi
                    alpha_hi, phi_hi = alpha_lo, phi_lo
                else:
                    alpha_recent, phi_recent = alpha_lo, phi_lo
                # Replace alpha_lo by alpha_j
                (alpha_lo, phi_lo, phiprime_lo) = (alpha_j, phi_alphaj,
                                                   phiprime_alphaj)

        else:
            alpha_star, phi_star = alpha_j, phi_alphaj
            about.warnings.cprint(
                "WARNING: The line search algorithm (zoom) did not converge.")

        return alpha_star, phi_star

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
        If no minimizer can be found return None
        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                C = fpa
                db = b - a
                dc = c - a
                denom = (db * dc) ** 2 * (db - dc)
                d1 = np.empty((2, 2))
                d1[0, 0] = dc ** 2
                d1[0, 1] = -db ** 2
                d1[1, 0] = -dc ** 3
                d1[1, 1] = db ** 3
                [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                                fc - fa - C * dc]).flatten())
                A /= denom
                B /= denom
                radical = B * B - 3 * A * C
                xmin = a + (-B + np.sqrt(radical)) / (3 * A)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa,
        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                D = fa
                C = fpa
                db = b - a * 1.0
                B = (fb - D - C * db) / (db * db)
                xmin = a - C / (2.0 * B)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin


class cyclic_storage(object):
    pass
























