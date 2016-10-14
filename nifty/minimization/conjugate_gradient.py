# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np

from nifty import Field

import logging
logger = logging.getLogger('NIFTy.CG')


class ConjugateGradient(object):
    def __init__(self, convergence_tolerance=1E-4, convergence_level=1,
                 iteration_limit=-1, reset_count=-1,
                 preconditioner=None, callback=None):
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
                conjugated directions (default: sqrt(b.dim)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        self.convergence_tolerance = np.float(convergence_tolerance)
        self.convergence_level = np.float(convergence_level)
        self.iteration_limit = int(iteration_limit)
        self.reset_count = int(reset_count)

        if preconditioner is None:
            preconditioner = lambda z: z

        self.preconditioner = preconditioner
        self.callback = callback

    def __call__(self, A, b, x0=None):
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
        if x0 is None:
            x0 = Field(A.domain, val=0)

        r = b - A(x0)
        d = self.preconditioner(r)
        previous_gamma = r.dot(d)
        if previous_gamma == 0:
            self.info("The starting guess is already perfect solution for "
                      "the inverse problem.")
            return x0, self.convergence_level+1

        x = x0
        convergence = 0
        iteration_number = 1
        logger.info("Starting conjugate gradient.")

        while(True):
            if self.callback is not None:
                self.callback(x, iteration_number)

            q = A(d)
            alpha = previous_gamma/d.dot(q)

            if not np.isfinite(alpha):
                logger.error("Alpha became infinite! Stopping.")
                return x0, 0

            x += d * alpha

            reset = False
            if alpha.real < 0:
                logger.warn("Positive definiteness of A violated!")
                reset = True
            if reset or iteration_number % self.reset_count == 0:
                logger.info("Resetting conjugate directions.")
                r = b - A(x)
            else:
                r -= q * alpha

            s = self.preconditioner(r)
            gamma = r.dot(s)

            if gamma.real < 0:
                logger.warn("Positive definitness of preconditioner violated!")

            beta = max(0, gamma/previous_gamma)

            d = s + d * beta

            #delta = previous_delta * np.sqrt(abs(gamma))
            delta = abs(1-np.sqrt(abs(previous_gamma))/np.sqrt(abs(gamma)))

            logger.debug("Iteration : %08u   alpha = %3.1E   beta = %3.1E   "
                         "delta = %3.1E" %
                         (iteration_number,
                          np.real(alpha),
                          np.real(beta),
                          np.real(delta)))

            if gamma == 0:
                convergence = self.convergence_level+1
                self.info("Reached infinite convergence.")
                break
            elif abs(delta) < self.convergence_tolerance:
                convergence += 1
                self.info("Updated convergence level to: %u" % convergence)
                if convergence == self.convergence_level:
                    self.info("Reached target convergence level.")
                    break
            else:
                convergence = max(0, convergence-1)

            if iteration_number == self.iteration_limit:
                self.warn("Reached iteration limit. Stopping.")
                break

            iteration_number += 1
            previous_gamma = gamma

        return x, convergence
