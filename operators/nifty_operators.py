# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
#
# Copyright (C) 2015 Max-Planck-Society
#
# Author: Marco Selig
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np
from nifty.config import about
from nifty.field import Field
from nifty.space import Space

from nifty_minimization import conjugate_gradient
from nifty_probing import trace_prober,\
    inverse_trace_prober,\
    diagonal_prober,\
    inverse_diagonal_prober
import nifty.nifty_utilities as utilities
import nifty.nifty_simple_math as nifty_simple_math


# =============================================================================

class operator(object):
    """
        ..                                                      __
        ..                                                    /  /_
        ..    ______    ______    _______   _____   ____ __  /   _/  ______    _____
        ..  /   _   | /   _   | /   __  / /   __/ /   _   / /  /   /   _   | /   __/
        .. /  /_/  / /  /_/  / /  /____/ /  /    /  /_/  / /  /_  /  /_/  / /  /
        .. \______/ /   ____/  \______/ /__/     \______|  \___/  \______/ /__/     class
        ..         /__/

        NIFTY base class for (linear) operators

        The base NIFTY operator class is an abstract class from which other
        specific operator subclasses, including those preimplemented in NIFTY
        (e.g. the diagonal operator class) must be derived.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool, *optional*
            Indicates whether the operator is self-adjoint or not
            (default: False)
        uni : bool, *optional*
            Indicates whether the operator is unitary or not
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)
        para : {single object, list of objects}, *optional*
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
            (default: None)

        See Also
        --------
        diagonal_operator :  An operator class for handling purely diagonal
            operators.
        power_operator : Similar to diagonal_operator but with handy features
            for dealing with diagonal operators whose diagonal
            consists of a power spectrum.
        vecvec_operator : Operators constructed from the outer product of two
            fields
        response_operator : Implements a modeled instrument response which
            translates a signal into data space.
        projection_operator : An operator that projects out one or more
            components in a basis, e.g. a spectral band
            of Fourier components.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
        para : {single object, list of objects}
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
    """

    def __init__(self, domain, codomain=None, sym=False, uni=False,
                 imp=False, target=None, cotarget=None, bare=False):
        """
            Sets the attributes for an operator class instance.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sym : bool, *optional*
                Indicates whether the operator is self-adjoint or not
                (default: False)
            uni : bool, *optional*
                Indicates whether the operator is unitary or not
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)
            para : {object, list of objects}, *optional*
                This is a freeform list of parameters that derivatives of the
                operator class can use. Not used in the base operators.
                (default: None)

            Returns
            -------
            None
        """
        # Check if the domain is realy a space
        if not isinstance(domain, space):
            raise TypeError(about._errors.cstring(
                "ERROR: invalid input. domain is not a space."))
        self.domain = domain
        # Parse codomain
        if self.domain.check_codomain(codomain) == True:
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        # Cast the symmetric and unitary input
        self.sym = bool(sym)
        self.uni = bool(uni)
        self.bare = bool(bare)

        # If no target is supplied, we assume that the operator is square
        # If the operator is symmetric or unitary, we know that the operator
        # must be square

        if self.sym or self.uni:
            target = self.domain
            cotarget = self.codomain
            if target is not None:
                about.warnings.cprint("WARNING: Ignoring target.")

        elif target is None:
            target = self.domain
            cotarget = self.codomain

        elif isinstance(target, space):
            self.target = target
            # Parse cotarget
            if self.target.check_codomain(cotarget) == True:
                self.codomain = codomain
            else:
                self.codomain = self.domain.get_codomain()
        else:
            raise TypeError(about._errors.cstring(
                "ERROR: invalid input. Target is not a space."))

        if self.domain.discrete and self.target.discrete:
            self.imp = True
        else:
            self.imp = bool(imp)

    def set_val(self, new_val):
        """
            Resets the field values.

            Parameters
            ----------
            new_val : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        self.val = new_val
        return self.val

    def get_val(self):
        return self.val

    def _multiply(self, x, **kwargs):
        # > applies the operator to a given field
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'multiply'."))

    def _adjoint_multiply(self, x, **kwargs):
        # > applies the adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_multiply'."))

    def _inverse_multiply(self, x, **kwargs):
        # > applies the inverse operator to a given field
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_multiply'."))

    def _adjoint_inverse_multiply(self, x, **kwargs):
        # > applies the inverse adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_inverse_multiply'."))

    def _inverse_adjoint_multiply(self, x, **kwargs):
        # > applies the adjoint inverse operator to a given field
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_adjoint_multiply'."))

    def _briefing(self, x, domain, codomain, inverse):
        # make sure, that the result_field of the briefing lives in the
        # given domain and codomain
        result_field = Field(domain=domain, val=x, codomain=codomain,
                             copy=False)

        # weight if necessary
        if (not self.imp) and (not domain.discrete) and (not inverse):
            result_field = result_field.weight(power=1)
        return result_field

    def _debriefing(self, x, y, target, cotarget, inverse):
        # The debriefing takes care that the result field lives in the same
        # fourier-type domain as the input field
        assert(isinstance(y, Field))

        # weight if necessary
        if (not self.imp) and (not target.discrete) and inverse:
            y = y.weight(power=-1)

        # if the operators domain as well as the target have the harmonic
        # attribute, try to match the result_field to the input_field
        if hasattr(self.domain, 'harmonic') and \
                hasattr(self.target, 'harmonic'):
            if x.domain.harmonic != y.domain.harmonic:
                y = y.transform()

        return y

    def times(self, x, **kwargs):
        """
            Applies the operator to a given object

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            Ox : field
                Mapped field on the target domain of the operator.
        """
        # prepare
        y = self._briefing(x, self.domain, self.codomain, inverse=False)
        # apply operator
        y = self._multiply(y, **kwargs)
        # evaluate
        return self._debriefing(x, y, self.target, self.cotarget,
                                inverse=False)

    def __call__(self, x, **kwargs):
        return self.times(x, **kwargs)

    def adjoint_times(self, x, **kwargs):
        """
            Applies the adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAx : field
                Mapped field on the domain of the operator.

        """
        # check whether self-adjoint
        if self.sym:
            return self.times(x, **kwargs)
        # check whether unitary
        if self.uni:
            return self.inverse_times(x, **kwargs)

        # prepare
        y = self._briefing(x, self.target, self.cotarget, inverse=False)
        # apply operator
        y = self._adjoint_multiply(y, **kwargs)
        # evaluate
        return self._debriefing(x, y, self.domain, self.codomain,
                                inverse=False)

    def inverse_times(self, x, **kwargs):
        """
            Applies the inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain space of the operator.

            Returns
            -------
            OIx : field
                Mapped field on the target space of the operator.
        """
        # check whether self-inverse
        if self.sym and self.uni:
            return self.times(x, **kwargs)

        # prepare
        y = self._briefing(x, self.target, self.cotarget, inverse=True)
        # apply operator
        y = self._inverse_multiply(y, **kwargs)
        # evaluate
        return self._debriefing(x, y, self.domain, self.codomain,
                                inverse=True)

    def adjoint_inverse_times(self, x, **kwargs):
        """
            Applies the inverse adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAIx : field
                Mapped field on the domain of the operator.

        """
        # check whether self-adjoint
        if self.sym:
            return self.inverse_times(x, **kwargs)
        # check whether unitary
        if self.uni:
            return self.times(x, **kwargs)

        # prepare
        y = self._briefing(x, self.domain, self.codomain, inverse=True)
        # apply operator
        y = self._adjoint_inverse_multiply(y, **kwargs)
        # evaluate
        return self._debriefing(x, y, self.target, self.cotarget,
                                inverse=True)

    def inverse_adjoint_times(self, x, **kwargs):
        """
            Applies the adjoint inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OIAx : field
                Mapped field on the domain of the operator.

        """
        # check whether self-adjoint
        if self.sym:
            return self.inverse_times(x, **kwargs)
        # check whether unitary
        if self.uni:
            return self.times(x, **kwargs)

        # prepare
        y = self._briefing(x, self.domain, self.codomain, inverse=True)
        # apply operator
        y = self._inverse_adjoint_multiply(y, **kwargs)
        # evaluate
        return self._debriefing(x, y, self.target, self.cotarget,
                                inverse=True)

    def tr(self, domain=None, codomain=None, random="pm1", nrun=8,
           varQ=False, **kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """

        return trace_prober(operator=self,
                            domain=domain,
                            codomain=codomain,
                            random=random,
                            nrun=nrun,
                            varQ=varQ,
                            **kwargs
                            )()

    def inverse_tr(self, domain=None, codomain=None, random="pm1", nrun=8,
                   varQ=False, **kwargs):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            nrun : int, *optional*
                total number of probes (default: 8)
            varQ : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).


            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """
        return inverse_trace_prober(operator=self,
                                    domain=domain,
                                    codomain=codomain,
                                    random=random,
                                    nrun=nrun,
                                    varQ=varQ,
                                    **kwargs
                                    )()

    def diag(self, domain=None, codomain=None, random="pm1", nrun=8,
             varQ=False, bare=False, **kwargs):
        """
            Computes the diagonal of the operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """

        diag = diagonal_prober(operator=self,
                               domain=domain,
                               codomain=codomain,
                               random=random,
                               nrun=nrun,
                               varQ=varQ,
                               **kwargs
                               )()
        if diag is None:
            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None

        if domain is None:
            domain = diag.domain
        # weight if ...
        if (not domain.discrete) and bare:
            if(isinstance(diag, tuple)):  # diag == (diag,variance)
                return (diag[0].weight(power=-1),
                        diag[1].weight(power=-1))
            else:
                return diag.weight(power=-1)
        else:
            return diag

    def inverse_diag(self, domain=None, codomain=None, random="pm1",
                     nrun=8, varQ=False, bare=False, **kwargs):
        """
            Computes the diagonal of the inverse operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        diag = inverse_diagonal_prober(operator=self,
                                       domain=domain,
                                       codomain=codomain,
                                       random=random,
                                       nrun=nrun,
                                       varQ=varQ,
                                       **kwargs
                                       )()
        if(diag is None):
            about.infos.cprint("INFO: forwarding 'NoneType'.")
            return None

        if domain is None:
            domain = diag.codomain
        # weight if ...
        if not domain.discrete and bare:
            if(isinstance(diag, tuple)):  # diag == (diag,variance)
                return (diag[0].weight(power=-1),
                        diag[1].weight(power=-1))
            else:
                return diag.weight(power=-1)
        else:
            return diag

    def det(self):
        """
            Computes the determinant of the operator.

            Returns
            -------
            det : float
                The determinant

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'det'."))

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator.

            Returns
            -------
            det : float
                The determinant

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_det'."))

    def log_det(self):
        """
            Computes the logarithm of the determinant of the operator
            (if applicable).

            Returns
            -------
            logdet : float
                The logarithm of the determinant

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'log_det'."))

    def tr_log(self):
        """
            Computes the trace of the logarithm of the operator
            (if applicable).

            Returns
            -------
            logdet : float
                The trace of the logarithm

        """
        return self.log_det()

    def hat(self, bare=False, domain=None, codomain=None, **kwargs):
        """
            Translates the operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if domain is None:
            domain = self.domain
        if codomain is None:
            codomain = self.codomain

        diag = self.diag(bare=bare, domain=domain, codomain=codomain,
                         var=False, **kwargs)
        if diag is None:
            about.infos.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diag

    def inverse_hat(self, bare=False, domain=None, codomain=None, **kwargs):
        """
            Translates the inverse operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if domain is None:
            domain = self.target
        if codomain is None:
            codomain = self.cotarget
        diag = self.inverse_diag(bare=bare, domain=domain, codomain=codomain,
                                 var=False, **kwargs)
        if diag is None:
            about.infos.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diag

    def hathat(self, domain=None, codomain=None, **kwargs):
        """
            Translates the operator's diagonal into a diagonal operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The matrix diagonal as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if domain is None:
            domain = self.domain
        if codomain is None:
            codomain = self.codomain

        diag = self.diag(bare=False, domain=domain, codomain=codomain,
                         var=False, **kwargs)
        if diag is None:
            about.infos.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diagonal_operator(domain=domain, codomain=codomain,
                                 diag=diag, bare=False)

    def inverse_hathat(self, domain=None, codomain=None, **kwargs):
        """
            Translates the inverse operator's diagonal into a diagonal
            operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The diagonal of the inverse matrix as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if domain is None:
            domain = self.target
        if codomain is None:
            codomain = self.cotarget

        diag = self.inverse_diag(bare=False, domain=domain, codomain=codomain,
                                 var=False, **kwargs)
        if diag is None:
            about.infos.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diagonal_operator(domain=domain, codomain=codomain,
                                 diag=diag, bare=False)

    def __repr__(self):
        return "<nifty_core.operator>"

# =============================================================================


class diagonal_operator(operator):
    """
        ..           __   __                                                     __
        ..         /  / /__/                                                   /  /
        ..    ____/  /  __   ____ __   ____ __   ______    __ ___    ____ __  /  /
        ..  /   _   / /  / /   _   / /   _   / /   _   | /   _   | /   _   / /  /
        .. /  /_/  / /  / /  /_/  / /  /_/  / /  /_/  / /  / /  / /  /_/  / /  /_
        .. \______| /__/  \______|  \___   /  \______/ /__/ /__/  \______|  \___/  operator class
        ..                         /______/

        NIFTY subclass for diagonal operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        diag : {scalar, ndarray, field}
            The diagonal entries of the operator. For a scalar, a constant
            diagonal is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the diagonal entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: False)

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        The inverse applications of the diagonal operator feature a ``pseudo``
        flag indicating if zero divison shall be ignored and return zero
        instead of causing an error.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
    """

    def __init__(self, domain=None, codomain=None, diag=1, bare=False):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            diag : {scalar, ndarray, field}, *optional*
                The diagonal entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        # Set the domain
        if domain is None:
            try:
                self.domain = diag.domain
            except(AttributeError):
                raise TypeError(about._errors.cstring(
                    "ERROR: Explicit or implicit, i.e. via diag domain " +
                    "inupt needed!"))

        else:
            self.domain = domain

        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        self.target = self.domain
        self.cotarget = self.codomain
        self.imp = True
        self.set_diag(new_diag=diag, bare=bare)

    def set_diag(self, new_diag, bare=False):
        """
            Sets the diagonal of the diagonal operator

            Parameters
            ----------
            new_diag : {scalar, ndarray, field}
                The new diagonal entries of the operator. For a scalar, a
                constant diagonal is defined having the value provided. If
                no domain is given, diag must be a field.

            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None
        """

        # Set the diag-val
        self.val = self.domain.cast(new_diag)

        # Set the bare-val #TODO Check with Theo
        self.bare = bare

        # Weight if necessary
        if not self.domain.discrete and bare:
            self.val = self.domain.calc_weight(self.val, power=1)

        # Check complexity attributes
        if self.domain.calc_real_Q(self.val) == True:
            self.sym = True
        else:
            self.sym = False

        # Check if unitary, i.e. identity
        if (self.val == 1).all() == True:
            self.uni = True
        else:
            self.uni = False

    def _multiply(self, x, **kwargs):
        # applies the operator to a given field
        y = x.copy(domain=self.target, codomain=self.cotarget)
        y *= self.get_val()
        return y

    def _adjoint_multiply(self, x, **kwargs):
        # applies the adjoint operator to a given field
        y = x.copy(domain=self.domain, codomain=self.codomain)
        y *= self.get_val().conjugate()
        return y

    def _inverse_multiply(self, x, pseudo=False, **kwargs):
        # applies the inverse operator to a given field
        y = x.copy(domain=self.domain, codomain=self.codomain)
        if (self.get_val() == 0).any():
            if not pseudo:
                raise AttributeError(about._errors.cstring(
                    "ERROR: singular operator."))
            else:
#                raise NotImplementedError(
#                    "ERROR: function not yet implemented!")
                y /= self.get_val()
                # TODO: implement this
                # the following code does not work. np.isnan is needed,
                # but on a level of fields
#                y[y == np.nan] = 0
#                y[y == np.inf] = 0
        else:
            y /= self.get_val()
        return y

    def _adjoint_inverse_multiply(self, x, pseudo=False, **kwargs):
        # > applies the inverse adjoint operator to a given field
        y = x.copy(domain=self.target, codomain=self.cotarget)
        if (self.get_val() == 0).any():
            if not pseudo:
                raise AttributeError(about._errors.cstring(
                    "ERROR: singular operator."))
            else:
                raise NotImplementedError(
                    "ERROR: function not yet implemented!")
                # TODO: implement this
                # the following code does not work. np.isnan is needed,
                # but on a level of fields
                y /= self.get_val().conjugate()
                y[y == np.nan] = 0
                y[y == np.inf] = 0
        else:
            y /= self.get_val().conjugate()
        return y

    def _inverse_adjoint_multiply(self, x, pseudo=False, **kwargs):
        # > applies the adjoint inverse operator to a given field
        return self._adjoint_inverse_multiply(x, pseudo=pseudo, **kwargs)

    def tr(self, varQ=False, **kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """

        tr = self.domain.unary_operation(self.val, 'sum')

        if varQ:
            return (tr, 1)
        else:
            return tr

    def inverse_tr(self, varQ=False, **kwargs):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if (self.get_val() == 0).any():
            raise AttributeError(about._errors.cstring(
                "ERROR: singular operator."))
        inverse_tr = self.domain.unary_operation(
            self.domain.binary_operation(self.val, 1, 'rdiv', cast=0),
            'sum')

        if varQ:
            return (inverse_tr, 1)
        else:
            return inverse_tr

    def diag(self, bare=False, domain=None, codomain=None,
             varQ=False, **kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """

        if (domain is None) or (domain == self.domain):
            if not self.domain.discrete and bare:
                diag_val = self.domain.calc_weight(self.val, power=-1)
            else:
                diag_val = self.val
            diag = Field(self.domain, codomain=self.codomain, val=diag_val)
        else:
            diag = super(diagonal_operator, self).diag(bare=bare,
                                                       domain=domain,
                                                       codomain=codomain,
                                                       nrun=1,
                                                       random='pm1',
                                                       varQ=False,
                                                       **kwargs)
        if varQ:
            return (diag, diag.domain.cast(1))
        else:
            return diag

    def inverse_diag(self, bare=False, domain=None, codomain=None,
                     varQ=False, **kwargs):
        """
            Computes the diagonal of the inverse operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """

        if (domain is None) or (domain == self.domain):
            inverse_val = 1. / self.val
            if not self.domain.discrete and bare:
                inverse_diag_val = self.domain.calc_weight(inverse_val,
                                                           power=-1)
            else:
                inverse_diag_val = inverse_val
            inverse_diag = Field(self.domain, codomain=self.codomain,
                                 val=inverse_diag_val)

        else:
            inverse_diag = super(diagonal_operator,
                                 self).inverse_diag(bare=bare,
                                                    domain=domain,
                                                    nrun=1,
                                                    random='pm1',
                                                    varQ=False,
                                                    **kwargs)
        if varQ:
            return (inverse_diag, inverse_diag.domain.cast(1))
        else:
            return inverse_diag

    def det(self):
        """
            Computes the determinant of the matrix.

            Returns
            -------
            det : float
                The determinant

        """
        if self.uni:  # identity
            return 1.
        else:
            return self.domain.unary_operation(self.val, op='prod')

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator.

            Returns
            -------
            det : float
                The determinant

        """
        if self.uni:  # identity
            return 1.

        det = self.det()

        if det != 0:
            return 1. / det
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: singular operator."))

    def log_det(self):
        """
            Computes the logarithm of the determinant of the operator.

            Returns
            -------
            logdet : float
                The logarithm of the determinant

        """
        if self.uni:  # identity
            return 0
        else:
            return self.domain.unary_operation(
                nifty_simple_math.log(self.val), op='sum')

    def get_random_field(self, domain=None, codomain=None):
        """
            Generates a Gaussian random field with variance equal to the
            diagonal.

            Parameters
            ----------
            domain : space, *optional*
                space wherein the field lives (default: None, indicates
                to use self.domain)
            target : space, *optional*
                space wherein the transform of the field lives
                (default: None, indicates to use target of domain)

            Returns
            -------
            x : field
                Random field.

        """
        temp_field = Field(domain=self.domain,
                           codomain=self.codomain,
                           random='gau',
                           std=nifty_simple_math.sqrt(
                                   self.diag(bare=True).get_val()))
        if domain is None:
            domain = self.domain
        if domain.check_codomain(codomain):
            codomain = codomain
        elif domain.check_codomain(self.codomain):
            codomain = self.codomain
        else:
            codomain = domain.get_codomain()

        return Field(domain=domain, val=temp_field, codomain=codomain)

#        if domain.harmonic != self.domain.harmonic:
#            temp_field = temp_field.transform(new_domain=domain)
#
#        if self.domain == domain and self.codomain == codomain:
#            return temp_field
#        else:
#            return temp_field.copy(domain=domain,
#                                   codomain=codomain)

    def __repr__(self):
        return "<nifty_core.diagonal_operator>"


class identity_operator(diagonal_operator):
    def __init__(self, domain, codomain=None, bare=False):
        super(identity_operator, self).__init__(domain=domain,
                                                codomain=codomain,
                                                diag=1,
                                                bare=bare)


def identity(domain, codomain=None):
    """
        Returns an identity operator.

        The identity operator is represented by a `diagonal_operator` instance,
        which is applicable to a field-like object; i.e., a scalar, list,
        array or field. (The identity operator is unrelated to PYTHON's
        built-in function :py:func:`id`.)

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        Returns
        -------
        id : diagonal_operator
            The identity operator as a `diagonal_operator` instance.

        See Also
        --------
        diagonal_operator

        Examples
        --------
        >>> I = identity(rg_space(8,dist=0.2))
        >>> I.diag()
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        >>> I.diag(bare=True)
        array([ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.])
        >>> I.tr()
        8.0
        >>> I(3)
        <nifty.field>
        >>> I(3).val
        array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.])
        >>> I(np.arange(8))[:]
        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> f = I.get_random_field()
        >>> print(I(f) - f)
        nifty.field instance
        - domain      = <nifty.rg_space>
        - val         = [...]
          - min.,max. = [0.0, 0.0]
          - med.,mean = [0.0, 0.0]
        - target      = <nifty.rg_space>
        >>> I.times(f) ## equal to I(f)
        <nifty.field>
        >>> I.inverse_times(f)
        <nifty.field>

    """
    about.warnings.cprint('WARNING: The identity function is deprecated. ' +
                          'Use the identity_operator class.')
    return diagonal_operator(domain=domain,
                             codomain=codomain,
                             diag=1,
                             bare=False)


class power_operator(diagonal_operator):
    """
        ..      ______    ______   __     __   _______   _____
        ..    /   _   | /   _   | |  |/\/  / /   __  / /   __/
        ..   /  /_/  / /  /_/  /  |       / /  /____/ /  /
        ..  /   ____/  \______/   |__/\__/  \______/ /__/     operator class
        .. /__/

        NIFTY subclass for (signal-covariance-type) diagonal operators
        containing a power spectrum

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        spec : {scalar, list, array, field, function}
            The power spectrum. For a scalar, a constant power
            spectrum is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: True)
        pindex : ndarray, *optional*
            indexing array, obtainable from domain.get_power_indices
            (default: None)

        Other Parameters
        ----------------
        log : bool, *optional*
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer, *optional*
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}, *optional*
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).            vmin : {scalar, list, ndarray, field},
            *optional*
            Lower limit of the uniform distribution if ``random == "uni"``
            (default: 0).

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """

    def __init__(self, domain, codomain=None, spec=1, bare=True, **kwargs):
        """
            Sets the diagonal operator's standard properties

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            spec : {scalar, list, array, field, function}
                The power spectrum. For a scalar, a constant power
                spectrum is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
                vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        # Set the domain
        if not isinstance(domain, Space):
            raise TypeError(about._errors.cstring(
                "ERROR: The given domain is not a nifty space."))
        self.domain = domain

        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        # Set the target
        self.target = self.domain
        # Set the cotarget
        self.cotarget = self.codomain

        # Set imp
        self.imp = True
        # Save the kwargs
        self.kwargs = kwargs
        # Set the diag
        self.set_power(new_spec=spec, bare=bare, **kwargs)

        self.sym = True

        # check whether identity
        if(np.all(spec == 1)):
            self.uni = True
        else:
            self.uni = False

    # The domain is used for calculations of the power-spectrum, not for
    # actual field values. Therefore the casting of self.val must be switched
    # off.
    def set_val(self, new_val):
        """
            Resets the field values.

            Parameters
            ----------
            new_val : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        self.val = new_val
        return self.val

    def get_val(self):
        return self.val

    def set_power(self, new_spec, bare=True, pindex=None, **kwargs):
        """
            Sets the power spectrum of the diagonal operator

            Parameters
            ----------
            newspec : {scalar, list, array, field, function}
                The entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).

        """

        # Cast the pontentially given pindex. If no pindex was given,
        # extract it from self.domain using the supplied kwargs.
        pindex = self._cast_pindex(pindex, **kwargs)

        # Cast the new powerspectrum function
        temp_spec = self.domain.enforce_power(new_spec)

        # Calculate the diagonal
        try:
            diag = pindex.apply_scalar_function(lambda x: temp_spec[x],
                                                dtype=temp_spec.dtype.type)
            diag.hermitian = True
        except(AttributeError):
            diag = temp_spec[pindex]

        # Weight if necessary
        if not self.domain.discrete and bare:
            self.val = self.domain.calc_weight(diag, power=1)
        else:
            self.val = diag

        # check whether identity
        if (self.val == 1).all() == True:
            self.uni = True
        else:
            self.uni = False

        return self.val

    def _cast_pindex(self, pindex=None, **kwargs):
        # Update the internal kwargs dict with the given one:
        temp_kwargs = self.kwargs
        temp_kwargs.update(kwargs)

        # Case 1:  no pindex given
        if pindex is None:
            pindex = self.domain.power_indices.get_index_dict(
                                                       **temp_kwargs)['pindex']
        # Case 2: explicit pindex given
        else:
            # TODO: Pindex casting could be done here. No must-have.
            assert(np.all(np.array(pindex.shape) ==
                          np.array(self.domain.shape)))
        return pindex

    def get_power(self, bare=True, **kwargs):
        """
            Computes the power spectrum

            Parameters
            ----------
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True)
            pundex : ndarray, *optional*
                unindexing array, obtainable from domain.get_power_indices
                (default: None)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            spec : ndarray
                The power spectrum

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
                vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """

        temp_kwargs = self.kwargs
        temp_kwargs.update(kwargs)

        # Weight the diagonal values if necessary
        if not self.domain.discrete and bare:
            diag = self.domain.calc_weight(self.val, power=-1)
        else:
            diag = self.val

        # Use the calc_power routine of the domain in order to to stay
        # independent of the implementation
        diag = diag**(0.5)

        power = self.domain.calc_power(diag, **temp_kwargs)

        return power

    def get_projection_operator(self, pindex=None, **kwargs):
        """
            Generates a spectral projection operator

            Parameters
            ----------
            pindex : ndarray
                indexing array obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            P : projection_operator

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
                vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """

        pindex = self._cast_pindex(pindex, **kwargs)
        return projection_operator(self.domain,
                                   codomain=self.codomain,
                                   assign=pindex)

    def __repr__(self):
        return "<nifty_core.power_operator>"


class projection_operator(operator):
    """
        ..                                     __                       __     __
        ..                                   /__/                     /  /_  /__/
        ..      ______    _____   ______     __   _______   _______  /   _/  __   ______    __ ___
        ..    /   _   | /   __/ /   _   |  /  / /   __  / /   ____/ /  /   /  / /   _   | /   _   |
        ..   /  /_/  / /  /    /  /_/  /  /  / /  /____/ /  /____  /  /_  /  / /  /_/  / /  / /  /
        ..  /   ____/ /__/     \______/  /  /  \______/  \______/  \___/ /__/  \______/ /__/ /__/  operator class
        .. /__/                        /___/

        NIFTY subclass for projection operators

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        assign : ndarray, *optional*
            Assignments of domain items to projection bands. An array
            of integers, negative integers are associated with the
            nullspace of the projection. (default: None)

        Other Parameters
        ----------------
        log : bool, *optional*
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer, *optional*
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}, *optional*
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
            Lower limit of the uniform distribution if ``random == "uni"``
            (default: 0).

        Notes
        -----
        The application of the projection operator features a ``band`` keyword
        specifying a single projection band (see examples), a ``bandsup``
        keyword specifying which projection bands to sum up, and a ``split``
        keyword.

        Examples
        --------
        >>> space = point_space(3)
        >>> P = projection_operator(space, assign=[0, 1, 0])
        >>> P.bands()
        2
        >>> P([1, 2, 3], band=0) # equal to P.times(field(space,val=[1, 2, 3]))
        <nifty.field>
        >>> P([1, 2, 3], band=0).domain
        <nifty.point_space>
        >>> P([1, 2, 3], band=0).val # projection on band 0 (items 0 and 2)
        array([ 1.,  0.,  3.])
        >>> P([1, 2, 3], band=1).val # projection on band 1 (item 1)
        array([ 0.,  2.,  0.])
        >>> P([1, 2, 3])
        <nifty.field>
        >>> P([1, 2, 3]).domain
        <nifty.nested_space>
        >>> P([1, 2, 3]).val # projection on all bands
        array([[ 1.,  0.,  3.],
               [ 0.,  2.,  0.]])

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        ind : ndarray
            Assignments of domain items to projection bands.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """

    def __init__(self, domain, assign=None, codomain=None, **kwargs):
        """
            Sets the standard operator properties and `indexing`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            assign : ndarray, *optional*
                Assignments of domain items to projection bands. An array
                of integers, negative integers are associated with the
                nullspace of the projection. (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
                vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        # Check the domain
        if not isinstance(domain, Space):
            raise TypeError(about._errors.cstring(
                "ERROR: The supplied domain is not a nifty space instance."))
        self.domain = domain
        # Parse codomain
        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        self.target = self.domain
        self.cotarget = self.codomain

        # Cast the assignment
        if assign is None:
            try:
                self.domain.power_indices['pindex']
            except AttributeError:
                assign = np.arange(self.domain.dim, dtype=np.int)

        self.assign = self.domain.cast(assign, dtype=np.dtype('int'),
                                       hermitianize=False)

        # build indexing
        self.ind = self.domain.unary_operation(self.assign, op='unique')

        self.sym = True
        self.uni = False
        self.imp = True

    def bands(self):
        about.warnings.cprint(
            "WARNING: projection_operator.bands() is deprecated. " +
            "Use get_band_num() instead.")
        return self.get_band_num()

    def get_band_num(self):
        """
            Computes the number of projection bands

            Returns
            -------
            bands : int
                The number of projection bands
        """
        return len(self.ind)

    def rho(self):
        """
            Computes the number of degrees of freedom per projection band.

            Returns
            -------
            rho : ndarray
                The number of degrees of freedom per projection band.
        """
        # Check if the space has some meta-degrees of freedom
        if self.domain.dim == self.domain.dof:
            # If not, compute the degeneracy factor directly
            rho = self.domain.calc_bincount(self.assign)
        else:
            meta_mask = self.domain.calc_weight(
                self.domain.meta_volume_split,
                power=-1)
            rho = self.domain.calc_bincount(self.assign,
                                            weights=meta_mask)
        return rho

    def _multiply(self, x, bands=None, bandsum=None, **kwargs):
        """
            Applies the operator to a given field.

            Parameters
            ----------
            x : field
                Valid input field.
            band : int, *optional*
                Projection band whereon to project (default: None).
            bandsup: {integer, list/array of integers}, *optional*
                List of projection bands whereon to project and which to sum
                up. The `band` keyword is prefered over `bandsup`
                (default: None).

            Returns
            -------
            Px : field
                projected field(!)
        """

        if bands is not None:
            # cast the band
            if np.isscalar(bands):
                bands_was_scalar = True
            else:
                bands_was_scalar = False
            bands = np.array(bands, dtype=np.int).flatten()

            # check for consistency
            if np.any(bands > self.get_band_num() - 1) or np.any(bands < 0):
                raise TypeError(about._errors.cstring("ERROR: Invalid bands."))

            if bands_was_scalar:
                new_field = x * (self.assign == bands[0])
            else:
                # build up the projection results
                # prepare the projector-carrier
                carrier = np.empty((len(bands),), dtype=np.object_)
                for i in xrange(len(bands)):
                    current_band = bands[i]
                    projector = (self.assign == current_band)
                    carrier[i] = projector
                # Use the carrier and tensor dot to do the projection
                new_field = x.tensor_product(carrier)
            return new_field

        elif bandsum is not None:
            if np.isscalar(bandsum):
                bandsum = np.arange(int(bandsum) + 1)
            else:
                bandsum = np.array(bandsum, dtype=np.int_).flatten()

            # check for consistency
            if np.any(bandsum > self.get_band_num() - 1) or \
               np.any(bandsum < 0):
                raise TypeError(about._errors.cstring(
                    "ERROR: Invalid bandsum."))
            new_field = x.copy_empty()
            # Initialize the projector array, completely
            projector_sum = (self.assign != self.assign)
            for i in xrange(len(bandsum)):
                current_band = bandsum[i]
                projector = self.domain.binary_operation(self.assign,
                                                         current_band,
                                                         'eq')
                projector_sum += projector
            new_field = x * projector_sum
            return new_field

        else:
            return self._multiply(x, bands=self.ind)

    def _inverse_multiply(self, x, **kwargs):
        raise AttributeError(about._errors.cstring(
            "ERROR: singular operator."))

    def pseudo_tr(self, x, axis=(), **kwargs):
        """
            Computes the pseudo trace of a given object for all projection
            bands

            Parameters
            ----------
            x : {field, operator}
                The object whose pseudo-trace is to be computed. If the input
                is
                a field, the pseudo trace equals the trace of
                the projection operator mutliplied by a vector-vector operator
                corresponding to the input field. This is also equal to the
                pseudo inner product of the field with projected field itself.
                If the input is a operator, the pseudo trace equals the trace
                of
                the projection operator multiplied by the input operator.
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Pseudo trace for all projection bands
        """
        # Parse the input
        # Case 1: x is a field
        # -> Compute the diagonal of the corresponding vecvec-operator:
        # x * x^dagger
        if isinstance(x, Field):
            # check if field is in the same signal/harmonic space as the
            # domain of the projection operator
            if self.domain != x.domain:
                # TODO: check if a exception should be raised, or if
                # one should try to fix stuff.
                pass
                # x = x.transform(new_domain=self.domain)
            vecvec = vecvec_operator(val=x)
            return self.pseudo_tr(x=vecvec, axis=axis, **kwargs)

        # Case 2: x is an operator
        # -> take the diagonal
        elif isinstance(x, operator):
            working_field = x.diag(bare=False,
                                   domain=self.domain,
                                   codomain=self.codomain)
            if self.domain != working_field.domain:
                # TODO: check if a exception should be raised, or if
                # one should try to fix stuff.
                pass
                # working_field = working_field.transform(new_domain=self.domain)

        # Case 3: x is something else
        else:
            raise TypeError(about._errors.cstring(
                "ERROR: x must be a field or an operator."))

        # Check for hidden degrees of freedom and compensate the trace
        # accordingly
        if self.domain.dim != self.domain.dof:
            working_field *= self.domain.calc_weight(
                self.domain.meta_volume_split,
                power=-1)
        # prepare the result object
        projection_result = utilities.field_map(
            working_field.ishape,
            lambda z: self.domain.calc_bincount(self.assign, weights=z),
            working_field.get_val())

        projection_result = np.sum(projection_result, axis=axis)
        return projection_result

    def __repr__(self):
        return "<nifty_core.projection_operator>"


class vecvec_operator(operator):
    """
        ..                                                                 __
        ..                                                             __/  /__
        ..  __   __   _______   _______  __   __   _______   _______ /__    __/
        .. |  |/  / /   __  / /   ____/ |  |/  / /   __  / /   ____/   /__/
        .. |     / /  /____/ /  /____   |     / /  /____/ /  /____
        .. |____/  \______/  \______/   |____/  \______/  \______/
        .. operator class

        NIFTY subclass for vector-vector operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If none is given, the
            space of the field given in val is used. (default: None)
        val : {scalar, ndarray, field}, *optional*
            The field from which to construct the operator. For a scalar,
            a constant
            field is defined having the value provided. If no domain
            is given, val must be a field. (default: 1)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            The field from which the operator is derived.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives.
    """

    def __init__(self, domain=None, codomain=None, val=1):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If none is given, the
                space of the field given in val is used. (default: None)
            val : {scalar, ndarray, field}, *optional*
                The field from which to construct the operator. For a scalar,
                a constant
                field is defined having the value provided. If no domain
                is given, val must be a field. (default: 1)

            Returns
            -------
            None
        """
        if isinstance(val, Field):
            if domain is None:
                domain = val.domain
            if codomain is None:
                codomain = val.codomain

        if not isinstance(domain, Space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        self.target = self.domain
        self.cotarget = self.codomain
        self.val = Field(domain=self.domain,
                         codomain=self.codomain,
                         val=val)

        self.sym = True
        self.uni = False
        self.imp = True

    def set_val(self, new_val, copy=False):
        """
            Sets the field values of the operator

            Parameters
            ----------
            newval : {scalar, ndarray, field}
                The new field values. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)

            Returns
            -------
            None
        """
        self.val = self.val.set_val(new_val=new_val, copy=copy)

    def _multiply(self, x, **kwargs):
        y = x.copy_empty(domain=self.target, codomain=self.cotarget)
        y.set_val(new_val=self.val, copy=True)
        y *= self.val.dot(x, axis=())
        return y

    def _inverse_multiply(self, x, **kwargs):
        raise AttributeError(about._errors.cstring(
            "ERROR: singular operator."))

    def tr(self, domain=None, axis=None, **kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if domain is None or domain == self.domain:
            return self.val.dot(self.val, axis=axis)
        else:
            return super(vecvec_operator, self).tr(domain=domain, **kwargs)

    def inverse_tr(self):
        """
        Inverse is ill-defined for this operator.
        """
        raise AttributeError(about._errors.cstring(
            "ERROR: singular operator."))

    def diag(self, bare=False, domain=None, **kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if domain is None or (domain == self.domain):
            diag_val = self.val * self.val.conjugate()  # bare diagonal
            # weight if ...
            if not self.domain.discrete and not bare:
                diag_val = diag_val.weight(power=1, overwrite=True)
            return diag_val
        else:
            return super(vecvec_operator, self).diag(bare=bare,
                                                     domain=domain,
                                                     **kwargs)

    def inverse_diag(self):
        """
            Inverse is ill-defined for this operator.

        """
        raise AttributeError(about._errors.cstring(
            "ERROR: singular operator."))

    def det(self):
        """
            Computes the determinant of the operator.

            Returns
            -------
            det : 0
                The determinant

        """
        return 0

    def inverse_det(self):
        """
            Inverse is ill-defined for this operator.

        """
        raise AttributeError(about._errors.cstring(
            "ERROR: singular operator."))

    def log_det(self):
        """
            Logarithm of the determinant is ill-defined for this singular
            operator.

        """
        raise AttributeError(about._errors.cstring(
                "ERROR: singular operator."))

    def __repr__(self):
        return "<nifty_core.vecvec_operator>"


class response_operator(operator):
    """
        ..     _____   _______   _______   ______    ______    __ ___    _______   _______
        ..   /   __/ /   __  / /  _____/ /   _   | /   _   | /   _   | /  _____/ /   __  /
        ..  /  /    /  /____/ /_____  / /  /_/  / /  /_/  / /  / /  / /_____  / /  /____/
        .. /__/     \______/ /_______/ /   ____/  \______/ /__/ /__/ /_______/  \______/  operator class
        ..                            /__/

        NIFTY subclass for response operators (of a certain family)

        Any response operator handles Gaussian convolutions, itemwise masking,
        and selective mappings.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sigma : float, *optional*
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution. (default: 0)
        mask : {scalar, ndarray}, *optional*
            Masking values for arguments (default: 1)
        assign : {list, ndarray}, *optional*
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
            (default: None)
        den : bool, *optional*
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
            (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives
        sigma : float
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution.
        mask : {scalar, ndarray}
            Masking values for arguments
        assign : {list, ndarray}
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
        den : bool
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
    """

    def __init__(self, domain, codomain=None, sigma=0, mask=1, assign=None,
                 den=False, target=None, cotarget=None):
        """
            Sets the standard properties and `density`, `sigma`, `mask` and
            `assignment(s)`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sigma : float, *optional*
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution. (default: 0)
            mask : {scalar, ndarray}, *optional*
                Masking values for arguments (default: 1)
            assign : {list, ndarray}, *optional*
                Assignments of codomain items to domain items. A list of
                indices/ index tuples or a one/ two-dimensional array.
                (default: None)
            den : bool, *optional*
                Whether to consider the arguments as densities or not.
                Mandatory for the correct incorporation of volume weights.
                (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)

            Returns
            -------
            None
        """
        if not isinstance(domain, Space):
            raise TypeError(about._errors.cstring(
                "ERROR: The domain must be a space instance."))
        self.domain = domain

        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        self.sym = False
        self.uni = False
        self.imp = False
        self.den = bool(den)

        self.set_mask(new_mask=mask)

        # check sigma
        self.sigma = np.float(sigma)
        if sigma < 0:
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))

        # check assignment(s)
        if assign is None:
            assignments = self.domain.dim
            self.assign = None
        elif isinstance(assign, list):
            # check that the advanced indexing entries are either scalar
            # or all have the same size
            shape_list = map(np.shape, assign)
            shape_list.remove(())
            if len(self.domain.shape) == 1:
                if len(shape_list) == 0:
                    assignments = len(assign)
                elif len(shape_list) == 1:
                    assignments = len(assign[0])
                else:
                    raise ValueError(about._errors.cstring(
                        "ERROR: Wrong number of indices!"))
            else:
                if len(assign) != len(self.domain.shape):
                    raise ValueError(about._errors.cstring(
                        "ERROR: Wrong number of indices!"))
                elif shape_list == []:
                    raise ValueError(about._errors.cstring(
                        "ERROR: Purely scalar entries in the assign list " +
                        "are only valid for one-dimensional fields!"))
                elif not all([x == shape_list[0] for x in shape_list]):
                    raise ValueError(about._errors.cstring(
                        "ERROR: Non-scalar entries of assign list all must " +
                        "have the same shape!"))
                else:
                    assignments = np.prod(shape_list[0])
            self.assign = assign
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: assign must be None or list of arrays!"))

        if target is None:
            # set target
            # TODO: Fix the target spaces
            target = Space(assignments,
                                 dtype=self.domain.dtype,
                                 datamodel=self.domain.datamodel)
        else:
            # check target
            if not isinstance(target, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given target is not a nifty space"))
            elif not target.discrete:
                raise ValueError(about._errors.cstring(
                    "ERROR: Given target must be a discrete space!"))
            elif len(target.shape) > 1:
                raise ValueError(about._errors.cstring(
                    "ERROR: Given target must be a one-dimensional space."))
            elif assignments != target.dim:
                raise ValueError(about._errors.cstring(
                    "ERROR: dimension mismatch ( " +
                    str(assignments) + " <> " +
                    str(target.dim) + " )."))
        self.target = target
        if self.target.check_codomain(cotarget):
            self.cotarget = cotarget
        else:
            self.cotarget = self.target.get_codomain()

    def set_sigma(self, new_sigma):
        """
            Sets the standard deviation of the response operator, indicating
            the amount of convolution.

            Parameters
            ----------
            sigma : float
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution.

            Returns
            -------
            None
        """
        # check sigma
        if new_sigma < 0:
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        self.sigma = np.float(new_sigma)

    def set_mask(self, new_mask):
        """
            Sets the masking values of the response operator

            Parameters
            ----------
            newmask : {scalar, ndarray}
                masking values for arguments

            Returns
            -------
            None
        """
        if np.isscalar(new_mask):
            self.mask = np.bool(new_mask)
        else:
            self.mask = self.domain.cast(new_mask, dtype=np.dtype('bool'),
                                         hermitianize=False)

    def _multiply(self, x, **kwargs):
        # smooth
        y = self.domain.calc_smooth(x.val, sigma=self.sigma)
        # mask
        y *= self.mask
        # assign and return
        if self.assign is not None:
            val = y[self.assign]
        else:
            try:
                val = y.flatten(inplace=True)
            except TypeError:
                val = y.flatten()
        return Field(self.target,
                     val=val,
                     codomain=self.cotarget)

    def _adjoint_multiply(self, x, **kwargs):
        if self.assign is None:
            y = self.domain.cast(x.val)
        else:
            y = self.domain.cast(0)
            y[self.assign] = x.val

        y *= self.mask
        y = self.domain.calc_smooth(y, sigma=self.sigma)
        return Field(self.domain,
                     val=y,
                     codomain=self.codomain)

    def _briefing(self, x, domain, codomain, inverse):
        # make sure, that the result_field of the briefing lives in the
        # given domain and codomain
        result_field = Field(domain=domain, val=x, codomain=codomain,
                             copy=False)

        # weight if necessary
        if (not self.imp) and (not domain.discrete) and (not inverse) and \
                self.den:
            result_field = result_field.weight(power=1)
        return result_field

    def _debriefing(self, x, y, target, cotarget, inverse):
        # The debriefing takes care that the result field lives in the same
        # fourier-type domain as the input field
        assert(isinstance(y, Field))

        # weight if necessary
        if (not self.imp) and (not target.discrete) and \
                (not self.den ^ inverse):
            y = y.weight(power=-1)

        return y
#
#
#        # > evaluates x and y after `multiply`
#        if y is None:
#            return None
#        else:
#            # inspect y
#            if not isinstance(y, field):
#                y = field(target, codomain=cotarget, val=y)
#            elif y.domain != target:
#                raise ValueError(about._errors.cstring(
#                    "ERROR: invalid output domain."))
#            # weight if ...
#            if (not self.imp) and (not target.discrete) and \
#                    (not self.den ^ inverse):
#                y = y.weight(power=-1)
#            # inspect x
#            if isinstance(x, field):
#                # repair if the originally field was living in the codomain
#                # of the operators domain
#                if self.domain == self.target == x.codomain:
#                    y = y.transform(new_domain=x.domain)
#                if x.domain == y.domain and (x.codomain != y.codomain):
#                    y.set_codomain(new_codomain=x.codomain)
#            return y

    def __repr__(self):
        return "<nifty_core.response_operator>"


class invertible_operator(operator):
    """
        ..       __                                       __     __   __        __
        ..     /__/                                     /  /_  /__/ /  /      /  /
        ..     __   __ ___  __   __   _______   _____  /   _/  __  /  /___   /  /   _______
        ..   /  / /   _   ||  |/  / /   __  / /   __/ /  /   /  / /   _   | /  /  /   __  /
        ..  /  / /  / /  / |     / /  /____/ /  /    /  /_  /  / /  /_/  / /  /_ /  /____/
        .. /__/ /__/ /__/  |____/  \______/ /__/     \___/ /__/  \______/  \___/ \______/  operator class

        NIFTY subclass for invertible, self-adjoint (linear) operators

        The invertible operator class is an abstract class for self-adjoint or
        symmetric (linear) operators from which other more specific operator
        subclassescan be derived. Such operators inherit an automated inversion
        routine, namely conjugate gradient.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        uni : bool, *optional*
            Indicates whether the operator is unitary or not.
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False).
        para : {single object, tuple/list of objects}, *optional*
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use (default: None).

        See Also
        --------
        operator

        Notes
        -----
        This class is not meant to be instantiated. Operator classes derived
        from this one only need a `_multiply` or `_inverse_multiply` instance
        method to perform the other. However, one of them needs to be defined.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives.
        para : {single object, list of objects}
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use. Not used in the base operators.

    """

    def __init__(self, domain, codomain=None, uni=False, imp=False):
        """
            Sets the standard operator properties.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            uni : bool, *optional*
                Indicates whether the operator is unitary or not.
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False).
            para : {single object, tuple/list of objects}, *optional*
                This is a freeform tuple/list of parameters that derivatives of
                the operator class can use (default: None).

        """
        if not isinstance(domain, Space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        if self.domain.check_codomain(codomain):
            self.codomain = codomain
        else:
            self.codomain = self.domain.get_codomain()

        self.sym = True
        self.uni = bool(uni)

        if(self.domain.discrete):
            self.imp = True
        else:
            self.imp = bool(imp)

        self.target = self.domain
        self.cotarget = self.codomain

    def _multiply(self, x, force=False, W=None, spam=None, reset=None,
                  note=False, x0=None, tol=1E-4, clevel=1, limii=None,
                  **kwargs):
        """
            Applies the invertible operator to a given field by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable
                to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget
                previous
                conjugated directions (default: sqrt(b.dim)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed
                (default: 10 * b.dim).

        """
        x_, convergence = conjugate_gradient(self.inverse_times,
                                             x,
                                             W=W,
                                             spam=spam,
                                             reset=reset,
                                             note=note)(x0=x0,
                                                        tol=tol,
                                                        clevel=clevel,
                                                        limii=limii)
        # check convergence
        if not convergence:
            if not force or x_ is None:
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        # TODO: A weighting here shoud be wrong, as this is done by
        # the (de)briefing methods -> Check!
#        # weight if ...
#        if not self.imp:  # continiuos domain/target
#            x_.weight(power=-1, overwrite=True)
        return x_

    def _inverse_multiply(self, x, force=False, W=None, spam=None, reset=None,
                          note=False, x0=None, tol=1E-4, clevel=1, limii=None,
                          **kwargs):
        """
            Applies the inverse of the invertible operator to a given field by
            invoking a conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable
                to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget
                previous
                conjugated directions (default: sqrt(b.dim)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed
                (default: 10 * b.dim).

        """
        x_, convergence = conjugate_gradient(self.times,
                                             x,
                                             W=W,
                                             spam=spam,
                                             reset=reset,
                                             note=note)(x0=x0,
                                                        tol=tol,
                                                        clevel=clevel,
                                                        limii=limii)
        # check convergence
        if not convergence:
            if not force or x_ is None:
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        # TODO: A weighting here shoud be wrong, as this is done by
        # the (de)briefing methods -> Check!
#        # weight if ...
#        if not self.imp:  # continiuos domain/target
#            x_.weight(power=1, overwrite=True)
        return x_

    def __repr__(self):
        return "<nifty_tools.invertible_operator>"



class propagator_operator(operator):
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

    def __init__(self, S=None, M=None, R=None, N=None):
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

        # parse the signal prior covariance
        if not isinstance(S, operator):
            raise ValueError(about._errors.cstring(
                "ERROR: The given S is not an operator."))

        self.S = S
        self.S_inverse_times = self.S.inverse_times

        # take signal-space domain from S as the domain for D
        S_is_harmonic = False
        if hasattr(S.domain, 'harmonic'):
            if S.domain.harmonic:
                S_is_harmonic = True

        if S_is_harmonic:
            self.domain = S.codomain
            self.codomain = S.domain
        else:
            self.domain = S.domain
            self.codomain = S.codomain

        self.target = self.domain
        self.cotarget = self.codomain

        # build up the likelihood contribution
        (self.M_times,
         M_domain,
         M_codomain,
         M_target,
         M_cotarget) = self._build_likelihood_contribution(M, R, N)

        # assert that S and M have matching domains
        if not (self.domain == M_domain and
                self.codomain == M_codomain and
                self.target == M_target and
                self.cotarget == M_cotarget):
            raise ValueError(about._errors.cstring(
                "ERROR: The (co)domains and (co)targets of the prior " +
                "signal covariance and the likelihood contribution must be " +
                "the same in the sense of '=='."))

        self.sym = True
        self.uni = False
        self.imp = True

    def _build_likelihood_contribution(self, M, R, N):
        # if a M is given, return its times method and its domains
        # supplier and discard R and N
        if M is not None:
            return (M.times, M.domain, M.codomain, M.target, M.cotarget)

        if N is not None:
            if R is not None:
                return (lambda z: R.adjoint_times(N.inverse_times(R.times(z))),
                        R.domain, R.codomain, R.domain, R.codomain)
            else:
                return (N.inverse_times,
                        N.domain, N.codomain, N.target, N.cotarget)
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: At least M or N must be given."))

    def _multiply(self, x, W=None, spam=None, reset=None, note=False,
                  x0=None, tol=1E-4, clevel=1, limii=None, **kwargs):

        if W is None:
            W = self.S
        (result, convergence) = conjugate_gradient(self._inverse_multiply,
                                                   x,
                                                   W=W,
                                                   spam=spam,
                                                   reset=reset,
                                                   note=note)(x0=x0,
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


class propagator_operator_old(operator):
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

    def __init__(self, S=None, M=None, R=None, N=None):
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
        # check signal prior covariance
        if(S is None):
            raise Exception(about._errors.cstring(
                "ERROR: insufficient input."))
        elif(not isinstance(S, operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        space1 = S.domain

        # check likelihood (pseudo) covariance
        if(M is None):
            if(N is None):
                raise Exception(about._errors.cstring(
                    "ERROR: insufficient input."))
            elif(not isinstance(N, operator)):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input."))
            if(R is None):
                space2 = N.domain
            elif(not isinstance(R, operator)):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input."))
            else:
                space2 = R.domain
        elif(not isinstance(M, operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        else:
            space2 = M.domain

        # set spaces
        self.domain = space2

        if(self.domain.check_codomain(space1)):
            self.codomain = space1
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        self.target = self.domain
        self.cotarget = self.codomain

        # define A1 == S_inverse
        self.S = S
        if(isinstance(S, diagonal_operator)):
            self._A1 = S._inverse_multiply  # S.imp == True
        else:
            self._A1 = S.inverse_times

        # define A2 == M == R_adjoint N_inverse R == N_inverse
        if(M is None):
            if(R is not None):
                self.RN = (R, N)
                if(isinstance(N, diagonal_operator)):
                    self._A2 = self._standard_M_times_1
                else:
                    self._A2 = self._standard_M_times_2
            elif(isinstance(N, diagonal_operator)):
                self._A2 = N._inverse_multiply  # N.imp == True
            else:
                self._A2 = N.inverse_times
        elif(isinstance(M, diagonal_operator)):
            self._A2 = M._multiply  # M.imp == True
        else:
            self._A2 = M.times

        self.sym = True
        self.uni = False
        self.imp = True

    # applies > R_adjoint N_inverse R assuming N is diagonal
    def _standard_M_times_1(self, x, **kwargs):
        # N.imp = True
        return self.RN[0].adjoint_times(self.RN[1]._inverse_multiply(self.RN[0].times(x)))

    def _standard_M_times_2(self, x, **kwargs):  # applies > R_adjoint N_inverse R
        return self.RN[0].adjoint_times(self.RN[1].inverse_times(self.RN[0].times(x)))

    # > applies A1 + A2 in self.codomain
    def _inverse_multiply_1(self, x, **kwargs):
        return self._A1(x, pseudo=True) + self._A2(x.transform(self.domain)).transform(self.codomain)

    def _inverse_multiply_2(self, x, **kwargs):  # > applies A1 + A2 in self.domain
        transformed_x = x.transform(self.codomain)
        aed_x = self._A1(transformed_x, pseudo=True)
        #print (vars(aed_x),)
        transformed_aed = aed_x.transform(self.domain)
        #print (vars(transformed_aed),)
        temp_to_add = self._A2(x)
        added = transformed_aed + temp_to_add
        return added
        # return
        # self._A1(x.transform(self.codomain),pseudo=True).transform(self.domain)+self._A2(x)

    def _briefing(self, x):  # > prepares x for `multiply`
        # inspect x
        if not isinstance(x, Field):
            return (Field(self.domain, codomain=self.codomain,
                          val=x),
                    False)
        # check x.domain
        elif x.domain == self.domain:
            return (x, False)
        elif x.domain == self.codomain:
            return (x, True)
        # transform
        else:
            return (x.transform(new_domain=self.codomain,
                                overwrite=False),
                    True)

    def _debriefing(self, x, x_, in_codomain):  # > evaluates x and x_ after `multiply`
        if x_ is None:
            return None
        # inspect x
        elif isinstance(x, Field):
            # repair ...
            if in_codomain == True and x.domain != self.codomain:
                x_ = x_.transform(new_domain=x.domain)  # ... domain
            if x_.codomain != x.codomain:
                x_.set_codomain(new_codomain=x.codomain)  # ... codomain
        return x_

    def times(self, x, W=None, spam=None, reset=None, note=False, x0=None, tol=1E-4, clevel=1, limii=None, **kwargs):
        """
            Applies the propagator to a given object by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            Dx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
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

        """
        if W is None:
            W = self.S
        # prepare
        x_, in_codomain = self._briefing(x)
        # apply operator
        if(in_codomain):
            A = self._inverse_multiply_1
        else:
            A = self._inverse_multiply_2
        x_, convergence = conjugate_gradient(A, x_, W=W, spam=spam, reset=reset, note=note)(
            x0=x0, tol=tol, clevel=clevel, limii=limii)
        # evaluate
        if not convergence:
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        return self._debriefing(x, x_, in_codomain)

    def inverse_times(self, x, **kwargs):
        """
            Applies the inverse propagator to a given object.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            DIx : field
                Mapped field with suitable domain.

        """
        # prepare
        x_, in_codomain = self._briefing(x)
        # apply operator
        if(in_codomain):
            x_ = self._inverse_multiply_1(x_)
        else:
            x_ = self._inverse_multiply_2(x_)
        # evaluate
        return self._debriefing(x, x_, in_codomain)

    def __repr__(self):
        return "<nifty_tools.propagator_operator>"
