# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
##
# Copyright (C) 2013 Max-Planck-Society
##
# Author: Marco Selig
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                 __   ____   __
    ..               /__/ /   _/ /  /_
    ..     __ ___    __  /  /_  /   _/  __   __
    ..   /   _   | /  / /   _/ /  /   /  / /  /
    ..  /  / /  / /  / /  /   /  /_  /  /_/  /
    .. /__/ /__/ /__/ /__/    \___/  \___   /  explicit
    ..                              /______/

    This module extends NIFTY's versatility to the usage of explicit matrix
    representations of linear operator by the :py:class:`explicit_operator`.
    In order to access explicit operators, this module provides the
    :py:class:`explicit_probing` class and the :py:func:`explicify` function.
    Those objects are supposed to support the user in solving information field
    theoretical problems in low (or moderate) dimensions, or in debugging
    algorithms by studying operators in detail.

"""
from __future__ import division
from sys import stdout as so
import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf
from multiprocessing import Pool as mp
from multiprocessing import Value as mv
from multiprocessing import Array as ma

from nifty.keepers import about
from nifty.nifty_core import space,                                              \
    field
from nifty_operators import operator,\
    diagonal_operator,\
    identity,\
    vecvec_operator
from nifty_probing_old import probing


# -----------------------------------------------------------------------------

class explicit_operator(operator):
    """
        ..
        ..
        ..                                    __     __             __   __
        ..                                  /  /   /__/           /__/ /  /_
        ..    _______  __   __    ______   /  /    __   _______   __  /   _/
        ..  /   __  / \  \/  /  /   _   | /  /   /  / /   ____/ /  / /  /
        .. /  /____/  /     /  /  /_/  / /  /_  /  / /  /____  /  / /  /_
        .. \______/  /__/\__\ /   ____/  \___/ /__/  \______/ /__/  \___/  operator class
        ..                   /__/

        NIFTY subclass for explicit linear operators.

        This class essentially supports linear operators with explicit matrix
        representation in the NIFTY framework. Note that this class is not
        suited for handling huge dimensionalities.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        matrix : {list, array}
            The matrix representation of the operator, ideally shaped in 2D
            according to dimensionality of target and domain (default: None).
        bare : {bool, 2-tuple}, *optional*
            Whether the matrix entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: True)
        sym : bool, *optional*
            Indicates whether the operator is self-adjoint or not
            (default: False).
        uni : bool, *optional*
            Indicates whether the operator is unitary or not (default: False).
        target : space, *optional*
            The space wherein the operator output lives (default: domain).

        See Also
        --------
        explicify : A function that returns an explicit oparator given an
            implicit one.

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

        Examples
        --------
        >>> x_space = rg_space(2) # default `dist` == 0.5
        >>> A = explicit_operator(x_space, matrix=[[2, 0], [1, 1]], bare=False)
        >>> A.get_matrix(bare=False)
        array([[2, 0],
               [1, 1]])
        >>> A.get_matrix(bare=True)
        array([[4, 0],
               [2, 2]])
        >>> c = field(x_space, val=[3, 5])
        >>> A(c).val
        array([ 6.,  8.])
        >>> A.inverse()
        <nifty_explicit.explicit_operator>
        >>> (A * A.inverse()).get_matrix(bare=False) # == identity
        array([[ 1.,  0.],
               [ 0.,  1.]])
        >>> B = A + diagonal_operator(x_space, diag=2, bare=False)
        >>> B.get_matrix(bare=False)
        array([[ 4.,  0.],
               [ 1.,  3.]])
        >>> B(c).val
        array([ 12.,  18.])
        >>> B.tr()
        7.0
        >>> B.det()
        12.000000000000005

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : array
            An array containing the `bare` matrix entries.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives.
        _hidden : array
            An array specifying hidden degrees of freedom in domain and target.
        _inv : array
            An array containing the inverse matrix; set when needed.

    """
    epsilon = 1E-12  # absolute precision for comparisons to identity

    def __init__(self, domain, matrix=None, bare=True, sym=None, uni=None, target=None):
        """
            Initializes the explicit operator and sets the standard operator
            properties as well as `values`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            matrix : {list, array}
                The matrix representation of the operator, ideally shaped in 2D
                according to dimensionality of target and domain (default: None).
            bare : {bool, 2-tuple}, *optional*
                Whether the matrix entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True).
            sym : bool, *optional*
                Indicates whether the operator is self-adjoint or not
                (default: False).
            uni : bool, *optional*
                Indicates whether the operator is unitary or not (default: False).
            target : space, *optional*
                The space wherein the operator output lives (default: domain).

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

            Raises
            ------
            TypeError
                If invalid input is given.
            ValueError
                If dimensions of `domain`, `target`, and `matrix` mismatch;
                or if `bare` is invalid.

        """
        # check domain
        if(not isinstance(domain, space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        # check matrix and target
        if(matrix is None):
            if(target is None):
                val = np.zeros((self.domain.dim(split=False), self.domain.dim(
                    split=False)), dtype=np.int, order='C')
                target = self.domain
            else:
                if(not isinstance(target, space)):
                    raise TypeError(about._errors.cstring(
                        "ERROR: invalid input."))
                elif(target != self.domain):
                    sym = False
                    uni = False
                val = np.zeros((target.dim(split=False), self.domain.dim(
                    split=False)), dtype=np.int, order='C')
        elif(np.size(matrix, axis=None) % self.domain.dim(split=False) == 0):
            val = np.array(matrix).reshape((-1, self.domain.dim(split=False)))
            if(target is not None):
                if(not isinstance(target, space)):
                    raise TypeError(about._errors.cstring(
                        "ERROR: invalid input."))
                elif(val.shape[0] != target.dim(split=False)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: dimension mismatch ( " + str(val.shape[0]) + " <> " + str(target.dim(split=False)) + " )."))
                elif(target != self.domain):
                    sym = False
                    uni = False
            elif(val.shape[0] == val.shape[1]):
                target = self.domain
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: insufficient input."))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(matrix, axis=None)) + " <> " + str(self.domain.dim(split=False)) + " )."))
#        if(val.size>1048576):
#            about.infos.cprint("INFO: matrix size > 2 ** 20.")
        self.target = target

        # check dtype
        if(np.any(np.iscomplex(val))):
            dtype, purelyreal = max(min(val.dtype, self.domain.dtype), min(
                val.dtype, self.target.dtype)), False
        else:
            dtype, purelyreal = max(min(val.dtype, self.domain.vol.dtype), min(
                val.dtype, self.target.vol.dtype)), True
        # weight if ... (given `domain` and `target`)
        if(isinstance(bare, tuple)):
            if(len(bare) != 2):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input."))
            else:
                val = self._calc_weight_rows(val, power=-int(not bare[0]))
                val = self._calc_weight_cols(val, power=-int(not bare[1]))
        elif(not bare):
            val = self._calc_weight_rows(val, -1)
        if(purelyreal):
            self.val = np.real(val).astype(dtype)
        else:
            self.val = val.astype(dtype)

        # check hidden degrees of freedom
        self._hidden = np.array([self.domain.dim(split=False) < self.domain.dof(
        ), self.target.dim(split=False) < self.target.dof()], dtype=np.bool)
#        if(np.any(self._hidden)):
#            about.infos.cprint("INFO: inappropriate space.")

        # check flags
        self.sym, self.uni = self._check_flags(sym=sym, uni=uni)
        if(self.domain.discrete)and(self.target.discrete):
            self.imp = True
        else:
            self.imp = False  # bare matrix is stored for efficiency

        self._inv = None  # defined when needed

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _check_flags(self, sym=None, uni=None):  # > determine `sym` and `uni`
        if(self.val.shape[0] == self.val.shape[1]):
            if(sym is None):
                adj = np.conjugate(self.val.T)
                sym = np.all(np.absolute(self.val - adj) < self.epsilon)
                if(uni is None):
                    uni = np.all(np.absolute(self._calc_mul(adj, 0) - np.diag(
                        1 / self.target.get_meta_volume(total=False).flatten(order='C'), k=0)) < self.epsilon)
            elif(uni is None):
                adj = np.conjugate(self.val.T)
                uni = np.all(np.absolute(self._calc_mul(adj, 0) - np.diag(
                    1 / self.target.get_meta_volume(total=False).flatten(order='C'), k=0)) < self.epsilon)
            return bool(sym), bool(uni)
        else:
            return False, False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _set_inverse(self):  # > define inverse matrix
        if(self._inv is None):
            if(np.any(self._hidden)):
                about.warnings.cprint("WARNING: inappropriate inversion.")
            self._inv = np.linalg.inv(self.weight(
                rowpower=1, colpower=1, overwrite=False))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def cast_domain(self, newdomain):
        """
            Casts the domain of the operator.

            Parameters
            ----------
            newdomain : space
                New space wherein valid argument lives.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If `newdomain` is no instance of the nifty_core.space class
            ValueError
                If `newdomain` does not match the (unsplit) dimensionality of
                the current domain.

        """
        if(not isinstance(newdomain, space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(newdomain.dim(split=False) != self.domain.dim(split=False)):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                newdomain.dim(split=False)) + " <> " + str(self.domain.dim(split=False)) + " )."))
        self.domain = newdomain

    def cast_target(self, newtarget):
        """
            Casts the target of the operator.

            Parameters
            ----------
            newtarget : space
                New space wherein the operator output lives.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If `newtarget` is no instance of the nifty_core.space class
            ValueError
                If `newtarget` does not match the (unsplit) dimensionality of
                the current target.

        """
        if(not isinstance(newtarget, space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(newtarget.dim(split=False) != self.target.dim(split=False)):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                newtarget.dim(split=False)) + " <> " + str(self.target.dim(split=False)) + " )."))
        self.target = newtarget

    def cast_spaces(self, newdomain=None, newtarget=None):
        """
            Casts the domain and/or the target of the operator.

            Parameters
            ----------
            newdomain : space, *optional*
                New space wherein valid argument lives (default: None).
            newtarget : space, *optional*
                New space wherein the operator output lives (default: None).

            Returns
            -------
            None

        """
        if(newdomain is not None):
            self.cast_domain(newdomain)
        if(newtarget is not None):
            self.cast_target(newtarget)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_matrix(self, newmatrix, bare=True, sym=None, uni=None):
        """
            Resets the entire matrix.

            Parameters
            ----------
            matrix : {list, array}
                New matrix representation of the operator, ideally shaped in 2D
                according to dimensionality of target and domain (default: None).
            bare : {bool, 2-tuple}, *optional*
                Whether the new matrix entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True).
            sym : bool, *optional*
                Indicates whether the new operator is self-adjoint or not
                (default: False).
            uni : bool, *optional*
                Indicates whether the new operator is unitary or not
                (default: False).

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

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If matrix' dimensions mismatch;
                or if `bare` is invalid.

        """
        # check matrix
        if(np.size(newmatrix, axis=None) == self.domain.dim(split=False) * self.target.dim(split=False)):
            val = np.array(newmatrix).reshape(
                (self.target.dim(split=False), self.domain.dim(split=False)))
            if(self.target != self.domain):
                sym = False
                uni = False
#            if(val.size>1048576):
#                about.infos.cprint("INFO: matrix size > 2 ** 20.")
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(np.size(
                newmatrix, axis=None)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))

        # check dtype
        if(np.any(np.iscomplex(val))):
            dtype, purelyreal = max(min(val.dtype, self.domain.dtype), min(
                val.dtype, self.target.dtype)), False
        else:
            dtype, purelyreal = max(min(val.dtype, self.domain.vol.dtype), min(
                val.dtype, self.target.vol.dtype)), True
        # weight if ... (given `domain` and `target`)
        if(isinstance(bare, tuple)):
            if(len(bare) != 2):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input."))
            else:
                val = self._calc_weight_rows(val, power=-int(not bare[0]))
                val = self._calc_weight_cols(val, power=-int(not bare[1]))
        elif(not bare):
            val = self._calc_weight_rows(val, -1)
        if(purelyreal):
            self.val = np.real(val).astype(dtype)
        else:
            self.val = val.astype(dtype)

        # check flags
        self.sym, self.uni = self._check_flags(sym=sym, uni=uni)
        self._inv = None  # reset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_matrix(self, bare=True):
        """
            Returns the entire matrix.

            Parameters
            ----------
            bare : {bool, 2-tuple}, *optional*
                Whether the returned matrix entries are `bare` or not
                (default: True).

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

            Returns
            -------
            matrix : numpy.array
                The matrix representation of the operator, shaped in 2D
                according to dimensionality of target and domain.

            Raises
            ------
            ValueError
                If `bare` is invalid.

        """
        if(bare == True)or(self.imp):
            return self.val
        # weight if ...
        elif(isinstance(bare, tuple)):
            if(len(bare) != 2):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input."))
            else:
                return self.weight(rowpower=int(not bare[0]), colpower=int(not bare[1]), overwrite=False)
        elif(not bare):
            return self.weight(rowpower=int(not bare), colpower=0, overwrite=False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_weight_row(self, x, power):  # > weight row and flatten
        return self.domain.calc_weight(x, power=power).flatten(order='C')

    def _calc_weight_col(self, x, power):  # > weight column and flatten
        return self.target.calc_weight(x, power=power).flatten(order='C')

    def _calc_weight_rows(self, X, power=1):  # > weight all rows
        if(np.any(np.iscomplex(X)))and(not issubclass(self.domain.dtype, np.complexfloating)):
            return (np.apply_along_axis(self._calc_weight_row, 1, np.real(X), power) + np.apply_along_axis(self._calc_weight_row, 1, np.imag(X), power) * 1j)
        else:
            return np.apply_along_axis(self._calc_weight_row, 1, X, power)

    def _calc_weight_cols(self, X, power=1):  # > weight all columns
        if(np.any(np.iscomplex(X)))and(not issubclass(self.target.dtype, np.complexfloating)):
            return (np.apply_along_axis(self._calc_weight_col, 0, np.real(X), power) + np.apply_along_axis(self._calc_weight_col, 0, np.imag(X), power) * 1j)
        else:
            return np.apply_along_axis(self._calc_weight_col, 0, X, power)

    def weight(self, rowpower=0, colpower=0, overwrite=False):
        """
            Returns the entire matrix, weighted with the volume factors to a
            given power. The matrix entries will optionally be overwritten.

            Parameters
            ----------
            rowpower : scalar, *optional*
                Specifies the power of the volume factors applied to the rows
                of the matrix (default: 0).
            rowpower : scalar, *optional*
                Specifies the power of the volume factors applied to the columns
                of the matrix (default: 0).
            overwrite : bool, *optional*
                Whether to overwrite the matrix or not (default: False).

            Returns
            -------
            field : field, *optional*
                If overwrite is ``False``, the weighted matrix is returned;
                otherwise, nothing is returned.

        """
        if(overwrite):
            if(not self.domain.discrete)and(rowpower):  # rowpower <> 0
                self.val = self._calc_weight_rows(self.val, rowpower)
            if(not self.target.discrete)and(colpower):  # colpower <> 0
                self.val = self._calc_weight_cols(self.val, colpower)
        else:
            X = np.copy(self.val)
            if(not self.domain.discrete)and(rowpower):  # rowpower <> 0
                X = self._calc_weight_rows(X, rowpower)
            if(not self.target.discrete)and(colpower):  # colpower <> 0
                X = self._calc_weight_cols(X, colpower)
            return X

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self, x, **kwargs):  # > applies the matirx to a given field
        if(self._hidden[0]):  # hidden degrees of freedom
            x_ = np.apply_along_axis(
                self.domain.calc_dot, 1, self.val, np.conjugate(x.val))
        else:
            x_ = np.dot(self.val, x.val.flatten(order='C'), out=None)
        return x_

    # > applies the adjoint operator to a given field
    def _adjoint_multiply(self, x, **kwargs):
        if(self._hidden[1]):  # hidden degrees of freedom
            x_ = np.apply_along_axis(
                self.target.calc_dot, 0, np.conjugate(self.val), np.conjugate(x.val))
        else:
            x_ = np.dot(np.conjugate(self.val.T),
                        x.val.flatten(order='C'), out=None)
        return x_

    # > applies the inverse operator to a given field
    def _inverse_multiply(self, x, **kwargs):
        if(self._hidden[1]):  # hidden degrees of freedom
            x_ = np.apply_along_axis(
                self.target.calc_dot, 1, self._inv, np.conjugate(x.val))
        else:
            x_ = np.dot(self._inv, x.val.flatten(order='C'), out=None)
        return x_

    # > applies the adjoint inverse operator to a given field
    def _inverse_adjoint_multiply(self, x, **kwargs):
        if(self._hidden[0]):  # hidden degrees of freedom
            x_ = np.apply_along_axis(
                self.domain.calc_dot, 0, np.conjugate(self.val), np.conjugate(x.val))
        else:
            x_ = np.dot(np.conjugate(self._inv.T),
                        x.val.flatten(order='C'), out=None)
        return x_

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self, x, domain, inverse):  # > prepares x for `multiply`
        # inspect x
        if(not isinstance(x, field)):
            x_ = field(domain, val=x, target=None)
        else:
            # check x.domain
            if(domain == x.domain):
                x_ = x
            # transform
            else:
                x_ = x.transform(target=domain, overwrite=False)
        # weight if ...
        if(not self.imp)and(not domain.discrete):
            x_ = x_.weight(power=1, overwrite=False)
        return x_

    def _debriefing(self, x, x_, target, inverse):  # > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            # inspect x_
            if(not isinstance(x_, field)):
                x_ = field(target, val=x_, target=None)
            elif(x_.domain != target):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid output domain."))
            # inspect x
            if(isinstance(x, field)):
                # repair ...
                if(self.domain == self.target != x.domain):
                    x_ = x_.transform(
                        target=x.domain, overwrite=False)  # ... domain
                if(x_.domain == x.domain)and(x_.target != x.target):
                    x_.set_target(newtarget=x.target)  # ... codomain
            return x_

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        # check whether self-inverse
        if(self.sym)and(self.uni):
            return self.times(x, **kwargs)

        # check whether square matrix
        elif(self.nrow() != self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for " +
                                                   str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))

        self._set_inverse()

        # prepare
        x_ = self._briefing(x, self.target, True)
        # apply operator
        x_ = self._inverse_multiply(x_, **kwargs)
        # evaluate
        return self._debriefing(x, x_, self.domain, True)

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

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        return self.inverse_adjoint_times(x, **kwargs)

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

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        # check whether square matrix
        if(self.nrow() != self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for " +
                                                   str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))

        # check whether self-adjoint
        if(self.sym):
            return self._inverse_times(x, **kwargs)
        # check whether unitary
        if(self.uni):
            return self.times(x, **kwargs)

        self._set_inverse()

        # prepare
        x_ = self._briefing(x, self.domain, True)
        # apply operator
        x_ = self._inverse_adjoint_multiply(x_, **kwargs)
        # evaluate
        return self._debriefing(x, x_, self.target, True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self, domain=None, **kwargs):
        """
            Computes the trace of the operator.

            Parameters
            ----------
            domain : space, *optional*
                Space wherein the probes live (default: self.domain).
            target : space, *optional*
                Space wherein the transform of the probes live (default: None).
            random : string, *optional*
                Specifies the pseudo random number generator (default: "pm1");
                supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard deviation or variance)

            ncpu : integer, *optional*
                Number of used CPUs to use (default: 2).
            nrun : integer, *optional*
                Total number of probes (default: 8).
            nper : integer, *optional*
                Number of tasks performed by one worker (default: 1).
            var : bool, *optional*
                Whether to additionally return the probing variance or not
                (default: False).
            loop : bool, *optional*
                Whether to perform a loop or to parallelise (default: False).

            Returns
            -------
            tr : float
                Trace of the operator.
            delta : float, *optional*
                Probing variance of the trace; returned if `var` is ``True`` in
                case of probing.

            See Also
            --------
            trace_probing

            Raises
            ------
            ValueError
                If `domain` and `target` are unequal.

        """
        if(self.domain != self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: trace ill-defined."))

        if(domain is None)or(domain == self.domain):
            diag = self.diag(bare=False, domain=self.domain)
            if(self._hidden[0]):  # hidden degrees of freedom
                # discrete inner product
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True), dtype=self.domain.dtype, order='C'), diag)
            else:
                return np.sum(diag, axis=None, dtype=None, out=None)
        else:
            # probing
            return super(explicit_operator, self).tr(domain=domain, **kwargs)

    def inverse_tr(self, domain=None, **kwargs):
        """
            Computes the trace of the inverse operator.

            Parameters
            ----------
            domain : space, *optional*
                Space wherein the probes live (default: self.domain).
            target : space, *optional*
                Space wherein the transform of the probes live (default: None).
            random : string, *optional*
                Specifies the pseudo random number generator (default: "pm1");
                supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard deviation or variance)

            ncpu : integer, *optional*
                Number of used CPUs to use (default: 2).
            nrun : integer, *optional*
                Total number of probes (default: 8).
            nper : integer, *optional*
                Number of tasks performed by one worker (default: 1).
            var : bool, *optional*
                Whether to additionally return the probing variance or not
                (default: False).
            loop : bool, *optional*
                Whether to perform a loop or to parallelise (default: False).

            Returns
            -------
            tr : float
                Trace of the operator.
            delta : float, *optional*
                Probing variance of the trace; returned if `var` is ``True`` in
                case of probing.

            See Also
            --------
            trace_probing

            Raises
            ------
            ValueError
                If `domain` and `target` are unequal.

        """
        if(self.domain != self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: trace ill-defined."))

        if(domain is None)or(domain == self.domain):
            diag = self.inverse_diag(bare=False, domain=self.domain)
            if(self._hidden[0]):  # hidden degrees of freedom
                # discrete inner product
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True), dtype=self.domain.dtype, order='C'), diag)
            else:
                return np.sum(diag, axis=None, dtype=None, out=None)
        else:
            # probing
            return super(explicit_operator, self).tr(domain=domain, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self, bare=False, domain=None, **kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Whether the returned diagonal entries are `bare` or not
                (default: True).
            domain : space, *optional*
                Space wherein the probes live (default: self.domain).
            target : space, *optional*
                Space wherein the transform of the probes live (default: None).
            random : string, *optional*
                Specifies the pseudo random number generator (default: "pm1");
                supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard deviation or variance)

            ncpu : integer, *optional*
                Number of used CPUs to use (default: 2).
            nrun : integer, *optional*
                Total number of probes (default: 8).
            nper : integer, *optional*
                Number of tasks performed by one worker (default: 1).
            var : bool, *optional*
                Whether to additionally return the probing variance or not
                (default: False).
            save : bool, *optional*
                Whether all individual probing results are saved or not
                (default: False).
            path : string, *optional*
                Path wherein the results are saved (default: "tmp").
            prefix : string, *optional*
                Prefix for all saved files (default: "")
            loop : bool, *optional*
                Whether to perform a loop or to parallelise (default: False).

            Returns
            -------
            diag : ndarray
                Diagonal of the operator.
            delta : float, *optional*
                Probing variance of the trace; returned if `var` is ``True`` in
                case of probing.

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

            See Also
            --------
            diagonal_probing

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        if(self.val.shape[0] != self.val.shape[1]):
            raise ValueError(about._errors.cstring("ERROR: diagonal ill-defined for " + str(
                self.val.shape[0]) + " x " + str(self.val.shape[1]) + " matrices."))
        if(self.domain != self.target)and(not bare):
            about.warnings.cprint("WARNING: ambiguous non-bare diagonal.")

        if(domain is None)or(domain == self.domain):
            diag = np.diagonal(self.val, offset=0, axis1=0, axis2=1)
            # weight if ...
            if(not self.domain.discrete)and(not bare):
                diag = self.domain.calc_weight(diag, power=1)
                # check complexity
                if(np.all(np.imag(diag) == 0)):
                    diag = np.real(diag)
            return diag
        elif(domain == self.target):
            diag = np.diagonal(np.conjugate(self.val.T),
                               offset=0, axis1=0, axis2=1)
            # weight if ...
            if(not self.target.discrete)and(not bare):
                diag = self.target.calc_weight(diag, power=1)
                # check complexity
                if(np.all(np.imag(diag) == 0)):
                    diag = np.real(diag)
            return diag
        else:
            # probing
            return super(explicit_operator, self).diag(bare=bare, domain=domain, **kwargs)

    def inverse_diag(self, bare=False, domain=None, **kwargs):
        """
            Computes the diagonal of the inverse operator.

            Parameters
            ----------
            bare : bool, *optional*
                Whether the returned diagonal entries are `bare` or not
                (default: True).
            domain : space, *optional*
                Space wherein the probes live (default: self.domain).
            target : space, *optional*
                Space wherein the transform of the probes live (default: None).
            random : string, *optional*
                Specifies the pseudo random number generator (default: "pm1");
                supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard deviation or variance)

            ncpu : integer, *optional*
                Number of used CPUs to use (default: 2).
            nrun : integer, *optional*
                Total number of probes (default: 8).
            nper : integer, *optional*
                Number of tasks performed by one worker (default: 1).
            var : bool, *optional*
                Whether to additionally return the probing variance or not
                (default: False).
            save : bool, *optional*
                Whether all individual probing results are saved or not
                (default: False).
            path : string, *optional*
                Path wherein the results are saved (default: "tmp").
            prefix : string, *optional*
                Prefix for all saved files (default: "")
            loop : bool, *optional*
                Whether to perform a loop or to parallelise (default: False).

            Returns
            -------
            diag : ndarray
                Diagonal of the inverse operator.
            delta : float, *optional*
                Probing variance of the trace; returned if `var` is ``True`` in
                case of probing.

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

            See Also
            --------
            diagonal_probing

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        if(self.val.shape[0] != self.val.shape[1]):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for " + str(
                self.val.shape[0]) + " x " + str(self.val.shape[1]) + " matrices."))
        if(self.domain != self.target)and(not bare):
            about.warnings.cprint("WARNING: ambiguous non-bare diagonal.")

        if(domain is None)or(domain == self.target):
            self._set_inverse()
            diag = np.diagonal(self._inv, offset=0, axis1=0, axis2=1)
            # weight if ...
            if(not self.target.discrete)and(not bare):
                diag = self.target.calc_weight(diag, power=1)
                # check complexity
                if(np.all(np.imag(diag) == 0)):
                    diag = np.real(diag)
            return diag
        elif(domain == self.domain):
            self._set_inverse()
            diag = np.diagonal(np.conjugate(self._inv.T),
                               offset=0, axis1=0, axis2=1)
            # weight if ...
            if(not self.domain.discrete)and(not bare):
                diag = self.domain.calc_weight(diag, power=1)
                # check complexity
                if(np.all(np.imag(diag) == 0)):
                    diag = np.real(diag)
            return diag
        else:
            # probing
            return super(explicit_operator, self).inverse_diag(bare=bare, domain=domain, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the operator.

            Parameters
            ----------
            None

            Returns
            -------
            det : float
                Determinant of the operator.

            Raises
            ------
            ValueError
                If `domain` and `target` are unequal.

        """
        if(self.domain != self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: determinant ill-defined."))

        if(np.any(self._hidden)):
            about.warnings.cprint(
                "WARNING: inappropriate determinant calculation.")
        det = np.linalg.det(self.weight(
            rowpower=0.5, colpower=0.5, overwrite=False))
        if(np.isreal(det)):
            return np.asscalar(np.real(det))
        else:
            return det

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator.

            Parameters
            ----------
            None

            Returns
            -------
            det : float
                Determinant of the inverse operator.

            Raises
            ------
            ValueError
                If it is a singular matrix.

        """
        det = self.det()
        if(det <> 0):
            return 1 / det
        else:
            raise ValueError(about._errors.cstring("ERROR: singular matrix."))

    def log_det(self):
        """
            Computes the logarithm of the determinant of the operator
            (if applicable).

            Returns
            -------
            logdet : float
                The logarithm of the determinant

            See Also
            --------
            numpy.linalg.slogdet

            Raises
            ------
            ValueError
                If `domain` and `target` are unequal or it is non-positive
                definite matrix.

        """
        if(self.domain != self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: determinant ill-defined."))

        if(np.any(self._hidden)):
            about.warnings.cprint(
                "WARNING: inappropriate determinant calculation.")
        sign, logdet = np.linalg.slogdet(self.weight(
            rowpower=0.5, colpower=0.5, overwrite=False))
        if(abs(sign) < 0.1):  # abs(sign) << 1
            raise ValueError(about._errors.cstring("ERROR: singular matrix."))
        if(sign == -1):
            raise ValueError(about._errors.cstring(
                "ERROR: non-positive definite matrix."))
        else:
            logdet += np.log(sign)
            if(np.isreal(logdet)):
                return np.asscalar(np.real(logdet))
            else:
                return logdet

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __len__(self):
        return int(self.nrow()[0])

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, value):
        self.val[key] = self.val.dtype(value)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __pos__(self):
        return explicit_operator(self.domain, self.val, bare=True, sym=self.sym, uni=self.uni, target=self.target)

    def __neg__(self):
        return explicit_operator(self.domain, -self.val, bare=True, sym=self.sym, uni=self.uni, target=self.target)

    def __abs__(self):
        return explicit_operator(self.domain, np.absolute(self.val), bare=True, sym=self.sym, uni=self.uni, target=self.target)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def transpose(self):
        """
            Computes the transposed matrix.

            Returns
            -------
            T : explicit_operator
                The transposed matrix.

        """
        return explicit_operator(self.target, self.val.T, bare=True, sym=self.sym, uni=self.uni, target=self.domain)

    def conjugate(self):
        """
            Computes the complex conjugated matrix.

            Returns
            -------
            CC : explicit_operator
                The complex conjugated matrix.

        """
        return explicit_operator(self.domain, np.conjugate(self.val), bare=True, sym=self.sym, uni=self.uni, target=self.target)

    def adjoint(self):
        """
            Computes the adjoint matrix.

            Returns
            -------
            A : explicit_operator
                The adjoint matrix.

        """
        return explicit_operator(self.target, np.conjugate(self.val.T), bare=True, sym=self.sym, uni=self.uni, target=self.domain)

    def inverse(self):
        """
            Computes the inverted matrix.

            Returns
            -------
            I : explicit_operator
                The inverted matrix.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        # check whether square matrix
        if(self.nrow() != self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for " +
                                                   str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
        self._set_inverse()
        return explicit_operator(self.target, self._inv, bare=True, sym=self.sym, uni=self.uni, target=self.domain)

    def adjoint_inverse(self):
        """
            Computes the adjoint inverted matrix.

            Returns
            -------
            AI : explicit_operator
                The adjoint inverted matrix.

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        return self.inverse_adjoint()

    def inverse_adjoint(self):
        """
            Computes the inverted adjoint matrix.

            Returns
            -------
            IA : explicit_operator
                The inverted adjoint matrix.

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        # check whether square matrix
        if(self.nrow() != self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for " +
                                                   str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
        self._set_inverse()
        return explicit_operator(self.target, np.conjugate(self._inv.T), bare=True, sym=self.sym, uni=self.uni, target=self.domain)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __add__(self, X):  # __add__ : self + X
        if(isinstance(X, operator)):
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X, explicit_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = self.val + X.val
            elif(isinstance(X, diagonal_operator)):
                if(self.target.dim(split=False) != X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: incompatible codomains."))
                # domain == X.domain
                matrix = self.val + \
                    np.diag(X.diag(bare=True, domain=None), k=0)
            elif(isinstance(X, vecvec_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = self.val + np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            if(self.nrow() != self.ncol()):
                raise ValueError(about._errors.cstring(
                    "ERROR: identity ill-defined for " + str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
            sym = self.sym
            uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = self.domain.calc_weight(X, power=-1).astype(
                max(min(X.dtype, self.domain.dtype), min(X.dtype, self.target.dtype)))
            matrix = self.val + np.diag(X.flatten(order='C'), k=0)
        elif(np.size(X) == np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
            matrix = self.val + X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))
        return explicit_operator(self.domain, matrix, bare=True, sym=sym, uni=uni, target=self.target)

    __radd__ = __add__  # __add__ : X + self

    def __iadd__(self, X):  # __iadd__ : self += X
        if(isinstance(X, operator)):
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
            self.sym = (self.sym and X.sym)
            self.uni = None
            if(isinstance(X, explicit_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                X = X.val
            elif(isinstance(X, diagonal_operator)):
                if(self.target.dim(split=False) != X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: incompatible codomains."))
                X = np.diag(X.diag(bare=True, domain=None),
                            k=0)  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                X = np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            if(self.nrow() != self.ncol()):
                raise ValueError(about._errors.cstring(
                    "ERROR: identity ill-defined for " + str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
            self.uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = np.diag(self.domain.calc_weight(X, power=-1).astype(max(min(X.dtype,
                                                                            self.domain.dtype), min(X.dtype, self.target.dtype))).flatten(order='C'), k=0)
        elif(np.size(X) == np.size(self.val)):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))

        # add
        if(X.dtype > self.val.dtype):
            about.warnings.cprint("WARNING: dtype reset.")
        self.val += X

        # check flags
        self.sym, self.uni = self._check_flags(sym=self.sym, uni=self.uni)

        self._inv = None  # reset

        return self

    def __sub__(self, X):  # __sub__ : self - X
        if(isinstance(X, operator)):
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X, explicit_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = self.val - X.val
            elif(isinstance(X, diagonal_operator)):
                if(self.target.dim(split=False) != X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: incompatible codomains."))
                # domain == X.domain
                matrix = self.val - \
                    np.diag(X.diag(bare=True, domain=None), k=0)
            elif(isinstance(X, vecvec_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = self.val - np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            if(self.nrow() != self.ncol()):
                raise ValueError(about._errors.cstring(
                    "ERROR: identity ill-defined for " + str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
            sym = self.sym
            uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = self.domain.calc_weight(X, power=-1).astype(
                max(min(X.dtype, self.domain.dtype), min(X.dtype, self.target.dtype)))
            matrix = self.val - np.diag(X.flatten(order='C'), k=0)
        elif(np.size(X) == np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
            matrix = self.val - X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))
        return explicit_operator(self.domain, matrix, bare=True, sym=sym, uni=uni, target=self.target)

    def __rsub__(self, X):  # __rsub__ : X - self
        if(isinstance(X, operator)):
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X, explicit_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = X.val - self.val
            elif(isinstance(X, diagonal_operator)):
                if(self.target.dim(split=False) != X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: incompatible codomains."))
                matrix = np.diag(X.diag(bare=True, domain=None),
                                 k=0) - self.val  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                matrix = np.tensordot(X.val, X.val, axes=0) - self.val
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            if(self.nrow() != self.ncol()):
                raise ValueError(about._errors.cstring(
                    "ERROR: identity ill-defined for " + str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
            sym = self.sym
            uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = self.domain.calc_weight(X, power=-1).astype(
                max(min(X.dtype, self.domain.dtype), min(X.dtype, self.target.dtype)))
            matrix = np.diag(X.flatten(order='C'), k=0) - self.val
        elif(np.size(X) == np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
            matrix = X - self.val
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))
        return explicit_operator(self.domain, matrix, bare=True, sym=sym, uni=uni, target=self.target)

    def __isub__(self, X):  # __isub__ : self -= X
        if(isinstance(X, operator)):
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
            self.sym = (self.sym and X.sym)
            self.uni = None
            if(isinstance(X, explicit_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                X = X.val
            elif(isinstance(X, diagonal_operator)):
                if(self.target.dim(split=False) != X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring(
                        "ERROR: incompatible codomains."))
                X = np.diag(X.diag(bare=True, domain=None),
                            k=0)  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                if(self.target != X.target):
                    raise ValueError(about._errors.cstring(
                        "ERROR: inequal codomains."))
                X = np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            if(self.nrow() != self.ncol()):
                raise ValueError(about._errors.cstring(
                    "ERROR: identity ill-defined for " + str(self.nrow()) + " x " + str(self.ncol()) + " matrices."))
            self.uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = np.diag(self.domain.calc_weight(X, power=-1).astype(max(min(X.dtype,
                                                                            self.domain.dtype), min(X.dtype, self.target.dtype))).flatten(order='C'), k=0)
        elif(np.size(X) == np.size(self.val)):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.ncol()) + " )."))

        # subtract
        if(X.dtype > self.val.dtype):
            about.warnings.cprint("WARNING: dtype reset.")
        self.val -= X

        # check flags
        self.sym, self.uni = self._check_flags(sym=self.sym, uni=self.uni)

        self._inv = None  # reset

        return self

    def _calc_mul(self, X, side):  # > multiplies self with X ...
        if(side == 0):  # ... from right
            if(self._hidden[0]):  # hidden degrees of freedom
                return np.array([np.apply_along_axis(self.domain.calc_dot, 0, X, np.conjugate(rr)) for rr in self.weight(rowpower=1, colpower=0, overwrite=False)])
            else:
                return np.dot(self.weight(rowpower=1, colpower=0, overwrite=False), X, out=None)
        elif(side == 1):  # ... from left
            if(self._hidden[1]):  # hidden degrees of freedom
                return np.array([np.apply_along_axis(self.target.calc_dot, 0, self.weight(rowpower=0, colpower=1, overwrite=False), np.conjugate(rr)) for rr in X])
            else:
                return np.dot(X, self.weight(rowpower=0, colpower=1, overwrite=False), out=None)
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input."))

    def __mul__(self, X):  # __mul__ : self * X
        if(isinstance(X, operator)):
            if(self.domain != X.target):
                raise ValueError(about._errors.cstring(
                    "ERROR: incompatible spaces."))
            newdomain = X.domain
            sym = None
            uni = (self.uni and X.uni)
            if(isinstance(X, explicit_operator)):
                X = X.val
            elif(isinstance(X, diagonal_operator)):
                X = np.diag(X.diag(bare=True, domain=None),
                            k=0)  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                X = np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            newdomain = self.domain
            sym = None
            uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = np.diag(self.domain.calc_weight(X, power=-1).astype(max(min(X.dtype,
                                                                            self.domain.dtype), min(X.dtype, self.target.dtype))).flatten(order='C'), k=0)
        elif(np.size(X) == self.val.shape[1]**2):
            newdomain = self.domain
            sym = None
            uni = None
            X = np.array(X).reshape((self.val.shape[1], self.val.shape[1]))
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.nrow()) + " )."))
        return explicit_operator(newdomain, self._calc_mul(X, 0), bare=True, sym=sym, uni=uni, target=self.target)

    def __rmul__(self, X):  # __mul__ : X * self
        if(isinstance(X, operator)):
            if(X.domain != self.target):
                raise ValueError(about._errors.cstring(
                    "ERROR: incompatible spaces."))
            newtarget = X.target
            sym = None
            uni = (self.uni and X.uni)
            if(isinstance(X, explicit_operator)):
                X = X.val
            elif(isinstance(X, diagonal_operator)):
                X = np.diag(X.diag(bare=True, domain=None),
                            k=0)  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                X = np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            newtarget = self.target
            sym = None
            uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = np.diag(self.domain.calc_weight(X, power=-1).astype(max(min(X.dtype,
                                                                            self.domain.dtype), min(X.dtype, self.target.dtype))).flatten(order='C'), k=0)
        elif(np.size(X) == self.val.shape[0]**2):
            newtarget = self.target
            sym = None
            uni = None
            X = np.array(X).reshape((self.val.shape[0], self.val.shape[0]))
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.ncol()) + " x " + str(self.ncol()) + " )."))
        return explicit_operator(self.domain, self._calc_mul(X, 1), bare=True, sym=sym, uni=uni, target=newtarget)

    def __imul__(self, X):  # __imul__ : self *= X
        if(isinstance(X, operator)):
            if(self.domain != X.target):
                raise ValueError(about._errors.cstring(
                    "ERROR: incompatible spaces."))
            if(self.domain != X.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: incompatible operator."))
            self.sym = None
            self.uni = (self.uni and X.uni)
            if(isinstance(X, explicit_operator)):
                X = X.val
            elif(isinstance(X, diagonal_operator)):
                X = np.diag(X.diag(bare=True, domain=None),
                            k=0)  # domain == X.domain
            elif(isinstance(X, vecvec_operator)):
                X = np.tensordot(X.val, X.val, axes=0)
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: unsupported or incompatible operator."))
        elif(np.size(X) == 1):
            self.sym = None
            self.uni = None
            X = X * np.ones(self.domain.dim(split=False),
                            dtype=np.int, order='C')
            X = np.diag(self.domain.calc_weight(X, power=-1).astype(max(min(X.dtype,
                                                                            self.domain.dtype), min(X.dtype, self.target.dtype))).flatten(order='C'), k=0)
        elif(np.size(X) == self.val.shape[1]**2):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape((self.val.shape[1], self.val.shape[1]))
            if(np.all(np.isreal(X))):
                X = np.real(X)
            X = X.astype(max(min(X.dtype, self.domain.dtype),
                             min(X.dtype, self.target.dtype)))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( " + str(
                np.size(X)) + " <> " + str(self.nrow()) + " x " + str(self.nrow()) + " )."))

        # multiply
        if(X.dtype > self.val.dtype):
            about.warnings.cprint("WARNING: dtype reset.")
        self.val = self._calc_mul(X, 0)

        # check flags
        self.sym, self.uni = self._check_flags(sym=self.sym, uni=self.uni)

        self._inv = None  # reset

        return self

    def __div__(self, X):
        raise Exception(about._errors.cstring(
            "ERROR: matrix division ill-defined."))

    __rdiv__ = __div__
    __idiv__ = __div__
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __itruediv__ = __idiv__

    def __pow__(self, x):  # __pow__(): self ** x
        if(not isinstance(x, (int, long))):
            raise TypeError(about._errors.cstring(
                "ERROR: non-integer exponent."))
        elif(self.domain <> self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: incompatible spaces."))
        elif(x == 0):
            return identity(self.domain)
        elif(x < 0):
            return self.inverse().__pow__(-x)
        elif(x == 1):
            return self
        else:
            matrix = self._calc_mul(self.val, 0)
            for ii in xrange(x - 1):
                matrix = self._calc_mul(matrix, 0)
            return explicit_operator(self.domain, matrix, bare=True, sym=None, uni=self.uni, target=self.target)

    def __rpow__(self, X):  # __pow__(): X ** self
        raise Exception(about._errors.cstring(
            "ERROR: matrix exponential ill-defined."))

    def __ipow__(self, x):  # __pow__(): self **= x
        if(not isinstance(x, (int, long))):
            raise TypeError(about._errors.cstring(
                "ERROR: non-integer exponent."))
        elif(self.domain <> self.target):
            raise ValueError(about._errors.cstring(
                "ERROR: incompatible spaces."))
        elif(x == 0):
            self.val = np.diag(self.domain.calc_weight(np.ones(self.domain.dim(
                split=False), dtype=np.int, order='C'), power=-1).astype(self.val.dtype), k=0)
        elif(x < 0):
            self.val = (self.inverse().__pow__(-x)).val
        elif(x == 1):
            pass
        else:
            matrix = self._calc_mul(self.val, 0)
            for ii in xrange(x - 1):
                matrix = self._calc_mul(matrix, 0)
            self.val = matrix

        # check flags
        self.sym, self.uni = self._check_flags(sym=None, uni=self.uni)

        self._inv = None  # reset

        return self

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def get_plot(X, title="", vmin=None, vmax=None, unit="", norm=None, cmap=None, cbar=True, **kwargs):
        """
            Creates a plot of the matrix according to the given specifications.

            Parameters
            ----------
            X : numpy.ndarray
                Array containing the matrix.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).

            Notes
            -----
            This method is static and thus also plot a valid numpy.ndarray.

            See Also
            --------
            explicit_operator.plot

            Raises
            ------
            ValueError
                If the matrix `X` is not two-dimensional;
                or if the logarithmic normalisation encounters negative values.

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(len(X.shape) != 2):
            raise ValueError(about._errors.cstring("ERROR: invalid matirx."))

        if(vmin is None):
            vmin = np.min(X, axis=None, out=None)
        if(vmax is None):
            vmax = np.max(X, axis=None, out=None)
        if(norm == "log")and(vmin <= 0):
            raise ValueError(about._errors.cstring(
                "ERROR: nonpositive value(s)."))

        s_ = np.array([X.shape[1] / max(X.shape), X.shape[0] /
                       max(X.shape) * (1.0 + 0.159 * bool(cbar))])
        fig = pl.figure(num=None, figsize=(
            6.4 * s_[0], 6.4 * s_[1]), dpi=None, facecolor=None, edgecolor=None, frameon=False, FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.06 / s_[0], 0.06 / s_[1],
                            1.0 - 0.12 / s_[0], 1.0 - 0.12 / s_[1]])
        if(norm == "log"):
            n_ = ln(vmin=vmin, vmax=vmax)
        else:
            n_ = None
        sub = ax0.pcolormesh(X[::-1, :], cmap=cmap,
                             norm=n_, vmin=vmin, vmax=vmax)
        ax0.set_xlim(0, X.shape[1])
        ax0.set_xticks([], minor=False)
        ax0.set_ylim(0, X.shape[0])
        ax0.set_yticks([], minor=False)
        ax0.set_aspect("equal")
        if(cbar):
            if(norm == "log"):
                f_ = lf(10, labelOnlyBase=False)
                b_ = sub.norm.inverse(np.linspace(0, 1, sub.cmap.N + 1))
                v_ = np.linspace(sub.norm.vmin, sub.norm.vmax, sub.cmap.N)
            else:
                f_ = None
                b_ = None
                v_ = None
            cb0 = fig.colorbar(sub, ax=ax0, orientation="horizontal", fraction=0.1, pad=0.05, shrink=0.75, aspect=20, ticks=[
                               vmin, vmax], format=f_, drawedges=False, boundaries=b_, values=v_)
            cb0.ax.text(0.5, -1.0, unit, fontdict=None, withdash=False, transform=cb0.ax.transAxes,
                        horizontalalignment="center", verticalalignment="center")
        ax0.set_title(title)

        if(bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor=None, edgecolor=None, orientation='portrait',
                        papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    def plot(self, bare=True, **kwargs):
        """
            Creates a plot of the matrix according to the given specifications.

            Parameters
            ----------
            bare : {bool, 2-tuple}, *optional*
                Whether the returned matrix entries are `bare` or not
                (default: True).

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).

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

            See Also
            --------
            explicit_operator.get_plot

        """
        interactive = pl.isinteractive()
        pl.matplotlib.interactive(not bool(kwargs.get("save", False)))

        X = self.get_matrix(bare=bare)

        if(np.any(np.iscomplex(X))):
            about.infos.cprint("INFO: absolute values and phases are plotted.")
            if(kwargs.has_key("title")):
                title = kwargs.get("title") + " "
                kwargs.__delitem__("title")
            else:
                title = ""
            self.get_plot(np.absolute(X), title=title + "(absolute)", **kwargs)
            if(kwargs.has_key("vmin")):
                kwargs.__delitem__("vmin")
            if(kwargs.has_key("vmin")):
                kwargs.__delitem__("vmax")
            if(kwargs.has_key("unit")):
                kwargs["unit"] = "rad"
            if(kwargs.has_key("norm")):
                kwargs["norm"] = None
            if(not kwargs.has_key("cmap")):
                kwargs["cmap"] = pl.cm.hsv_r
            self.get_plot(np.angle(X, deg=False), title=title + "(phase)",
                          vmin=-3.1416, vmax=3.1416, **kwargs)  # values in [-pi,pi]
        else:
            self.get_plot(np.real(X), **kwargs)

        pl.matplotlib.interactive(interactive)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_explicit.explicit_operator>"

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

class _share(object):

    __init__ = None

    @staticmethod
    def _init_share(_mat, _num):
        _share.mat = _mat
        _share.num = _num

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


class explicit_probing(probing):
    """
        ..
        ..
        ..                                    __     __             __   __
        ..                                  /  /   /__/           /__/ /  /_
        ..    _______  __   __    ______   /  /    __   _______   __  /   _/
        ..  /   __  / \  \/  /  /   _   | /  /   /  / /   ____/ /  / /  /
        .. /  /____/  /     /  /  /_/  / /  /_  /  / /  /____  /  / /  /_
        .. \______/  /__/\__\ /   ____/  \___/ /__/  \______/ /__/  \___/  probing class
        ..                   /__/


        NIFTY subclass for explicit probing (using multiprocessing)

        Called after initialization, an explicit matrix representation of a
        linear operator or function is sampled by applying (weighted) canonical
        vectors in a specified basis.

        Parameters
        ----------
        op : operator, *optional*
            Operator to be probed; if not given, the probing will resort to
            `function` (default: None).
        function : function, *optional*
            Function applied to the probes; either `op` or `function` must
            be given (default: `op.times`).
        domain : space, *optional*
            Space wherein the probes live, defines the domain of the
            explicified operator (default: `op.domain`).
        codomain : space, *optional*
            Space wherein the output of the explicified operator lives
            (default: `op.target`).
        target : space, *optional*
            Space wherein the transform of the probes live (default: None).
        ncpu : integer, *optional*
            Number of used CPUs to use (default: 2).
        nper : integer, *optional*
            Number of tasks performed by one worker (default: 1).

        See Also
        --------
        probing

        Examples
        --------
        >>> v = field(point_space(3), val=[1, 2, 3])
        >>> W = vecvec_operator(val=v)              # implicit operator
        >>> W_ij = explicit_probing(Wim)(loop=True) # explicit operator
        >>> W_ij.get_matrix()
        array([[ 1.,  2.,  3.],
               [ 2.,  4.,  6.],
               [ 3.,  6.,  9.]])

        Attributes
        ----------
        function : function
            Function applied to the probes.
        domain : space, *optional*
            Space wherein the probes live, defines the domain of the
            explicified operator.
        codomain : space, *optional*
            Space wherein the output of the explicified operator lives.
        target : space
            Space wherein the transform of the probes live.
        ncpu : integer
            Number of used CPUs to use.
        nper : integer
            Number of tasks performed by one worker.
        quargs : dict
            Keyword arguments passed to `function` in each call.

    """

    def __init__(self, op=None, function=None, domain=None, codomain=None, target=None, ncpu=2, nper=1, **quargs):
        """
            Initializes the explicit probing and sets the standard probing
            attributes except for `random`, `nrun`, and `var`.

            Parameters
            ----------
            op : operator, *optional*
                Operator to be probed; if not given, the probing will resort to
                `function` (default: None).
            function : function, *optional*
                Function applied to the probes; either `op` or `function` must
                be given (default: `op.times`).
            domain : space, *optional*
                Space wherein the probes live, defines the domain of the
                explicified operator (default: `op.domain`).
            codomain : space, *optional*
                Space wherein the output of the explicified operator lives
                (default: `op.target`).
            target : space, *optional*
                Space wherein the transform of the probes live (default: None).
            ncpu : integer, *optional*
                Number of used CPUs to use (default: 2).
            nper : integer, *optional*
                Number of tasks performed by one worker (default: 1).

            Other Parameters
            ----------------
            quargs : dict
                Keyword arguments passed to `function` in each call.

            Raises
            ------
            TypeError
                If input is invalid or insufficient.
            NameError
                If `function` is not an attribute of `op`.
            ValueError
                If spaces are incompatible.

        """
        if(op is None):
            # check whether callable
            if(function is None)or(not hasattr(function, "__call__")):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            # check given domain
            if(domain is None):
                raise TypeError(about._errors.cstring(
                    "ERROR: insufficient input."))
            elif(not isinstance(domain, space)):
                raise TypeError(about._errors.cstring(
                    "ERROR: invalid input domain."))
            # check given codomain
            if(codomain is None):
                raise TypeError(about._errors.cstring(
                    "ERROR: insufficient input."))
            elif(not isinstance(codomain, space)):
                raise TypeError(about._errors.cstring(
                    "ERROR: invalid input codomain."))
        else:
            if(not isinstance(op, operator)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            # check whether callable
            if(function is None)or(not hasattr(function, "__call__")):
                function = op.times
            elif(op == function):
                function = op.times
            # check whether correctly bound
            if(op != function.im_self):
                raise NameError(about._errors.cstring(
                    "ERROR: invalid input function."))
            # check given domain
            if(domain is None)or(not isinstance(domain, space)):
                if(function in [op.inverse_times, op.adjoint_times]):
                    domain = op.target
                else:
                    domain = op.domain
            else:
                if(function in [op.inverse_times, op.adjoint_times]):
                    op.target.check_codomain(domain)  # a bit pointless
                else:
                    op.domain.check_codomain(domain)  # a bit pointless
            # check given codomain
            if(codomain is None)or(not isinstance(codomain, space)):
                if(function in [op.inverse_times, op.adjoint_times]):
                    codomain = op.domain
                else:
                    codomain = op.target
            else:
                if(function in [op.inverse_times, op.adjoint_times]):
                    if(not op.domain.check_codomain(domain))and(op.domain.dim(split=False) != codomain.dim(split=False)):
                        raise ValueError(about._errors.cstring(
                            "ERROR: incompatible input codomain."))
                else:
                    if(not op.target.check_codomain(domain))and(op.target.dim(split=False) != codomain.dim(split=False)):
                        raise ValueError(about._errors.cstring(
                            "ERROR: incompatible input codomain."))

        self.function = function
        self.domain = domain
        self.codomain = codomain

        # check target
        if(target is None):
            target = self.domain.get_codomain()
        else:
            self.domain.check_codomain(target)  # a bit pointless
        self.target = target

        # check shape
        if(self.domain.dim(split=False) * self.codomain.dim(split=False) > 1048576):
            about.warnings.cprint("WARNING: matrix size > 2 ** 20.")

        self.ncpu = int(max(1, ncpu))
        self.nrun = self.domain.dim(split=False)
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1, min(self.nrun // self.ncpu, nper)))

        self.quargs = quargs

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure(self, **kwargs):
        """
            Changes the performance attributes of the instance.

            Parameters
            ----------
            ncpu : integer, *optional*
                Number of used CPUs to use.
            nper : integer, *optional*
                Number of tasks performed by one worker.

        """
        if("ncpu" in kwargs):
            self.ncpu = int(max(1, kwargs.get("ncpu")))
        if("nper" in kwargs):
            if(kwargs.get("nper") is None):
                self.nper = None
            else:
                self.nper = int(
                    max(1, min(self.nrun // self.ncpu, kwargs.get("nper"))))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def gen_probe(self, index, value):
        """
            Generates a probe.

            For explicit probing, each probe is a (weighted) canonical base.

            Parameters
            ----------
            index : int
                Index where to put the value.
            value : scalar
                Weighted 1.

            Returns
            -------
            probe : field
                Weighted canonical base.

        """
        probe = field(self.domain, val=None, target=self.target)
        probe[np.unravel_index(index, self.domain.dim(
            split=True), order='C')] = value
        return probe

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self, idnum, probe):
        """
            Computes a single probing result meaning a single column of the
            linear operators matrix representation.

            Parameters
            ----------
            probe : field
                Weighted canonical base on which `function` will be applied.
            idnum : int
                Identification number; obsolete.

            Returns
            -------
            result : ndarray
                Result of function evaluation (equaling a column of the matrix).

        """
        f = self.function(probe, **self.quargs)  # field
        if(f is None):
            return None
        elif(isinstance(f, field)):
            if(f.domain != self.codomain):
                try:
                    f.transform(target=self.codomain, overwrite=True)
                except(ValueError):  # unsupported transformation
                    # checkless version of
                    # f.cast_domain(self.codomain,newtarget=None,force=True)
                    pass
            return f.val.flatten(order='C')
        else:
            return f.flatten(order='C')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def evaluate(self, matrix, num):
        """
            Evaluates the probing results.

            Parameters
            ----------
            matrix : numpy.array
                Matrix representation of the probed linear operator.
            num : int
                Number of successful probings (not returning ``None``).

            Returns
            -------
            result : explicit_operator
                The probed linear operator as explicit operator instance.

        """
        if(num < self.nrun):
            about.infos.cflush(" ( %u probe(s) failed, effectiveness == %.1f%% )\n" % (
                self.nrun - num, 100 * num / self.nrun))
            if(num == 0):
                about.warnings.cprint("WARNING: probing failed.")
                return None
        else:
            about.infos.cflush("\n")

        return explicit_operator(self.domain, matrix, bare=True, sym=None, uni=None, target=self.codomain)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _single_probing(self, zipped):  # > performs one probing operation
        # generate probe
        probe = self.gen_probe(*zipped)
        # do the actual probing
        return self.probing(zipped[0], probe)

    def _serial_probing(self, zipped):  # > performs the probing operation serially
        try:
            result = self._single_probing(zipped)
        except Exception as exception:
            raise exception
        except BaseException:  # capture system-exiting exception including KeyboardInterrupt
            raise Exception(about._errors.cstring("ERROR: unknown."))
        else:
            if(result is not None):
                result = np.array(result).flatten(order='C')
                rindex = zipped[0] * self.codomain.dim(split=False)
                if(isinstance(_share.mat, tuple)):
                    _share.mat[0].acquire(block=True, timeout=None)
                    _share.mat[0][rindex:rindex +
                                  self.codomain.dim(split=False)] = np.real(result)
                    _share.mat[0].release()
                    _share.mat[1].acquire(block=True, timeout=None)
                    _share.mat[1][rindex:rindex +
                                  self.codomain.dim(split=False)] = np.imag(result)
                    _share.mat[1].release()
                else:
                    _share.mat.acquire(block=True, timeout=None)
                    _share.mat[rindex:rindex +
                               self.codomain.dim(split=False)] = result
                    _share.mat.release()
                _share.num.acquire(block=True, timeout=None)
                _share.num.value += 1
                _share.num.release()
                self._progress(_share.num.value)

    def _parallel_probing(self):  # > performs the probing operations in parallel
        # define weighted canonical base
        base = self.domain.calc_weight(self.domain.enforce_values(
            1, extend=True), power=-1).flatten(order='C')
        # define shared objects
        if(issubclass(self.codomain.dtype, np.complexfloating)):
            _mat = (ma('d', np.empty(self.nrun * self.codomain.dim(split=False), dtype=np.float64, order='C'), lock=True), ma('d',
                                                                                                                              np.empty(self.nrun * self.codomain.dim(split=False), dtype=np.float64, order='C'), lock=True))  # tuple(real,imag)
        else:
            _mat = ma('d', np.empty(self.nrun * self.codomain.dim(split=False),
                                    dtype=np.float64, order='C'), lock=True)
        _num = mv('I', 0, lock=True)
        # build pool
        if(about.infos.status):
            so.write(about.infos.cstring(
                "INFO: multiprocessing " + (' ') * 10))
            so.flush()
        pool = mp(processes=self.ncpu, initializer=_share._init_share,
                  initargs=(_mat, _num), maxtasksperchild=self.nper)
        try:
            # pooling
            pool.map_async(self._serial_probing, zip(np.arange(
                self.nrun, dtype=np.int), base), chunksize=None, callback=None).get(timeout=None)
            # close and join pool
            about.infos.cflush(" done.")
            pool.close()
            pool.join()
        except BaseException as exception:
            # terminate and join pool
            about._errors.cprint("\nERROR: terminating pool.")
            pool.terminate()
            pool.join()
            # re-raise exception
            raise exception  # traceback by looping
        # evaluate
        if(issubclass(self.codomain.dtype, np.complexfloating)):
            _mat = (np.array(_mat[0][:]) +
                    np.array(_mat[1][:]) * 1j)  # comlpex array
        else:
            _mat = np.array(_mat[:])
        _mat = _mat.reshape((self.nrun, self.codomain.dim(split=False))).T
        return self.evaluate(_mat, _num.value)

    # > performs the probing operations one after another
    def _nonparallel_probing(self):
        # define weighted canonical base
        base = self.domain.calc_weight(self.domain.enforce_values(
            1, extend=True), power=-1).flatten(order='C')
        # looping
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: looping " + (' ') * 10))
            so.flush()
        _mat = np.empty((self.nrun, self.codomain.dim(split=False)),
                        dtype=self.codomain.dtype, order='C')
        _num = 0
        for ii in xrange(self.nrun):
            result = self._single_probing((ii, base[ii]))  # zipped tuple
            if(result is None):
                _mat[ii] = np.zeros(self.codomain.dim(
                    split=False), dtype=self.codomain.dtype)
            else:
                _mat[ii] = np.array(result, dtype=self.codomain.dtype)
                _num += 1
                self._progress(_num)
        about.infos.cflush(" done.")
        # evaluate
        return self.evaluate(_mat.T, _num)

    def __call__(self, loop=False, **kwargs):
        """
            Start the explicit probing procedure.

            Parameters
            ----------
            loop : bool, *optional*
                Whether to perform a loop or to parallelise (default: False).

            Returns
            -------
            result : explicit_operator
                The probed linear operator as explicit operator instance.

            Other Parameters
            ----------------
            ncpu : integer, *optional*
                Number of used CPUs to use.
            nper : integer, *optional*
                Number of tasks performed by one worker.



        """
        self.configure(**kwargs)
        if(not about.multiprocessing.status)or(loop):
            return self._nonparallel_probing()
        else:
            return self._parallel_probing()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_explicit.explicit_probing>"

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


def explicify(op, newdomain=None, newtarget=None, ncpu=2, nper=1, loop=False, **quargs):
    """
        Explicifys an (implicit) operator.

        This function wraps the :py:class:`explicit_probing` class with a more
        user-friendly interface.

        Parameters
        ----------
        op : operator
            Operator to be explicified.
        newdomain : space, *optional*
            Space wherein the probes live, defines the domain of the
            explicified operator (default: `op.domain`).
        newtarget : space, *optional*
            Space wherein the output of the explicified operator lives
            (default: `op.target`).
        ncpu : integer, *optional*
            Number of used CPUs to use (default: 2).
        nper : integer, *optional*
            Number of tasks performed by one worker (default: 1).

        Returns
        -------
        EO : explicit_operator
            The explicified linear operator as explicit operator instance.

        Other Parameters
        ----------------
        quargs : dict
            Keyword arguments passed to `function` in each call.

        See Also
        --------
        explicit_probing

        Examples
        --------
        >>> x_space = rg_space(4)
        >>> k_space = x_space.get_codomain()
        >>> S = power_operator(k_space, spec=[9, 3, 1])
        >>> S.diag()          # implicit operator in k_space
        array([ 1.,  3.,  9.,  3.])
        >>> S_kq = explicify(S)
        >>> S_kq.get_matrix() # explicit operator in k_space
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  3.,  0.,  0.],
               [ 0.,  0.,  9.,  0.],
               [ 0.,  0.,  0.,  3.]])
        >>> S_xy = explicify(S, newdomain=x_space)
        >>> S_xy.get_matrix() # explicit operator in x_space
        array([[ 16.,   8.,   4.,   8.],
               [  8.,  16.,   8.,   4.],
               [  4.,   8.,  16.,   8.],
               [  8.,   4.,   8.,  16.]])

        Raises
        ------
        TypeError
            If `op` is no operator instance.

    """
    if(not isinstance(op, operator)):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))
    elif(newdomain is not None)and(newtarget is None)and(op.domain == op.target):
        newtarget = newdomain
    if(newdomain is None)or(newdomain == op.domain):
        target_ = None
    else:
        target_ = op.domain
    return explicit_probing(op=op, function=op.times, domain=newdomain, codomain=newtarget, target=target_, ncpu=ncpu, nper=nper, **quargs)(loop=loop)

# -----------------------------------------------------------------------------
