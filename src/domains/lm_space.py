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
#
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..field import Field
from .structured_domain import StructuredDomain


class LMSpace(StructuredDomain):
    """Represents a set of spherical harmonic coefficients.

    Its harmonic partner spaces are :class:`~nifty8.domains.hp_space.HPSpace`
    and :class:`~nifty8.domains.gl_space.GLSpace`.

    Parameters
    ----------
    lmax : int
        The maximum :math:`l` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        Must be :math:`\\ge 0`.

    mmax : int, optional
        The maximum :math:`m` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        If not supplied, it is set to `lmax`.
        Must be :math:`\\ge 0` and :math:`\\le` `lmax`.
    """

    _needed_for_hash = ["_lmax", "_mmax"]

    def __init__(self, lmax, mmax=None):
        self._lmax = int(lmax)
        if self._lmax < 0:
            raise ValueError("lmax must be >=0.")
        if mmax is None:
            mmax = self._lmax
        self._mmax = int(mmax)
        if self._mmax < 0 or self._mmax > self._lmax:
            raise ValueError("mmax must be >=0 and <=lmax.")

    def __repr__(self):
        return "LMSpace(lmax={}, mmax={})".format(self.lmax, self.mmax)

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        l, m = self._lmax, self._mmax
        # the LMSpace consists of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        return (l+1)**2 - (l-m)*(l-m+1)

    @property
    def scalar_dvol(self):
        return 1.

    def get_k_length_array(self):
        lmax = self._lmax
        mmax = self._mmax
        ldist = np.empty((self.size,), dtype=np.float64)
        ldist[0:lmax+1] = np.arange(lmax+1, dtype=np.float64)
        tmp = np.empty((2*lmax+2), dtype=np.float64)
        tmp[0::2] = np.arange(lmax+1)
        tmp[1::2] = np.arange(lmax+1)
        idx = lmax+1
        for m in range(1, mmax+1):
            ldist[idx:idx+2*(lmax+1-m)] = tmp[2*m:]
            idx += 2*(lmax+1-m)
        return Field.from_raw(self, ldist)

    def get_unique_k_lengths(self):
        return np.arange(self.lmax+1, dtype=np.float64)

    @staticmethod
    def _kernel(x, sigma):
        # cf. "All-sky convolution for polarimetry experiments"
        # by Challinor et al.
        # https://arxiv.org/abs/astro-ph/0008228
        from ..sugar import exp
        return exp((x+1.) * x * (-0.5*sigma*sigma))

    def get_fft_smoothing_kernel_function(self, sigma):
        return lambda x: self._kernel(x, sigma)

    def get_conv_kernel_from_func(self, func):
        """Creates a convolution kernel defined by a function.

        Assumes the function to be radially symmetric, e.g. only dependant on
        theta in radians.

        Parameters
        ----------
        func: function
            This function needs to take exactly one argument, which is
            colatitude in radians, and return the kernel amplitude at that
            colatitude.
        """
        from ducc0.misc import GL_thetas

        from ..operators.harmonic_operators import HarmonicTransformOperator
        from .gl_space import GLSpace

        # define azimuthally symmetric spaces for kernel transform
        gl = GLSpace(self.lmax + 1, 1)
        lm0 = gl.get_default_codomain()
        theta = GL_thetas(gl.nlat)
        # evaluate the kernel function at the required thetas
        kernel_sphere = Field.from_raw(gl, func(theta))
        # normalize the kernel such that the integral over the sphere is 4pi
        kernel_sphere = kernel_sphere * (4 * np.pi / kernel_sphere.s_integrate())
        # compute the spherical harmonic coefficients of the kernel
        op = HarmonicTransformOperator(lm0, gl)
        kernel_lm = op.adjoint_times(kernel_sphere.weight(1)).val
        # evaluate the k lengths of the harmonic space
        k_lengths = self.get_k_length_array().val.astype(np.int64)
        return Field.from_raw(self, kernel_lm[k_lengths])

    @property
    def lmax(self):
        """int : maximum allowed :math:`l`

        The maximum :math:`l` value of any spherical harmonic
        coefficient :math:`a_{lm}` that is represented in this domain.
        """
        return self._lmax

    @property
    def mmax(self):
        """int : maximum allowed :math:`m`

        The maximum :math:`m` value of any spherical harmonic
        coefficient :math:`a_{lm}` that is represented in this domain.
        """
        return self._mmax

    def get_default_codomain(self):
        """Returns a :class:`~nifty8.domains.gl_space.GLSpace` object, which is
        capable of storing an accurate representation of data residing on
        `self`.

        Returns
        -------
        GLSpace
            The partner domain
        """
        from ..domains.gl_space import GLSpace
        return GLSpace(self.lmax+1, self.mmax*2+1)

    def check_codomain(self, codomain):
        """Raises `TypeError` if `codomain` is not a matching partner domain
        for `self`.

        Notes
        -----
        This function only checks whether `codomain` is of type
        :class:`GLSpace` or :class:`HPSpace`.
        """
        from ..domains.gl_space import GLSpace
        from ..domains.hp_space import HPSpace
        if not isinstance(codomain, (GLSpace, HPSpace)):
            raise TypeError("codomain must be a GLSpace or HPSpace.")
