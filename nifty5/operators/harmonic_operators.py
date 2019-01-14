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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .. import dobj, utilities, fft
from ..domain_tuple import DomainTuple
from ..domains.gl_space import GLSpace
from ..domains.lm_space import LMSpace
from ..domains.rg_space import RGSpace
from ..field import Field
from .diagonal_operator import DiagonalOperator
from .linear_operator import LinearOperator
from .scaling_operator import ScalingOperator


class FFTOperator(LinearOperator):
    """Transforms between a pair of position and harmonic RGSpaces.

    Parameters
    ----------
    domain: Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target: Domain, optional
        The target (sub-)domain of the transform operation.
        If omitted, a domain will be chosen automatically.
    space: int, optional
        The index of the subdomain on which the operator should act
        If None, it is set to 0 if `domain` contains exactly one space.
        `domain[space]` must be an RGSpace.

    Notes
    -----
    This operator performs full FFTs, which implies that its output field will
    always have complex type, regardless of the type of the input field.
    If a real field is desired after a forward/backward transform couple, it
    must be manually cast to real.
    """

    def __init__(self, domain, target=None, space=None):
        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._capability = self._all_ops
        self._space = utilities.infer_space(self._domain, space)

        adom = self._domain[self._space]
        if not isinstance(adom, RGSpace):
            raise TypeError("FFTOperator only works on RGSpaces")
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

    def apply(self, x, mode):
        self._check_input(x, mode)
        ncells = x.domain[self._space].size
        if x.domain[self._space].harmonic:  # harmonic -> position
            func = fft.fftn
            fct = 1.
        else:
            func = fft.ifftn
            fct = ncells
        axes = x.domain.axes[self._space]
        tdom = self._tgt(mode)
        oldax = dobj.distaxis(x.val)
        if oldax not in axes:  # straightforward, no redistribution needed
            ldat = x.local_data
            ldat = func(ldat, axes=axes)
            tmp = dobj.from_local_data(x.val.shape, ldat, distaxis=oldax)
        elif len(axes) < len(x.shape) or len(axes) == 1:
            # we can use one FFT pass in between the redistributions
            tmp = dobj.redistribute(x.val, nodist=axes)
            newax = dobj.distaxis(tmp)
            ldat = dobj.local_data(tmp)
            ldat = func(ldat, axes=axes)
            tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
            tmp = dobj.redistribute(tmp, dist=oldax)
        else:  # two separate FFTs needed
            rem_axes = tuple(i for i in axes if i != oldax)
            tmp = x.val
            ldat = dobj.local_data(tmp)
            ldat = func(ldat, axes=rem_axes)
            if oldax != 0:
                raise ValueError("bad distribution")
            ldat2 = ldat.reshape((ldat.shape[0],
                                  np.prod(ldat.shape[1:])))
            shp2d = (x.val.shape[0], np.prod(x.val.shape[1:]))
            tmp = dobj.from_local_data(shp2d, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp)
            ldat2 = func(ldat2, axes=(1,))
            tmp = dobj.from_local_data(tmp.shape, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp).reshape(ldat.shape)
            tmp = dobj.from_local_data(x.val.shape, ldat2, distaxis=0)
        Tval = Field(tdom, tmp)
        if mode & (LinearOperator.TIMES | LinearOperator.ADJOINT_TIMES):
            fct *= self._domain[self._space].scalar_dvol
        else:
            fct *= self._target[self._space].scalar_dvol
        return Tval if fct == 1 else Tval*fct


class HartleyOperator(LinearOperator):
    """Transforms between a pair of position and harmonic RGSpaces.

    Parameters
    ----------
    domain: Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target: Domain, optional
        The target (sub-)domain of the transform operation.
        If omitted, a domain will be chosen automatically.
    space: int, optional
        The index of the subdomain on which the operator should act
        If None, it is set to 0 if `domain` contains exactly one space.
        `domain[space]` must be an RGSpace.

    Notes
    -----
    This operator always produces output fields with the same data type as
    its input. This is achieved by performing so-called Hartley transforms
    (https://en.wikipedia.org/wiki/Discrete_Hartley_transform).
    For complex input fields, the operator will transform the real and
    imaginary parts separately and use the results as real and imaginary parts
    of the result field, respectivey.
    In many contexts the Hartley transform is a perfect substitute for the
    Fourier transform, but in some situations (e.g. convolution with a general,
    non-symmetric kernel, the full FFT must be used instead.
    """

    def __init__(self, domain, target=None, space=None):
        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._capability = self._all_ops
        self._space = utilities.infer_space(self._domain, space)

        adom = self._domain[self._space]
        if not isinstance(adom, RGSpace):
            raise TypeError("HartleyOperator only works on RGSpaces")
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if utilities.iscomplextype(x.dtype):
            return (self._apply_cartesian(x.real, mode) +
                    1j*self._apply_cartesian(x.imag, mode))
        else:
            return self._apply_cartesian(x, mode)

    def _apply_cartesian(self, x, mode):
        axes = x.domain.axes[self._space]
        tdom = self._tgt(mode)
        oldax = dobj.distaxis(x.val)
        if oldax not in axes:  # straightforward, no redistribution needed
            ldat = x.local_data
            ldat = fft.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(x.val.shape, ldat, distaxis=oldax)
        elif len(axes) < len(x.shape) or len(axes) == 1:
            # we can use one Hartley pass in between the redistributions
            tmp = dobj.redistribute(x.val, nodist=axes)
            newax = dobj.distaxis(tmp)
            ldat = dobj.local_data(tmp)
            ldat = fft.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
            tmp = dobj.redistribute(tmp, dist=oldax)
        else:  # two separate, full FFTs needed
            # ideal strategy for the moment would be:
            # - do real-to-complex FFT on all local axes
            # - fill up array
            # - redistribute array
            # - do complex-to-complex FFT on remaining axis
            # - add re+im
            # - redistribute back
            rem_axes = tuple(i for i in axes if i != oldax)
            tmp = x.val
            ldat = dobj.local_data(tmp)
            ldat = fft.my_fftn_r2c(ldat, axes=rem_axes)
            if oldax != 0:
                raise ValueError("bad distribution")
            ldat2 = ldat.reshape((ldat.shape[0],
                                  np.prod(ldat.shape[1:])))
            shp2d = (x.val.shape[0], np.prod(x.val.shape[1:]))
            tmp = dobj.from_local_data(shp2d, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp)
            ldat2 = fft.my_fftn(ldat2, axes=(1,))
            ldat2 = ldat2.real+ldat2.imag
            tmp = dobj.from_local_data(tmp.shape, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp).reshape(ldat.shape)
            tmp = dobj.from_local_data(x.val.shape, ldat2, distaxis=0)
        Tval = Field(tdom, tmp)
        if mode & (LinearOperator.TIMES | LinearOperator.ADJOINT_TIMES):
            fct = self._domain[self._space].scalar_dvol
        else:
            fct = self._target[self._space].scalar_dvol
        return Tval if fct == 1 else Tval*fct


class SHTOperator(LinearOperator):
    """Transforms between a harmonic domain on the sphere and a position
    domain counterpart.

    Built-in domain pairs are
      - an LMSpace and a HPSpace
      - an LMSpace and a GLSpace

    The supported operations are times() and adjoint_times().

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target : Domain, optional
        The target domain of the transform operation.
        If omitted, a domain will be chosen automatically.
        Whenever the input domain of the transform is an RGSpace, the codomain
        (and its parameters) are uniquely determined.
        For LMSpace, a GLSpace of sufficient resolution is chosen.
    space : int, optional
        The index of the domain on which the operator should act
        If None, it is set to 0 if domain contains exactly one subdomain.
        domain[space] must be a LMSpace.
    """

    def __init__(self, domain, target=None, space=None):
        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = utilities.infer_space(self._domain, space)

        hspc = self._domain[self._space]
        if not isinstance(hspc, LMSpace):
            raise TypeError("SHTOperator only works on a LMSpace domain")
        if target is None:
            target = hspc.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        hspc.check_codomain(target)
        target.check_codomain(hspc)

        from pyHealpix import sharpjob_d
        self.lmax = hspc.lmax
        self.mmax = hspc.mmax
        self.sjob = sharpjob_d()
        self.sjob.set_triangular_alm_info(self.lmax, self.mmax)
        if isinstance(target, GLSpace):
            self.sjob.set_Gauss_geometry(target.nlat, target.nlon)
        else:
            self.sjob.set_Healpix_geometry(target.nside)

    def __reduce__(self):
        return (_unpickleSHTOperator,
                (self._domain, self._target[self._space], self._space))

    def apply(self, x, mode):
        self._check_input(x, mode)
        if utilities.iscomplextype(x.dtype):
            return (self._apply_spherical(x.real, mode) +
                    1j*self._apply_spherical(x.imag, mode))
        else:
            return self._apply_spherical(x, mode)

    def _slice_p2h(self, inp):
        rr = self.sjob.alm2map_adjoint(inp)
        if len(rr) != ((self.mmax+1)*(self.mmax+2))//2 + \
                      (self.mmax+1)*(self.lmax-self.mmax):
            raise ValueError("array length mismatch")
        res = np.empty(2*len(rr)-self.lmax-1, dtype=rr[0].real.dtype)
        res[0:self.lmax+1] = rr[0:self.lmax+1].real
        res[self.lmax+1::2] = np.sqrt(2)*rr[self.lmax+1:].real
        res[self.lmax+2::2] = np.sqrt(2)*rr[self.lmax+1:].imag
        return res/np.sqrt(np.pi*4)

    def _slice_h2p(self, inp):
        res = np.empty((len(inp)+self.lmax+1)//2, dtype=(inp[0]*1j).dtype)
        if len(res) != ((self.mmax+1)*(self.mmax+2))//2 + \
                       (self.mmax+1)*(self.lmax-self.mmax):
            raise ValueError("array length mismatch")
        res[0:self.lmax+1] = inp[0:self.lmax+1]
        res[self.lmax+1:] = np.sqrt(0.5)*(inp[self.lmax+1::2] +
                                          1j*inp[self.lmax+2::2])
        res = self.sjob.alm2map(res)
        return res/np.sqrt(np.pi*4)

    def _apply_spherical(self, x, mode):
        axes = x.domain.axes[self._space]
        axis = axes[0]
        v = x.val
        v, idat = dobj.ensure_not_distributed(v, (axis,))
        distaxis = dobj.distaxis(v)

        p2h = not x.domain[self._space].harmonic
        tdom = self._tgt(mode)
        func = self._slice_p2h if p2h else self._slice_h2p
        odat = np.empty(dobj.local_shape(tdom.shape, distaxis=distaxis),
                        dtype=x.dtype)
        for slice in utilities.get_slice_list(idat.shape, axes):
            odat[slice] = func(idat[slice])
        odat = dobj.from_local_data(tdom.shape, odat, distaxis)
        return Field(tdom, dobj.ensure_default_distributed(odat))


def _unpickleSHTOperator(*args):
    return SHTOperator(*args)


class HarmonicTransformOperator(LinearOperator):
    """Transforms between a harmonic domain and a position domain counterpart.

    Built-in domain pairs are
      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - an LMSpace and a HPSpace
      - an LMSpace and a GLSpace

    The supported operations are times() and adjoint_times().

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target : Domain, optional
        The target domain of the transform operation.
        If omitted, a domain will be chosen automatically.
        Whenever the input domain of the transform is an RGSpace, the codomain
        (and its parameters) are uniquely determined.
        For LMSpace, a GLSpace of sufficient resolution is chosen.
    space : int, optional
        The index of the domain on which the operator should act
        If None, it is set to 0 if domain contains exactly one subdomain.
        domain[space] must be a harmonic domain.
    """

    def __init__(self, domain, target=None, space=None):
        domain = DomainTuple.make(domain)
        space = utilities.infer_space(domain, space)

        hspc = domain[space]
        if not hspc.harmonic:
            raise TypeError(
                "HarmonicTransformOperator only works on a harmonic space")
        if isinstance(hspc, RGSpace):
            self._op = HartleyOperator(domain, target, space)
        else:
            self._op = SHTOperator(domain, target, space)
        self._domain = self._op.domain
        self._target = self._op.target
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._op.apply(x, mode)


def HarmonicSmoothingOperator(domain, sigma, space=None):
    """Returns an operator that carries out a smoothing with a Gaussian kernel
    of width `sigma` on the part of `domain` given by `space`

    Parameters
    ----------
    domain : Domain, tuple of Domain, or DomainTuple
       The total domain of the operator's input and output fields
    sigma : float>=0
       The sigma of the Gaussian used for smoothing. It has the same units as
       the RGSpace the operator is working on.
       If `sigma==0`, an identity operator will be returned.
    space : int, optional
       The index of the sub-domain on which the smoothing is performed.
       Can be omitted if `domain` only has one sub-domain.

    Notes
    -----
    The sub-domain on which the smoothing is carried out *must* be a
    non-harmonic `RGSpace`.
    """

    sigma = float(sigma)
    if sigma < 0.:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.:
        return ScalingOperator(1., domain)

    domain = DomainTuple.make(domain)
    space = utilities.infer_space(domain, space)
    if domain[space].harmonic:
        raise TypeError("domain must not be harmonic")
    Hartley = HartleyOperator(domain, space=space)
    codomain = Hartley.domain[space].get_default_codomain()
    kernel = codomain.get_k_length_array()
    smoother = codomain.get_fft_smoothing_kernel_function(sigma)
    kernel = smoother(kernel)
    ddom = list(domain)
    ddom[space] = codomain
    diag = DiagonalOperator(kernel, ddom, space)
    return Hartley.inverse(diag(Hartley))
