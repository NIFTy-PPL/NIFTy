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
# Copyright(C) 2019-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Copyright(C) 2025 Philipp Arras
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from scipy.constants import speed_of_light

from ..any_array import AnyArray
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..ducc_dispatch import nthreads
from ..field import Field
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.domain_tuple_field_inserter import DomainTupleFieldInserter
from ..operators.harmonic_operators import FFTShiftOperator
from ..operators.linear_operator import LinearOperator
from ..operators.operator import Operator, is_linearization
from ..operators.simple_linear_operators import (DomainChangerAndReshaper,
                                                 Variable)
from ..sugar import makeDomain, makeField


class Gridder(LinearOperator):
    """Compute non-uniform 2D FFT with ducc.

    Parameters
    ----------
    target : Domain, tuple of domains or DomainTuple.
        Target domain, must be a single two-dimensional RGSpace.
    uv : np.ndarray
        Coordinates of the data-points, shape (n, 2).
    eps : float
        Requested precision, defaults to 2e-10.

    Note
    ----
    If this operator is called on inputs stored on device, it will copy them to
    host, perform the operations with the CPU and copy back.
    """
    def __init__(self, target, uv, eps=2e-10):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = makeDomain(target)
        for ii in [0, 1]:
            if target.shape[ii] % 2 != 0:
                raise ValueError("even number of pixels is required for gridding operation")
        if (len(self._target) != 1 or not isinstance(self._target[0], RGSpace)
                or not len(self._target.shape) == 2):
            raise ValueError("need target with exactly one 2D RGSpace")
        if uv.ndim != 2:
            raise ValueError("uv must be a 2D array")
        if uv.shape[1] != 2:
            raise ValueError("second dimension of uv must have length 2")
        self._domain = DomainTuple.make(UnstructuredDomain((uv.shape[0])))
        # wasteful hack to adjust to shape required by ducc0.wgridder
        self._uvw = np.empty((uv.shape[0], 3), dtype=np.float64)
        self._uvw[:, 0:2] = uv
        self._uvw[:, 2] = 0.
        self._eps = float(eps)

    def apply(self, x, mode):
        from ducc0.wgridder import dirty2ms, ms2dirty
        self._check_input(x, mode)
        freq = np.array([speed_of_light])
        nxdirty, nydirty = self._target[0].shape
        dstx, dsty = self._target[0].distances
        if mode == self.TIMES:
            res = ms2dirty(self._uvw, freq, x.asnumpy().reshape((-1,1)), None, nxdirty,
                           nydirty, dstx, dsty, 0, 0,
                           self._eps, False, nthreads(), 0)
        else:
            res = dirty2ms(self._uvw, freq, x.asnumpy(), None, dstx, dsty, 0, 0,
                           self._eps, False, nthreads(), 0)
            res = res.reshape((-1,))
        return makeField(self._tgt(mode), res).at(x.device_id)


class Nufft(LinearOperator):
    """Compute non-uniform 1D, 2D and 3D FFTs.

    Parameters
    ----------
    target : Domain, tuple of domains or DomainTuple.
        Target domain, must be an RGSpace with one to three dimensions.
    pos : np.ndarray
        Coordinates of the data-points, shape (n, ndim).
    eps: float
        Requested precision, defaults to 2e-10.

    Note
    ----
    If this operator is called on inputs stored on device, it will copy them to
    host, perform the operations with the CPU and copy back.
    """
    def __init__(self, target, pos, eps=2e-10):
        try:
            from ducc0.nufft import nu2u, u2nu
        except ImportError:
            raise ImportError("ducc0 needs to be installed for nifty.cl.Nufft()")
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = makeDomain(target)
        if not isinstance(self._target[0], RGSpace):
            raise TypeError("target needs to be an RGSpace")
        if len(self._target.shape) > 3:
            raise ValueError("Only 1D, 2D and 3D FFTs are supported")
        if pos.ndim != 2:
            raise TypeError(f"pos needs to be 2d array (got shape {pos.shape})")
        self._domain = DomainTuple.make(UnstructuredDomain((pos.shape[0])))
        dst = np.array(self._target[0].distances)
        pos = (2*np.pi*pos*dst) % (2*np.pi)
        self._args = dict(nthreads=1, epsilon=float(eps), coord=pos)

    def apply(self, x, mode):
        from ducc0.nufft import nu2u, u2nu

        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.empty(self.target.shape, dtype=x.dtype)
            nu2u(points=x.asnumpy(), out=res, forward=False, **self._args)
            res = res.real
        else:
            res = u2nu(grid=x.asnumpy().astype('complex128'), forward=True, **self._args)
            #if res.ndim == 0:
                #res = np.array([res])
        return makeField(self._tgt(mode), res).at(x.device_id)


class VariablePositionNufft(Operator):
    """Type 2 NUFFT operator for variable non-uniform positions.

    This operator performs a type 2 Non-Uniform Fast Fourier Transform (NUFFT,
    uniform to non-uniform) for a signal sampled on a regular grid and evaluated
    at arbitrary spatial positions.

    The resulting domain of this operator is a MultiDomain with two subspaces:
      - "grid": `grid_domain`
      - "coord": `(UnstructuredDomain(npoints), UnstructuredDomain(ndim))` where
        `ndim` is the number of dimensions of `grid_domain`
    or if `pre_domain` is set:
      - "grid": `(pre_domain, grid_domain)`
      - "coord": `(UnstructuredDomain(npoints), UnstructuredDomain(ndim))`

    The target of this operator is a `DomainTuple`:
    `(UnstructuredDomain(npoints),)` or `(pre_domain,
    UnstructuredDomain(npoints))` if `pre_domain` is not None.

    Parameters
    ----------
    grid_domain : DomainTuple or RGSpace
        The regular grid domain over which the input signal is defined.
    npoints : int
        Number of non-uniform evaluation points.
    epsilon : float
        Precision parameter for the NUFFT computation. Smaller values lead to
        higher accuracy but slower performance.
    pre_domain : Arbitrary 1d space, optional
        If pre_domain is not None, it will be prepended to grid_domain. All
        transformations are performed for each element in pre_domain. Default is
        None.
    """
    def __init__(self, grid_domain, npoints, epsilon, pre_domain=None):
        grid_domain = DomainTuple.make(grid_domain)
        nufft_ndim = len(grid_domain.shape)
        assert len(grid_domain) == 1
        assert isinstance(grid_domain[0], RGSpace)
        assert 1 <= nufft_ndim <= 3
        if pre_domain is None:
            domain = {
                "grid": grid_domain,
                "coord": (UnstructuredDomain(npoints), UnstructuredDomain(nufft_ndim)),
            }
            target = UnstructuredDomain(npoints)
            self._ntrans = None
        else:
            pre_domain = DomainTuple.make(pre_domain)
            assert len(pre_domain) == 1
            assert len(pre_domain.shape) == 1
            domain = {
                "grid": (pre_domain[0], grid_domain[0]),
                "coord": (UnstructuredDomain(npoints), UnstructuredDomain(nufft_ndim)),
            }
            target = (pre_domain[0], UnstructuredDomain(npoints))
            self._ntrans = pre_domain.shape[0]
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._epsilon = float(epsilon)

    def apply(self, x):
        self._check_input(x)
        xval = x.val if is_linearization(x) else x
        plans = _NufftPlans(self._domain["grid"][-1], xval["coord"].val, self._epsilon, self._ntrans)
        val = plans.execute(xval["grid"].val, nufft_type=2, forward=True)
        val = Field(self._target, val)
        if not is_linearization(x):
            return val
        else:
            jac = _VariablePositionNufftJacobian(xval, self._target, plans,
                                                 val.device_id)
            return x.new(val, jac)


class _VariablePositionNufftJacobian(LinearOperator):
    def __init__(self, pos, tgt, plans, device_id):
        self._domain = pos.domain
        self._target = tgt
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._pos = pos
        assert self._pos.domain is self._domain
        if device_id == -1:
            args = [(-(ss//2) + np.arange(ss)) for ss in self._domain["grid"][-1].shape]
            self._xyz = np.meshgrid(*args, indexing='ij')
        else:
            import cupy as cp
            with cp.cuda.Device(device_id):
                args = [(-(ss//2) + cp.arange(ss)) for ss in self._domain["grid"][-1].shape]
                self._xyz = cp.meshgrid(*args, indexing='ij')
        self._xyz = tuple(map(AnyArray, self._xyz))
        assert self._xyz[0].device_id == device_id

        rgdom = self._domain["grid"][-1]
        self._grid0 = self._pos["grid"].val
        self._plans = plans
        self._dst = AnyArray(2*np.pi*np.array(rgdom.distances))

    def _device_preparation(self, x, mode):
        if mode == self.TIMES:
            device_id = x["grid"].device_id
            assert x["coord"].device_id == x["grid"].device_id
        else:
            device_id = x.device_id
        self._grid0 = self._grid0.at(device_id)
        self._xyz = tuple(map(lambda x: x.at(device_id), self._xyz))
        self._dst = self._dst.at(device_id)
        self._pos = self._pos.at(device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        self._device_preparation(x, mode)
        ndim = self._dst.shape[0]
        if mode == self.TIMES:
            grid = x["grid"].val
            coord = x["coord"].val*self._dst
            # Grid
            res = self._plans.execute(grid, nufft_type=2, forward=True)
            # Coord
            for idim in range(ndim):
                tmp = self._plans.execute(-1j*self._xyz[idim]*self._grid0,
                                          nufft_type=2, forward=True)
                res += tmp*coord[:, idim]
        else:
            points = x.val
            # Grid
            res_grid = self._plans.execute(points, nufft_type=1, forward=False)
            # Coord
            res_coord = np.empty_like(self._pos["coord"].val)
            gg = 1j*self._grid0.conj()
            ntrans = self._plans.ntrans
            if ntrans is None:
                gg = gg[None]
                points = points[None]
                ntrans = 1

            for idim in range(ndim):
                tmp = self._plans.execute(gg*self._xyz[idim][None],
                                          nufft_type=2, forward=False)
                res_coord[:, idim] = np.sum((tmp*points).real, axis=0)

            # Merge
            res = {"grid": res_grid, "coord": res_coord*self._dst}
        return makeField(self._tgt(mode), res)


def ShiftedPositionFFT(grid_domain, eps, pre_domain=None, shift_directions=None):
    """Type 2 NUFFT operator for shifted grid-like positions.

    Constructs a NUFFT operator that emulates an FFT on a regular grid, while
    allowing evaluation at positions offset from the standard sampling points.
    This enables perturbations of sampling coordinates, which can be interpreted
    as small displacements in either domain of a Fourier transform pair,
    depending on context.

    The resulting domain of this operator is a MultiDomain with two subspaces:
      - "grid": `grid_domain`
      - "delta_coord": `(grid_domain, UnstructuredDomain(ndim))` where `ndim` is
        the number of dimensions of `grid_domain`
    or if `pre_domain` is set:
      - "grid": `(pre_domain, grid_domain)`
      - "delta_coord": `(pre_domain, grid_domain, UnstructuredDomain(ndim))`

    The target of the operator is `(hgrid_domain,)` or if `pre_domain` is set
    `(pre_domain, hgrid_domain)`, where `hgrid_domain =
    grid_domain.get_default_codomain()`

    "delta_coord" specifies the relative shift of the sampling positions with
    respect to the regular FFT grid, expressed in units of Nyquist-scaled
    frequency spacing.

    A value of 0 means the Fourier sampling is performed exactly at the standard
    FFT grid points. A value of 1 shifts the sampling position by one full grid
    step in frequency space, effectively aligning with the next neighboring FFT
    frequency point. Fractional values enable interpolation between FFT
    frequencies.

    Parameters
    ----------
    grid_domain : RGSpace or DomainTuple
        The input domain of the signal.
    eps : float
        Precision parameter for the NUFFT computation. Smaller values lead to
        higher accuracy but slower performance.
    pre_domain : Arbitrary 1d space, optional
        If pre_domain is not None, it will be prepended to grid_domain. All
        transformations are performed for each element in pre_domain. Default is
        None.
    shift_directions : int, set of ints or None
        Collection of integers representing the directions in which a shift
        shall be modeled. Default is None.
    """
    shape = grid_domain.shape
    nufft = VariablePositionNufft(grid_domain,
                                  np.prod(grid_domain.shape),
                                  eps,
                                  pre_domain)
    domain = nufft.domain["grid"][-1]  # TODO: refactor
    coord_dom = domain.get_default_codomain(), UnstructuredDomain(len(shape))
    freq_axes = [np.fft.fftfreq(ss, dd)
                 for (ss, dd) in zip(shape, domain.distances)]
    fft_coord = np.stack(np.meshgrid(*freq_axes, indexing="ij"), axis=-1)
    fft_coord = makeField(coord_dom, fft_coord)

    nyquist_scaling = np.array(domain.get_default_codomain().distances)
    nyquist_scaling = DiagonalOperator(
        makeField(coord_dom[1], nyquist_scaling), coord_dom, spaces=1
    )
    pre_coord = ((nyquist_scaling @ Variable(coord_dom, "delta_coord")) + fft_coord)
    pre_coord = pre_coord.ducktape_left(nufft.domain["coord"])
    if pre_domain is None:
        pre_grid = FFTShiftOperator(domain)
    else:
        pre_grid = FFTShiftOperator(nufft.domain["grid"], spaces=-1)
    pre = pre_coord.ducktape_left("coord") + \
          pre_grid.ducktape("grid").ducktape_left("grid")

    tgt = domain.get_default_codomain()
    if pre_domain is not None:
        tgt = pre_domain, tgt
    post = DomainChangerAndReshaper(nufft.target, tgt)
    post = domain.scalar_dvol * post

    if shift_directions is None:
        shift_directions = tuple(range(len(shape)))
    elif isinstance(shift_directions, int):
        shift_directions = shift_directions,
    if min(shift_directions) < 0 or \
       max(shift_directions) >= len(shape) or \
       len(set(shift_directions)) != len(shift_directions):
        raise ValueError("Invalid values in shift_directions")
    if len(shift_directions) != len(shape) and len(shift_directions) == 1:
        ins = DomainTupleFieldInserter(pre.domain["delta_coord"], 1,
                                       shift_directions)
        ins = ins.ducktape((pre.domain["delta_coord"][0], UnstructuredDomain(1)))
        pre = pre @ ins.ducktape("delta_coord").ducktape_left("delta_coord")
    elif len(shift_directions) != len(shape):
        raise NotImplementedError
    return post @ nufft @ pre


class _NufftPlans:
    def __init__(self, grid_domain, coord, epsilon, ntrans):
        assert isinstance(coord, AnyArray)
        self._grid_domain = grid_domain
        self._coord = coord
        self._plans = {}
        self._cpu_args = dict(epsilon=epsilon,
                              periodicity=1/np.array(grid_domain.distances))
        self._cu_args = dict(modeord=0, eps=epsilon/10)
        self._ntrans = ntrans

    @property
    def ntrans(self):
        return self._ntrans

    def execute(self, data, nufft_type, forward):
        assert isinstance(data, AnyArray)
        device_id = data.device_id

        if device_id == -1:
            # Forward and backward plan is the same on CPU
            key = nufft_type
        else:
            # TODO: As soon as new cufinufft is released, also use just one plan
            # for both directions
            ntrans = 1 if self.ntrans is None else self.ntrans
            key = (device_id, nufft_type, forward, ntrans)

        grid_shape = self._grid_domain.shape

        # Cache plan
        if key not in self._plans:
            coord = self._coord.at(device_id)
            if device_id == -1:
                from ducc0.nufft import plan
                self._plans[key] = plan(nu2u=False, coord=coord.val,
                                        grid_shape=grid_shape,
                                        nthreads=nthreads(), **self._cpu_args)
            else:
                try:
                    from cufinufft import Plan
                except ImportError:
                    raise ImportError("cufinufft needs to be installed for "
                                      "non-uniform Fourier transforms on the GPU")
                period = AnyArray(self._cpu_args["periodicity"][None])
                period = period.at(device_id, check_fail=False)
                coord = 2 * np.pi * coord / period
                ndim = len(grid_shape)
                # TODO: Add single-precision support
                p = Plan(nufft_type=nufft_type, n_modes=grid_shape,
                         dtype="complex128",
                         isign=-1 if forward else 1, n_trans=ntrans, **self._cu_args)
                p.setpts(*[coord.val[:, i] for i in range(ndim)])
                self._plans[key] = p

        # Execute plan
        if device_id == -1:
            if nufft_type == 2:
                res = self._plans[key].u2nu(grid=data.val, forward=forward)
            elif nufft_type == 1:
                # TODO: add single precision support
                if self._ntrans is None:
                    shp = grid_shape
                else:
                    shp = (self._ntrans,) + grid_shape
                out = np.empty(shp, dtype=np.complex128)
                res = self._plans[key].nu2u(points=data.val, forward=forward, out=out)
        else:
            res = self._plans[key].execute(data.val)

        return AnyArray(res)
