# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank,
# Matteo Guardiani, Julian Rüstig

import operator
import dataclasses
from collections import namedtuple
from collections.abc import Mapping
from functools import partial, reduce
from typing import Callable, Optional, Tuple, Union

import numpy as np
from jax import numpy as jnp, vmap
from scipy.special import gamma, jv

from ..config import _config
from .gauss_markov import IntegratedWienerProcess
from .logger import logger
from .misc import wrap
from .model import Model, WrappedCall
from .num import lognormal_prior, normal_prior
from .tree_math import ShapeWithDtype, random_like


def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes=axes)
    c = _config.get("hartley_convention")
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub
    return add_or_sub(tmp.real, tmp.imag)


def get_sht(nside, axis, lmax, mmax, nthreads):
    from jaxbind.contrib.jaxducc0 import get_healpix_sht

    jsht = get_healpix_sht(nside, lmax, mmax, 0, nthreads)
    axis = int(axis)

    def f(inp):
        # Explicitly move axes around, may introduce unnecessary copies
        def trafo(x):
            return np.sqrt(4 * np.pi) * jsht(x[jnp.newaxis])[0][0]

        axs = axis % inp.ndim
        for i in reversed(range(inp.ndim)):
            if i < axs:
                trafo = vmap(trafo, in_axes=0, out_axes=0)
            elif i > axs:
                trafo = vmap(trafo, in_axes=1, out_axes=1)
        return trafo(inp)

    return f


def _unique_mode_distributor(m_length, uniqueness_rtol=1e-12):
    # Construct an array of unique mode lengths
    um = np.unique(m_length)
    tol = uniqueness_rtol * um[-1]
    um = um[np.diff(np.append(um, 2 * um[-1])) > tol]
    # Group modes based on their length and store the result as power
    # distributor
    binbounds = 0.5 * (um[:-1] + um[1:])
    m_length_idx = np.searchsorted(binbounds, m_length)
    m_count = np.bincount(m_length_idx.ravel(), minlength=um.size)
    if np.any(m_count == 0) or um.shape != m_count.shape:
        raise RuntimeError("invalid harmonic mode(s) encountered")
    return m_length_idx, um, m_count


def get_spherical_mode_distributor(
    nside: int, lmax=None, mmax=None, uniqueness_rtol=1e-12, distance_dtype=np.float64
):
    """Get the unique lengths of Spherical harmonic modes, a mapping from a mode
    to its length index and the multiplicity of each unique mode length.

    Parameters
    ----------
    nside : int
        Nside of the HEALPix sphere for which the associated harmonic space is
        constructed
    lmax : int, optional
        The maximum :math:`l` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        Must be :math:`\\ge 0`. If not supplied, it is set to `nside * 2`
    mmax : int, optional
        The maximum :math:`m` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        If not supplied, it is set to `lmax`.
        Must be :math:`\\ge 0` and :math:`\\le` `lmax`.
    uniqueness_rtol : float
        Relative tolerance to define unique lengths of harmonic mode vectors
        (k-vectors). Vectors with lengths that have a smaller relative distance
        to each other are treated identically
    distance_dtype : Any
        Dtype of the array of harmonic mode lengths canstructed internally

    Returns
    -------
    mode_length_idx : jnp.ndarray
        Index in power-space for every mode in harmonic-space. Can be used to
        distribute power from a power-space to the full harmonic grid.
    unique_mode_length : jnp.ndarray
        Unique length of spherical modes.
    mode_multiplicity : jnp.ndarray
        Multiplicity for each unique mode length.
    """
    if lmax is None:
        lmax = 2 * nside
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be >=0.")
    if mmax is None:
        mmax = lmax
    mmax = int(mmax)
    if mmax < 0 or mmax > lmax:
        raise ValueError("mmax must be >=0 and <=lmax.")
    size = (lmax + 1) ** 2 - (lmax - mmax) * (lmax - mmax + 1)

    ldist = np.empty((size,), dtype=distance_dtype)
    ldist[0 : lmax + 1] = np.arange(lmax + 1, dtype=distance_dtype)
    tmp = np.repeat(np.arange(lmax + 1, dtype=distance_dtype), 2)
    idx = lmax + 1
    for m in range(1, mmax + 1):
        ldist[idx : idx + 2 * (lmax + 1 - m)] = tmp[2 * m :]
        idx += 2 * (lmax + 1 - m)

    return _unique_mode_distributor(ldist, uniqueness_rtol=uniqueness_rtol), (
        lmax,
        mmax,
        size,
    )


def get_fourier_mode_distributor(
    shape: Union[tuple, int], distances: Union[tuple, float], uniqueness_rtol=1e-12
):
    """Get the unique lengths of the Fourier modes, a mapping from a mode to
    its length index and the multiplicity of each unique Fourier mode length.

    Parameters
    ----------
    shape : tuple of int or int
        Position-space shape.
    distances : tuple of float or float
        Position-space distances.

    Returns
    -------
    mode_length_idx : jnp.ndarray
        Index in power-space for every mode in harmonic-space. Can be used to
        distribute power from a power-space to the full harmonic grid.
    unique_mode_length : jnp.ndarray
        Unique length of Fourier modes.
    mode_multiplicity : jnp.ndarray
        Multiplicity for each unique Fourier mode length.
    uniqueness_rtol : float
        Relative tolerance to define unique lengths of harmonic mode vectors
        (k-vectors). Vectors with lengths that have a smaller relative distance
        to each other are treated identically
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    # Compute length of modes
    mspc_distances = 1.0 / (np.array(shape) * np.array(distances))
    m_length = np.arange(shape[0])
    m_length = np.minimum(m_length, shape[0] - m_length) * mspc_distances[0]
    if len(shape) != 1:
        m_length *= m_length
        for i in range(1, len(shape)):
            tmp = np.arange(shape[i])
            tmp = np.minimum(tmp, shape[i] - tmp) * mspc_distances[i]
            tmp *= tmp
            m_length = np.expand_dims(m_length, axis=-1) + tmp
        m_length = np.sqrt(m_length)

    return _unique_mode_distributor(m_length, uniqueness_rtol=uniqueness_rtol)


RegularCartesianGrid = namedtuple(
    "RegularCartesianGrid",
    (
        "shape",
        "total_volume",
        "distances",
        "harmonic_grid",
    ),
    defaults=(None,),
)

RegularFourierGrid = namedtuple(
    "RegularFourierGrid",
    (
        "shape",
        "power_distributor",
        "mode_multiplicity",
        "mode_lengths",
        "relative_log_mode_lengths",
        "log_volume",
    ),
)

HEALPixGrid = namedtuple(
    "HEALPixGrid",
    (
        "nside",
        "shape",
        "total_volume",
        "harmonic_grid",
    ),
    defaults=(None,),
)

LMGrid = namedtuple(
    "LMGrid",
    (
        "lmax",
        "mmax",
        "shape",
        "power_distributor",
        "mode_multiplicity",
        "mode_lengths",
        "relative_log_mode_lengths",
        "log_volume",
    ),
)


def _log_modes(m_length):
    um = m_length.copy()
    um[1:] = np.log(um[1:])
    um[1:] -= um[1]
    assert um[0] == 0.0
    log_vol = um[2:] - um[1:-1]
    assert um.shape[0] - 2 == log_vol.shape[0]
    return um, log_vol


def make_grid(
    shape, distances, harmonic_type
) -> Union[RegularCartesianGrid, HEALPixGrid]:
    """Creates the grid for the amplitude model"""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    # Pre-compute lengths of modes and indices for distributing power
    if harmonic_type.lower() == "fourier":
        distances = tuple(np.broadcast_to(distances, np.shape(shape)))

        totvol = np.prod(np.array(shape) * np.array(distances))
        m_length_idx, m_length, m_count = get_fourier_mode_distributor(shape, distances)
        um, log_vol = _log_modes(m_length)
        harmonic_grid = RegularFourierGrid(
            shape=shape,
            power_distributor=m_length_idx,
            mode_multiplicity=m_count,
            mode_lengths=m_length,
            relative_log_mode_lengths=um,
            log_volume=log_vol,
        )
        # TODO: cache results such that only references are used afterwards
        grid = RegularCartesianGrid(
            shape=shape,
            total_volume=totvol,
            distances=distances,
            harmonic_grid=harmonic_grid,
        )
    elif harmonic_type.lower() == "spherical":
        if len(shape) != 1:
            msg = "`shape` must be length one. Its the nside of the spherical grid."
            raise ValueError(msg)
        nside = shape[0]
        ((m_length_idx, m_length, m_count), (lmax, mmax, size)) = (
            get_spherical_mode_distributor(nside)
        )
        um, log_vol = _log_modes(m_length)
        harmonic_grid = LMGrid(
            lmax=lmax,
            mmax=mmax,
            shape=(size,),
            power_distributor=m_length_idx,
            mode_multiplicity=m_count,
            mode_lengths=m_length,
            relative_log_mode_lengths=um,
            log_volume=log_vol,
        )
        grid = HEALPixGrid(
            nside=nside,
            shape=(12 * nside**2,),
            total_volume=4 * np.pi,
            harmonic_grid=harmonic_grid,
        )
    else:
        ve = f"invalid `harmonic_type` {harmonic_type!r}"
        raise ValueError(ve)
    return grid


def _remove_slope(rel_log_mode_dist, x):
    sc = rel_log_mode_dist / rel_log_mode_dist[-1]
    return x - x[-1] * sc


class MaternAmplitude(Model):
    cutoff: Callable = dataclasses.field(metadata=dict(static=False))
    loglogslope: Callable = dataclasses.field(metadata=dict(static=False))
    scale: Callable = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        grid: Union[RegularCartesianGrid, RegularFourierGrid, HEALPixGrid, LMGrid],
        scale: Optional[Callable],
        cutoff: Callable,
        loglogslope: Callable,
        renormalize_amplitude: bool,
        prefix: str = "",
        kind: str = "amplitude",
    ):
        """Initializes a model that computes the amplitude of a Matérn-kernel
        power spectrum.

        See
        :class:`nifty.re.correlated_field.CorrelatedFieldMaker.add_fluctuations
        _matern`
        for more details on the parameters.

        See also
        --------
        `Causal, Bayesian, & non-parametric modeling of the SARS-CoV-2 viral
        load vs. patient's age`, Guardiani, Matteo and Frank, Philipp and Kostić,
        Andrija and Edenhofer, Gordian and Roth, Jakob and Uhlmann, Berit and
        Enßlin, Torsten, `<https://arxiv.org/abs/2105.13483>`_
        `<https://doi.org/10.1371/journal.pone.0275011>`_
        """

        self.grid = grid
        self.cutoff = WrappedCall(cutoff, name=prefix + "cutoff")
        self.loglogslope = WrappedCall(loglogslope, name=prefix + "loglogslope")
        self.scale = (
            WrappedCall(scale, name=prefix + "scale") if scale is not None else None
        )

        self.kind = kind.lower()

        supported_kinds = {"amplitude", "power"}
        if self.kind not in supported_kinds:
            raise ValueError(
                f"Invalid kind specified {self.kind!r}, "
                f"supported kinds: {supported_kinds}"
            )

        self.renormalize_amplitude = renormalize_amplitude

        models = [self.scale, self.cutoff, self.loglogslope]
        domain = reduce(operator.or_, [m.domain for m in models if m is not None])

        super().__init__(domain=domain, white_init=True)

    def __call__(self, primals: Mapping) -> jnp.ndarray:
        if self.scale is None:
            scl = 1.0
        else:
            scl = self.scale(primals)

        ctf = self.cutoff(primals)
        slp = self.loglogslope(primals)

        ln_spectrum = (
            0.25 * slp * jnp.log1p((self.grid.harmonic_grid.mode_lengths / ctf) ** 2)
        )
        spectrum = jnp.exp(ln_spectrum)

        norm = 1.0
        if self.renormalize_amplitude:
            if self.kind == "amplitude":
                norm = jnp.sqrt(
                    jnp.sum(
                        self.grid.harmonic_grid.mode_multiplicity[1:]
                        * spectrum[1:] ** 2
                    )
                )
            elif self.kind == "power":
                norm = jnp.sqrt(
                    jnp.sum(
                        self.grid.harmonic_grid.mode_multiplicity[1:] * spectrum[1:]
                    )
                )

            # Due to integral in harmonic space
            norm /= jnp.sqrt(self.grid.total_volume)

        if self.kind.lower() == "power":
            spectrum = jnp.sqrt(spectrum)
        spectrum = scl * (jnp.sqrt(self.grid.total_volume) / norm) * spectrum
        spectrum = spectrum.at[0].set(self.grid.total_volume)

        return spectrum


class NonParametricAmplitude(Model):
    loglogavgslope: Callable = dataclasses.field(metadata=dict(static=False))
    fluctuations: Callable = dataclasses.field(metadata=dict(static=False))
    flexibility: Optional[Callable] = dataclasses.field(metadata=dict(static=False))
    asperity: Optional[Callable] = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        grid: Union[RegularCartesianGrid, RegularFourierGrid, HEALPixGrid, LMGrid],
        fluctuations: Optional[Callable],
        loglogavgslope: Callable,
        flexibility: Optional[Callable] = None,
        asperity: Optional[Callable] = None,
        prefix: str = "",
        kind: str = "amplitude",
    ):
        """Initializes a model which computes the amplitude of a non-parametric
        power spectrum.

        See
        :class:`nifty.re.correlated_field.CorrelatedFieldMaker.add_fluctuations`
        for more details on the parameters.

        See also
        --------
        `Variable structures in M87* from space, time and frequency resolved
        interferometry`, Arras, Philipp and Frank, Philipp and Haim, Philipp
        and Knollmüller, Jakob and Leike, Reimar and Reinecke, Martin and
        Enßlin, Torsten, `<https://arxiv.org/abs/2002.05218>`_
        `<http://dx.doi.org/10.1038/s41550-021-01548-0>`_
        """
        self.grid = grid
        log_vol = grid.harmonic_grid.log_volume
        self.kind = kind.lower()

        supported_kinds = {"amplitude", "power"}
        if self.kind not in supported_kinds:
            raise ValueError(
                f"Invalid kind specified {self.kind!r}, "
                f"supported kinds: {supported_kinds}"
            )

        self._loglogavgslope = WrappedCall(
            loglogavgslope, name=prefix + "loglogavgslope", white_init=True
        )
        self.fluctuations = (
            WrappedCall(fluctuations, name=prefix + "fluctuations", white_init=True)
            if fluctuations is not None
            else None
        )
        if flexibility is not None and (log_vol.size > 0):
            flexibility = WrappedCall(
                flexibility, name=prefix + "flexibility", white_init=True
            )
            assert log_vol is not None
            assert (
                self.grid.harmonic_grid.relative_log_mode_lengths.ndim
                == log_vol.ndim
                == 1
            )
            if asperity is not None:
                asperity = WrappedCall(
                    asperity, name=prefix + "asperity", white_init=True
                )
            self._deviations = IntegratedWienerProcess(
                jnp.zeros((2,)),
                flexibility,
                log_vol,
                name=prefix + "spectrum",
                asperity=asperity,
            )
        else:
            self._deviations = None

        models = [
            self.fluctuations,
            self._loglogavgslope,
            self._deviations,
        ]
        domain = reduce(operator.or_, [m.domain for m in models if m is not None])

        super().__init__(domain=domain, white_init=True)

    def __call__(self, primals: Mapping) -> jnp.ndarray:
        mode_multiplicity = self.grid.harmonic_grid.mode_multiplicity
        relative_log_mode_lengths = self.grid.harmonic_grid.relative_log_mode_lengths

        flu = 1.0 if self.fluctuations is None else self.fluctuations(primals)
        slope = self._loglogavgslope(primals)
        slope *= relative_log_mode_lengths
        ln_spectrum = slope

        if self._deviations is not None:
            twolog = self._deviations(primals)
            # Prepend zeromode
            twolog = jnp.concatenate((jnp.zeros((1,)), twolog[:, 0]))
            ln_spectrum += _remove_slope(relative_log_mode_lengths, twolog)

        # Exponentiate and norm the power spectrum
        spectrum = jnp.exp(ln_spectrum)

        # Take the sqrt of the integral of the slope w/o fluctuations and
        # zero-mode while taking into account the multiplicity of each mode
        if self.kind == "amplitude":
            norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:] ** 2))
            norm /= jnp.sqrt(
                self.grid.total_volume
            )  # Due to integral in harmonic space
            amplitude = flu * (jnp.sqrt(self.grid.total_volume) / norm) * spectrum
        elif self.kind == "power":
            norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:]))
            norm /= jnp.sqrt(
                self.grid.total_volume
            )  # Due to integral in harmonic space
            amplitude = (
                flu * (jnp.sqrt(self.grid.total_volume) / norm) * jnp.sqrt(spectrum)
            )
        amplitude = amplitude.at[0].set(self.grid.total_volume)
        return amplitude

IsotropicGrid = namedtuple(
    "IsotropicGrid", ("dim", "total_volume", "radial_bins", "harmonic_grid")
)
IsotropicGrid.__doc__ = """Grid of the amplitude models of kernel-based spaces.

Fields
------
dim : int
    Dimension of the space the points live in.
total_volume : float
    Always ``1.0``; the covariance is normalized by its value at zero
    distance, which is the sum of ``mode_multiplicity * amplitude**2``.
radial_bins : array ``(n_r,)``
    Radii at which the covariance is tabulated, ``radial_bins[0] == 0``.
harmonic_grid : RegularFourierGrid
    ``mode_lengths`` are the edges of the ``k`` bins (``mode_lengths[0] == 0``
    is the zero mode) and ``mode_multiplicity`` the volumes of the ``k``
    shells between them; the amplitude models need nothing else.
"""


def make_isotropic_grid(
    dim, r_min=None, r_max=None, n_bins=None, *, radial_bins=None, k_bins=None
) -> IsotropicGrid:
    """Builds the grid for the amplitude model of a kernel-based space.

    By default the radii and the ``k`` bin edges are logarithmically spaced
    with a leading zero, the radii between `r_min` and `r_max`, the ``k``
    edges between ``1 / r_max`` and ``1 / r_min`` (``k`` counts oscillations
    per unit length as for the regular grids). Explicit `radial_bins` and
    `k_bins` take precedence.
    """
    if radial_bins is None:
        radial_bins = np.concatenate(([0.0], np.geomspace(r_min, r_max, n_bins - 1)))
    if k_bins is None:
        k_bins = np.concatenate(
            ([0.0], np.geomspace(1.0 / r_max, 1.0 / r_min, n_bins - 1))
        )
    radial_bins = np.asarray(radial_bins, dtype=np.float64)
    k_bins = np.asarray(k_bins, dtype=np.float64)
    if radial_bins.ndim != 1 or radial_bins[0] != 0.0 or np.any(np.diff(radial_bins) <= 0):
        raise ValueError("`radial_bins` must be increasing and start at zero")
    if k_bins.ndim != 1 or k_bins[0] != 0.0 or np.any(np.diff(k_bins) <= 0):
        raise ValueError("`k_bins` must be increasing and start at zero (the zero mode)")
    um, log_vol = _log_modes(k_bins.copy())
    # volume of the k ball of radius k_bins in `dim` dimensions
    ball = np.pi ** (dim / 2) / gamma(dim / 2 + 1) * k_bins**dim
    multiplicity = np.concatenate(([0.0], np.diff(ball)))
    harmonic_grid = RegularFourierGrid(
        shape=k_bins.shape,
        power_distributor=np.arange(k_bins.size),
        mode_multiplicity=multiplicity,
        mode_lengths=k_bins,
        relative_log_mode_lengths=um,
        log_volume=log_vol,
    )
    return IsotropicGrid(
        dim=int(dim),
        total_volume=1.0,
        radial_bins=radial_bins,
        harmonic_grid=harmonic_grid,
    )


def _hankel_weights(k_bins, radial_bins, dim):
    """Matrix ``W`` with ``C(r) = W @ P`` for a power spectrum ``P`` that is
    constant on the ``k`` shells between the edges `k_bins` (`P` has one entry
    less than `k_bins`), ``C(r) = int P(k) exp(2 pi i k.r) d^dim k``.
    """
    k = np.asarray(k_bins, dtype=np.float64)[np.newaxis, :]
    r = np.asarray(radial_bins, dtype=np.float64)[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        lim = (k / r) ** (dim / 2) * jv(dim / 2, 2 * np.pi * k * r)
    ball = np.pi ** (dim / 2) / gamma(dim / 2 + 1) * k**dim
    return np.where(r > 0, lim[:, 1:] - lim[:, :-1], ball[:, 1:] - ball[:, :-1])


class Space:
    """Position space of one factor of a correlated field.

    A correlated field is the outer product of factors, one per call of
    :meth:`CorrelatedFieldMaker.add_fluctuations`. Each factor is described by
    a `Space` which knows the shape of the field along it, the shape of its
    latent excitations and the grid its amplitude model lives on, and which
    applies the factor's linear map to the excitations.

    The excitations of every factor contain its zero mode: for the harmonic
    spaces at index zero of the harmonic grid, for the kernel-based spaces as
    an additional leading entry (``excitation_shape == (n_points + 1,)``). The
    zero mode maps to the constant one along the factor, so that the field is
    ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)`` as before.

    Attributes
    ----------
    shape : tuple of int
        Shape of the field along this factor.
    excitation_shape : tuple of int
        Shape of the excitations of this factor.
    grid :
        Grid of the amplitude model (with ``harmonic_grid`` and
        ``total_volume``).
    """

    def __init__(self, shape, excitation_shape, grid):
        self.shape = tuple(int(s) for s in shape)
        self.excitation_shape = tuple(int(s) for s in excitation_shape)
        self.grid = grid

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def make_apply(self, axes):
        """Returns ``apply(x, amplitude, zeromode)``, the linear map of this
        factor along the `axes` of `x` holding its excitations: the zero mode
        becomes the constant one, the remaining excitations are correlated with
        the amplitude divided by the zero mode (see
        :meth:`CorrelatedFieldMaker.get_normalized_amplitudes`). The output has
        ``self.shape`` in place of the excitation axes.
        """
        raise NotImplementedError()


class HarmonicSpace(Space):
    """Factor with a harmonic partner space (regular or spherical grid)."""

    def __init__(self, grid):
        super().__init__(grid.shape, grid.harmonic_grid.shape, grid)

    def harmonic_transform(self, axes):
        """Harmonic transform acting on `axes`."""
        raise NotImplementedError()

    def make_apply(self, axes):
        axes = tuple(int(ax) for ax in axes)
        ht = self.harmonic_transform(axes)
        harmonic_dvol = 1.0 / self.grid.total_volume
        power_distributor = self.grid.harmonic_grid.power_distributor

        def apply(x, amplitude, zeromode):
            amp = amplitude.at[1:].mul(1.0 / zeromode)[power_distributor]
            shp = [1] * x.ndim
            for ax, s in zip(axes, amp.shape):
                shp[ax] = s
            return harmonic_dvol * ht(amp.reshape(shp) * x)

        return apply


class RegularSpace(HarmonicSpace):
    """Regular Cartesian grid with periodic boundaries, Fourier harmonic space.

    Parameters
    ----------
    shape : tuple of int or int
        Shape of the grid.
    distances : tuple of float or float
        Pixel distances along the axes.
    """

    def __init__(self, shape, distances):
        super().__init__(make_grid(shape, distances, "fourier"))

    @property
    def distances(self):
        return self.grid.distances

    def harmonic_transform(self, axes):
        return partial(hartley, axes=axes)


class SphericalSpace(HarmonicSpace):
    """HEALPix sphere with spherical harmonics as harmonic space.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    """

    def __init__(self, nside):
        super().__init__(make_grid((int(nside),), None, "spherical"))

    @property
    def nside(self):
        return self.grid.nside

    def harmonic_transform(self, axes):
        (axis,) = axes
        return get_sht(
            nside=self.grid.nside,
            axis=axis,
            lmax=self.grid.harmonic_grid.lmax,
            mmax=self.grid.harmonic_grid.mmax,
            nthreads=1,
        )


class KernelSpace(Space):
    """Factor whose correlations are an isotropic covariance function.

    The amplitude model lives on the ``k`` bins of an :class:`IsotropicGrid`.
    Its square, the power spectrum, is taken constant on the ``k`` shells
    between the edges and transformed to the covariance on the radial bins,
    ``C(r) = int P(k) exp(2 pi i k.r) d^dim k``, with ``C(0)`` the sum of
    ``mode_multiplicity * amplitude**2``, that is ``fluctuations**2`` for the
    amplitude models of :class:`CorrelatedFieldMaker`. The zero mode of the
    amplitude is not used; the zero mode of the factor is the leading entry
    of the excitations. Subclasses generate the field from the covariance
    (:meth:`apply_sqrt`).

    Parameters
    ----------
    shape : tuple of int
        Shape of the field along this factor.
    dim : int
        Dimension of the space of the points.
    n_points : int, optional
        Number of excitations without the zero mode (the size of `shape` by
        default; more for multigrid generators).
    r_min, r_max, n_bins : float, float, int
        Range and number of the radial bins and of the ``k`` bins, see
        :func:`make_isotropic_grid`. `r_max` must exceed the largest distance
        the field generator evaluates the covariance at.
    jitter : float
        Relative inflation of ``C(0)`` for the numerical stability of the
        generator (its "nugget").
    radial_bins, k_bins : arrays, optional
        Explicit bins in place of `r_min`, `r_max` and `n_bins`.
    """

    def __init__(
        self,
        shape,
        dim,
        *,
        n_points=None,
        r_min=None,
        r_max=None,
        n_bins=None,
        jitter=1e-6,
        radial_bins=None,
        k_bins=None,
    ):
        grid = make_isotropic_grid(
            dim, r_min, r_max, n_bins, radial_bins=radial_bins, k_bins=k_bins
        )
        n_points = int(np.prod(shape)) if n_points is None else int(n_points)
        super().__init__(shape, (n_points + 1,), grid)
        self.dim = int(dim)
        self.jitter = float(jitter)
        self._hankel = jnp.asarray(
            _hankel_weights(self.k_bins, self.radial_bins, self.dim)
        )

    @property
    def radial_bins(self):
        return self.grid.radial_bins

    @property
    def k_bins(self):
        return self.grid.harmonic_grid.mode_lengths

    @property
    def n_points(self) -> int:
        return self.excitation_shape[0] - 1

    def covariance(self, amplitude):
        """Covariance on :attr:`radial_bins` for `amplitude` on the ``k`` bin
        edges (zero mode ignored), ``C(0)`` inflated by `jitter`."""
        cov = self._hankel @ amplitude[1:] ** 2
        return cov.at[0].mul(1.0 + self.jitter)

    def apply_sqrt(self, xi, covariance):
        """Field of shape ``(batch,) + shape`` for excitations `xi` of shape
        ``(batch, n_points)`` and the `covariance` on :attr:`radial_bins`."""
        raise NotImplementedError()

    def make_apply(self, axes):
        (axis,) = (int(ax) for ax in axes)
        ndim = len(self.shape)

        def apply(x, amplitude, zeromode):
            cov = self.covariance(amplitude)
            x = jnp.moveaxis(x, axis, -1)
            batch = x.shape[:-1]
            x = x.reshape((-1, x.shape[-1]))
            out = self.apply_sqrt(x[:, 1:], cov) / zeromode
            out = out + x[:, :1].reshape((-1,) + (1,) * ndim)
            out = out.reshape(batch + self.shape)
            src = tuple(range(len(batch), len(batch) + ndim))
            return jnp.moveaxis(out, src, tuple(range(axis, axis + ndim)))

        return apply


def _parse_space(shape, distances, harmonic_type) -> Space:
    if isinstance(shape, Space):
        if distances is not None:
            te = (
                "`distances` must not be given together with a `Space`;"
                " pass the remaining arguments by keyword"
            )
            raise TypeError(te)
        return shape
    grid = make_grid(shape, distances, harmonic_type)
    if isinstance(grid, HEALPixGrid):
        return SphericalSpace(grid.nside)
    return RegularSpace(grid.shape, grid.distances)


class CorrelatedFieldMaker:
    """Construction helper for hierarchical correlated field models.

    The correlated field models are parametrized by creating square roots of
    power spectrum operators ("amplitudes") via calls to
    :func:`add_fluctuations*` that act on the targeted field subgrids.
    During creation of the :class:`CorrelatedFieldMaker`, a global offset from
    zero of the field model can be defined and an operator applying
    fluctuations around this offset is parametrized.

    Creation of the model operator is completed by calling the method
    :func:`finalize`, which returns the configured operator.

    See the method's initialization, :func:`add_fluctuations`,
    :func:`add_fluctuations_matern` and :func:`finalize` for further
    usage information."""

    def __init__(self, prefix: str):
        """Instantiate a CorrelatedFieldMaker object.

        Parameters
        ----------
        prefix : string
            Prefix to the names of the parameters of the CF operator.
        """
        self._azm = None
        self._offset_mean = None
        self._fluctuations = []
        self._spaces = []
        self._target_grids = []
        self._parameter_tree = {}

        self._prefix = prefix

    def add_fluctuations(
        self,
        shape: Union[tuple, int, Space],
        distances: Union[tuple, float, None] = None,
        fluctuations: Union[tuple, Callable, None] = None,
        loglogavgslope: Union[tuple, Callable, None] = None,
        flexibility: Union[tuple, Callable, None] = None,
        asperity: Union[tuple, Callable, None] = None,
        prefix: str = "",
        harmonic_type: str = "fourier",
        non_parametric_kind: str = "amplitude",
    ):
        """Adds a correlation structure to the to-be-made field.

        Correlations are described by their power spectrum and the subgrid on
        which they apply.

        Multiple calls to `add_fluctuations` are possible, in which case
        the constructed field will have the outer product of the individual
        power spectra as its global power spectrum.

        The parameters `fluctuations`, `flexibility`, `asperity` and
        `loglogavgslope` configure either the amplitude or the power
        spectrum model used on the target field subgrid of type
        `harmonic_type`. It is assembled as the sum of a power law component
        (linear slope in log-log amplitude-frequency-space), a smooth varying
        component (integrated Wiener process) and a ragged component
        (un-integrated Wiener process).

        Parameters
        ----------
        shape : tuple of int or Space
            Shape of the position space grid, or a :class:`Space` describing
            the factor (:class:`RegularSpace`, :class:`SphericalSpace`,
            :class:`nifty.re.ICRSpace`, :class:`nifty.re.GraphGPSpace`, ...);
            with a `Space`, `distances` must be left out and `harmonic_type`
            is ignored.
        distances : tuple of float or float
            Distances in the position space grid
        fluctuations : tuple of float (mean, std) or callable
            Total spectral energy, i.e. amplitude of the fluctuations
            (by default a priori log-normal distributed)
        loglogavgslope : tuple of float (mean, std) or callable
            Power law component exponent
            (by default a priori normal distributed)
        flexibility : tuple of float (mean, std) or callable or None
            Amplitude of the non-power-law power spectrum component
            (by default a priori log-normal distributed)
        asperity : tuple of float (mean, std) or callable or None
            Roughness of the non-power-law power spectrum component; use it to
            accommodate single frequency peak
            (by default a priori log-normal distributed)
        prefix : str
            Prefix of the power spectrum parameter names
        harmonic_type : str
            Description of the harmonic partner domain in which the amplitude
            lives
        non_parametric_kind : str
            If set to `'amplitude'`, the amplitude spectrum is described
            by the correlated field model parameters in the above.
            If set to `'power'`, the power spectrum is described by the
            correlated field model parameters in the above
            (by default set to `'amplitude'`).

        See also
        --------
        `Variable structures in M87* from space, time and frequency resolved
        interferometry`, Arras, Philipp and Frank, Philipp and Haim, Philipp
        and Knollmüller, Jakob and Leike, Reimar and Reinecke, Martin and
        Enßlin, Torsten, `<https://arxiv.org/abs/2002.05218>`_
        `<http://dx.doi.org/10.1038/s41550-021-01548-0>`_
        """
        space = _parse_space(shape, distances, harmonic_type)
        grid = space.grid

        flu = fluctuations
        if isinstance(flu, (tuple, list)):
            flu = lognormal_prior(*flu)
        elif not callable(flu):
            te = f"invalid `fluctuations` specified; got '{type(fluctuations)}'"
            raise TypeError(te)
        slp = loglogavgslope
        if isinstance(slp, (tuple, list)):
            slp = normal_prior(*slp)
        elif not callable(slp):
            te = f"invalid `loglogavgslope` specified; got '{type(loglogavgslope)}'"
            raise TypeError(te)

        flx = flexibility
        if isinstance(flx, (tuple, list)):
            flx = lognormal_prior(*flx)
        elif flx is not None and not callable(flx):
            te = f"invalid `flexibility` specified; got '{type(flexibility)}'"
            raise TypeError(te)
        asp = asperity
        if isinstance(asp, (tuple, list)):
            asp = lognormal_prior(*asp)
        elif asp is not None and not callable(asp):
            te = f"invalid `asperity` specified; got '{type(asperity)}'"
            raise TypeError(te)

        npa = NonParametricAmplitude(
            grid=grid,
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
        )
        self._fluctuations.append(npa)
        self._spaces.append(space)
        self._target_grids.append(grid)
        self._parameter_tree.update(npa.domain)

    def add_fluctuations_matern(
        self,
        shape: Union[tuple, int, Space],
        distances: Union[tuple, float, None] = None,
        scale: Union[tuple, Callable, None] = None,
        cutoff: Union[tuple, Callable, None] = None,
        loglogslope: Union[tuple, Callable, None] = None,
        renormalize_amplitude: bool = True,
        prefix: str = "",
        harmonic_type: str = "fourier",
        non_parametric_kind: str = "amplitude",
    ):
        """Adds a Matérn-kernel correlation structure to the
        field to be made.

        The Matérn-kernel spectrum is parametrized by

        .. math ::
            A(k) = \\frac{a}{\\left(1 + { \
                \\left(\\frac{|k|}{b}\\right) \
            }^2\\right)^{-c/4}}

        where :math:`a` is called the scale parameter, :math:`b`
        the represents the cutoff mode, and :math:`c` the spectral index
        of the resulting power spectrum.

        Parameters
        ----------
        shape : tuple of int or Space
            Shape of the position space grid, or a :class:`Space` describing
            the factor (see :meth:`add_fluctuations`); with a `Space`,
            `distances` must be left out and `harmonic_type` is ignored.
        distances : tuple of float or float
            Distances in the position space grid.
        scale : tuple of float (mean, std) or callable
            Total spectral energy, i.e. amplitude of the fluctuations
            (by default a priori log-normal distributed).
        cutoff : tuple of float (mean, std) or callable
            Power law component exponent
            (by default a priori normal distributed).
        loglogslope : tuple of float (mean, std) or callable or None
            Amplitude of the non-power-law power spectrum component
            (by default a priori log-normal distributed).
        renormalize_amplitude : bool
            Whether the amplitude of the process should be renormalized to
            ensure that the `scale` parameter relates to the scale of the
            fluctuations along the specified axis.
        prefix : str
            Prefix of the power spectrum parameter names.
        harmonic_type : str
            Description of the harmonic partner domain in which the amplitude
            lives.
        non_parametric_kind : str
            If set to `'amplitude'`, the amplitude spectrum is described
            by the Matérn kernel function in the above.
            If set to `'power'`, the power spectrum is described by the
            Matérn kernel function in the above
            (by default `'amplitude'`).

        See also
        --------
        `Causal, Bayesian, & non-parametric modeling of the SARS-CoV-2 viral
        load vs. patient's age`, Guardiani, Matteo and Frank, Kostić Andrija
        and Edenhofer, Gordian and Roth, Jakob and Uhlmann, Berit and
        Enßlin, Torsten, `<https://arxiv.org/abs/2105.13483>`_
        `<https://doi.org/10.1371/journal.pone.0275011>`_
        """
        space = _parse_space(shape, distances, harmonic_type)
        grid = space.grid

        if isinstance(scale, (tuple, list)):
            scale = lognormal_prior(*scale)
        elif not callable(scale):
            te = f"invalid `scale` specified; got '{type(scale)}'"
            raise TypeError(te)
        if isinstance(cutoff, (tuple, list)):
            cutoff = lognormal_prior(*cutoff)
        elif not callable(cutoff):
            te = f"invalid `cutoff` specified; got '{type(cutoff)}'"
            raise TypeError(te)
        if isinstance(loglogslope, (tuple, list)):
            loglogslope = normal_prior(*loglogslope)
        elif not callable(loglogslope):
            te = f"invalid `loglogslope` specified; got '{type(loglogslope)}'"
            raise TypeError(te)

        ma = MaternAmplitude(
            grid=grid,
            scale=scale,
            cutoff=cutoff,
            loglogslope=loglogslope,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
            renormalize_amplitude=renormalize_amplitude,
        )
        self._fluctuations.append(ma)
        self._spaces.append(space)
        self._target_grids.append(grid)
        self._parameter_tree.update(ma.domain)

    def set_amplitude_total_offset(
        self, offset_mean: float, offset_std: Union[tuple, Callable]
    ):
        """Sets the zero-mode for the combined amplitude operator

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the correlated field to be made.
        offset_std : tuple of float or callable
            Mean standard deviation and standard deviation of the standard
            deviation of the offset. No, this is not a word duplication.
            (By default a priori log-normal distributed)
        """
        if self._offset_mean is not None and self._azm is not None:
            msg = "Overwriting the previous mean offset and zero-mode"
            logger.warning(msg)

        self._offset_mean = offset_mean
        zm = offset_std
        if not callable(zm):
            if zm is None or len(zm) != 2:
                raise TypeError(f"`offset_std` of invalid type {type(zm)!r}")
            zm = lognormal_prior(*zm)

        self._azm = wrap(zm, self._prefix + "zeromode")
        self._parameter_tree[self._prefix + "zeromode"] = ShapeWithDtype(())

    @property
    def amplitude_total_offset(self) -> Callable:
        """Returns the total offset of the amplitudes"""
        if self._azm is None:
            nie = "You need to set the `amplitude_total_offset` first"
            raise NotImplementedError(nie)
        return self._azm

    @property
    def azm(self):
        """Alias for `amplitude_total_offset`"""
        return self.amplitude_total_offset

    @property
    def fluctuations(self) -> Tuple[Callable, ...]:
        """Returns the added fluctuations, i.e. un-normalized amplitudes

        Their scales are only meaningful relative to one another. Their
        absolute scale bares no information.
        """
        return tuple(self._fluctuations)

    def get_normalized_amplitudes(self) -> Tuple[Callable, ...]:
        """Returns the normalized amplitude operators used in the final model

        The amplitude operators are corrected for the otherwise degenerate
        zero-mode. Their scales are only meaningful relative to one another.
        Their absolute scale bares no information.
        """

        def _mk_normed_amp(amp):  # Avoid late binding
            def normed_amplitude(p):
                return amp(p).at[1:].mul(1.0 / self.azm(p))

            return normed_amplitude

        return tuple(_mk_normed_amp(amp) for amp in self._fluctuations)

    @property
    def amplitude(self) -> Callable:
        """Returns the added fluctuation, i.e. un-normalized amplitude"""
        if len(self._fluctuations) > 1:
            s = (
                "If more than one spectrum is present in the model,"
                " no unique set of amplitudes exist because only the"
                " relative scale is determined."
            )
            raise NotImplementedError(s)
        amp = self._fluctuations[0]

        def ampliude_w_zm(p):
            return amp(p).at[0].mul(self.azm(p))

        return ampliude_w_zm

    @property
    def power_spectrum(self) -> Callable:
        """Returns the power spectrum"""
        amp = self.amplitude

        def power(p):
            return amp(p) ** 2

        return power

    def finalize(self) -> Model:
        """Finishes off the model construction process and returns the
        constructed operator.

        The field is the outer product of the factors added with
        :meth:`add_fluctuations` (or :meth:`add_fluctuations_matern`),
        ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)``, applied to a
        single array of excitations ``prefix + "xi"`` whose shape is the
        concatenation of the factors' excitation shapes (see :class:`Space`);
        the zero mode of a kernel-based factor is the leading entry along its
        axis. The returned model has the attributes ``spaces``,
        ``target_grids`` (grids of the amplitude models) and
        ``normalized_amplitudes``.
        """
        if len(self._spaces) == 0:
            raise ValueError("add fluctuations before finalizing the model")
        applies = []
        excitation_shape = ()
        offset = 0
        for space in self._spaces:
            n_exc = len(space.excitation_shape)
            applies.append(space.make_apply(tuple(range(offset, offset + n_exc))))
            excitation_shape += space.excitation_shape
            offset += len(space.shape)

        # Register the parameters for the excitations
        # TODO: actually account for the dtype here
        pfx = self._prefix + "xi"
        self._parameter_tree[pfx] = ShapeWithDtype(excitation_shape)

        amplitudes = tuple(self._fluctuations)
        namps = self.get_normalized_amplitudes()

        def correlated_field(p):
            zeromode = self.azm(p)
            x = p[pfx]
            for apply, amp in zip(applies, amplitudes):
                x = apply(x, amp(p), zeromode)
            return self._offset_mean + zeromode * x

        init = {
            k: partial(random_like, primals=v) for k, v in self._parameter_tree.items()
        }
        cf = Model(correlated_field, domain=self._parameter_tree.copy(), init=init)
        cf.normalized_amplitudes = namps
        cf.target_grids = tuple(self._target_grids)
        cf.spaces = tuple(self._spaces)
        return cf
