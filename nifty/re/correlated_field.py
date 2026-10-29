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
import jax
from jax import numpy as jnp, vmap
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from scipy.special import gamma, jv

from ..config import _config
from .gauss_markov import IntegratedWienerProcess
from .logger import logger
from .misc import wrap
from .model import Model, ModelMeta, WrappedCall
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


class Space(metaclass=ModelMeta):
    """Geometry of one factor of a correlated field.

    A correlated field is the outer product of factors, one per call of
    :meth:`CorrelatedFieldMaker.add_fluctuations`. A `Space` describes a
    factor's geometry only: the shape of the field along it, the shape of its
    latent excitations and the grid its amplitude model lives on. The linear
    maps are implemented elsewhere: harmonic transforms for the harmonic
    spaces (in the correlated field) and the Gaussian process models
    :class:`~nifty.re.gaussian_process.GraphGP` and
    :class:`~nifty.re.gaussian_process.ICRGP` for the kernel-based spaces, which the
    correlated field builds from the space's settings and its amplitude.

    The excitations of a factor are one array (``excitation_shapes[0]``) or,
    for a multigrid factor, a list of arrays (one per level). Harmonic factors
    carry their zero mode at index zero of their harmonic array; for
    kernel-based factors the zero mode is a separate excitation block of the
    correlated field (see :func:`excitation_blocks`), so that the field is
    ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)`` as before.

    Spaces are JAX pytrees (like models): their arrays are dynamic leaves.

    Attributes
    ----------
    shape : tuple of int
        Shape of the field along this factor.
    excitation_shapes : tuple of tuples of int
        Shapes of the excitation arrays of this factor.
    grid :
        Grid of the amplitude model (with ``harmonic_grid`` and
        ``total_volume``).
    """

    def __init__(self, shape, excitation_shapes, grid):
        self.shape = tuple(int(s) for s in shape)
        self.excitation_shapes = tuple(tuple(int(s) for s in sh) for sh in excitation_shapes)
        self.grid = grid

    @property
    def structured(self) -> bool:
        """Whether the factor's excitations are a list of arrays (a multigrid factor)."""
        return len(self.excitation_shapes) > 1

    @property
    def excitation_shape(self):
        """The shape of the (single) excitation array of the factor."""
        if self.structured:
            raise AttributeError("the excitations are a list of arrays, see `excitation_shapes`")
        return self.excitation_shapes[0]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


class HarmonicSpace(Space):
    """Factor with a harmonic partner space (regular or spherical grid)."""

    def __init__(self, grid):
        super().__init__(grid.shape, (grid.harmonic_grid.shape,), grid)


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


class SpectralKernel(Model):
    """Covariance table of a kernel-based factor from its amplitude model.

    The amplitude lives on the ``k`` bin edges of an :class:`IsotropicGrid`;
    its square, the power spectrum, is taken constant on the ``k`` shells
    between the edges and transformed to the covariance on the radial bins,
    ``C(r) = int P(k) exp(2 pi i k.r) d^dim k`` (`hankel` holds the weights,
    see :func:`_hankel_weights`), so that ``C(0)`` is the sum of
    ``mode_multiplicity * amplitude**2``, that is ``fluctuations**2`` for the
    amplitude models of :class:`CorrelatedFieldMaker`. ``C(0)`` is inflated by
    `jitter`. The zero mode of the amplitude is not used.
    """

    amplitude: Model = dataclasses.field(metadata=dict(static=False))
    hankel: jnp.ndarray = dataclasses.field(metadata=dict(static=False))
    radial_bins: jnp.ndarray = dataclasses.field(metadata=dict(static=False))

    def __init__(self, amplitude, hankel, radial_bins, jitter=0.0):
        self.amplitude = amplitude
        self.hankel = jnp.asarray(hankel)
        self.radial_bins = jnp.asarray(radial_bins)
        self.jitter = float(jitter)
        super().__init__(domain=amplitude.domain, init=amplitude.init)

    def __call__(self, p):
        cov = self.hankel @ self.amplitude(p)[1:] ** 2
        return cov.at[0].mul(1.0 + self.jitter)


class KernelSpace(Space):
    """Factor whose correlations are an isotropic covariance function.

    The amplitude model lives on the ``k`` bins of an :class:`IsotropicGrid`
    and is turned into the covariance on the radial bins by a
    :class:`SpectralKernel` (:meth:`spectral_kernel`). The zero mode of the
    factor is the leading entry of the excitations. Subclasses hold the
    geometry and settings of a Gaussian process model that generates the
    field from a covariance table (see :class:`Space`).

    Parameters
    ----------
    shape : tuple of int
        Shape of the field along this factor.
    dim : int
        Dimension of the space of the points.
    excitation_shapes : tuple of tuples, optional
        Shapes of the excitation arrays of the generator (``(shape,)`` by
        default; the level shapes for a multigrid generator).
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

    radial_bins: jnp.ndarray = dataclasses.field(metadata=dict(static=False))
    _hankel: jnp.ndarray = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        shape,
        dim,
        *,
        excitation_shapes=None,
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
        excitation_shapes = (tuple(shape),) if excitation_shapes is None else tuple(excitation_shapes)
        super().__init__(shape, excitation_shapes, grid)
        self.dim = int(dim)
        self.jitter = float(jitter)
        self.radial_bins = jnp.asarray(grid.radial_bins)
        self._hankel = jnp.asarray(
            _hankel_weights(self.k_bins, grid.radial_bins, self.dim)
        )

    @property
    def k_bins(self):
        return self.grid.harmonic_grid.mode_lengths

    @property
    def n_points(self) -> int:
        """Total number of excitations of the generator."""
        return int(sum(np.prod(sh) for sh in self.excitation_shapes))

    def covariance(self, amplitude):
        """Covariance on :attr:`radial_bins` for `amplitude` values on the
        ``k`` bin edges (zero mode ignored), ``C(0)`` inflated by `jitter`."""
        cov = self._hankel @ amplitude[1:] ** 2
        return cov.at[0].mul(1.0 + self.jitter)

    def spectral_kernel(self, amplitude: Model) -> SpectralKernel:
        """The kernel model of this factor for the amplitude model."""
        return SpectralKernel(amplitude, self._hankel, self.radial_bins, self.jitter)

    @property
    def gp_kwargs(self) -> dict:
        """Settings of the Gaussian process model built for this factor."""
        raise NotImplementedError()


def build_gp(space: KernelSpace, kernel, *, mesh=None):
    """The Gaussian process model of a kernel-based `space` for `kernel` (a
    kernel model or a fixed table, see
    :class:`~nifty.re.gaussian_process.KernelGP`); with a one-dimensional
    `mesh` the model is generated on the mesh, sharded along the space's
    leading axis (graphgp graphs are then built on the devices)."""
    from .graphgp_space import GraphGPSpace
    from .multi_grid.icr_space import ICRSpace

    if isinstance(space, ICRSpace):
        from .gaussian_process import ICRGP

        return ICRGP(space.multigrid, kernel, mesh=mesh, **space.gp_kwargs)
    if isinstance(space, GraphGPSpace):
        from .gaussian_process import GraphGP

        return GraphGP(space.points_or_graph(), kernel, shape=space.shape, mesh=mesh, **space.gp_kwargs)
    raise TypeError(f"no Gaussian process model for {type(space).__name__}")


def _harmonic_transform(space: HarmonicSpace, axes):
    """The harmonic transform of a harmonic space acting on `axes`."""
    if isinstance(space, RegularSpace):
        return partial(hartley, axes=axes)
    if isinstance(space, SphericalSpace):
        (axis,) = axes
        return get_sht(
            nside=space.grid.nside,
            axis=axis,
            lmax=space.grid.harmonic_grid.lmax,
            mmax=space.grid.harmonic_grid.mmax,
            nthreads=1,
        )
    raise TypeError(f"no harmonic transform for {type(space).__name__}")


def _apply_harmonic(space, x, axes, amplitude, zeromode):
    """Harmonic factor: the normalized amplitude times the excitations,
    transformed to position space (the zero mode becomes the constant one)."""
    axes = tuple(int(ax) for ax in axes)
    amp = amplitude.at[1:].mul(1.0 / zeromode)[space.grid.harmonic_grid.power_distributor]
    shp = [1] * x.ndim
    for ax, s in zip(axes, amp.shape):
        shp[ax] = s
    return _harmonic_transform(space, axes)(amp.reshape(shp) * x) / space.grid.total_volume


def _apply_kernel(gp, x, pos, cov_vals, zeromode):
    """Kernel-based factor: `x` holds the factor's excitations at axes
    ``pos : pos + n`` (one array, or a list of per-level arrays for a
    structured generator); they are generated by the Gaussian process `gp`
    with the covariance table, divided by the zero mode, and the field's axes
    take their place."""
    pos = int(pos)
    structured = gp.structured
    if isinstance(x, list) and not structured:  # a multigrid factor elsewhere in the block: per level
        return [_apply_kernel(gp, a, pos, cov_vals, zeromode) for a in x]
    arrs = list(x) if structured else [x]
    n_exc = len(gp.excitation_shapes[0])
    moved = []
    for a in arrs:
        src = tuple(range(pos, pos + n_exc))
        a = jnp.moveaxis(a, src, tuple(range(a.ndim - n_exc, a.ndim)))
        moved.append(a)
    batch = moved[0].shape[: moved[0].ndim - n_exc]
    flat = [a.reshape((-1,) + a.shape[-n_exc:]) for a in moved]
    out = gp.generate(flat if structured else flat[0], cov_vals) / zeromode
    ndim = len(gp.shape)
    out = out.reshape(batch + gp.shape)
    src = tuple(range(len(batch), len(batch) + ndim))
    return jnp.moveaxis(out, src, tuple(range(pos, pos + ndim)))


def _init_sharded(draw, sharding, key, *args, **kwargs):
    """`draw` (random_like of a block) placed on `sharding` (one or per array)."""
    x = draw(key, *args, **kwargs)
    if isinstance(sharding, list):
        return [jax.device_put(a, s) for a, s in zip(x, sharding)]
    return jax.device_put(x, sharding)


def excitation_blocks(spaces, prefix):
    """The excitation blocks of a correlated field over `spaces`.

    The field is ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)``.
    Expanding the product over the kernel-based factors gives one block of
    excitations per subset ``Z`` of them that sit at their zero mode (the
    constant one): the main block (``Z`` empty, ``prefix + "xi"``) carries the
    excitations of every factor, the zero-mode blocks (``prefix + "xi0"`` for a
    single kernel-based factor, ``prefix + "xi0_<i>_<j>..."`` with the factor
    indices otherwise) carry those of the remaining factors only. Harmonic
    factors keep their zero mode inside their harmonic array, as before.

    Returns a list of ``(name, Z, shapes)``: `Z` the tuple of factor indices at
    zero, `shapes` the shapes of the block's arrays (several for a multigrid
    factor, one per level; one otherwise).
    """
    kernel_idx = [i for i, sp in enumerate(spaces) if isinstance(sp, KernelSpace)]
    structured = [i for i, sp in enumerate(spaces) if sp.structured]
    if len(structured) > 1:
        raise NotImplementedError("at most one multigrid (structured) factor per correlated field")
    from itertools import combinations

    blocks = []
    subsets = [()] + [c for r in range(1, len(kernel_idx) + 1) for c in combinations(kernel_idx, r)]
    for Z in subsets:
        active = [i for i in range(len(spaces)) if i not in Z]
        shapes = [()]
        for i in active:
            shapes = [sh + esh for sh in shapes for esh in spaces[i].excitation_shapes]
        if Z == ():
            name = prefix + "xi"
        elif len(kernel_idx) == 1:
            name = prefix + "xi0"
        else:
            name = prefix + "xi0_" + "_".join(str(i) for i in Z)
        blocks.append((name, Z, tuple(shapes)))
    return blocks


class CorrelatedField(Model):
    """Correlated field made by :meth:`CorrelatedFieldMaker.finalize`: the
    outer product of the factors `spaces` with the amplitude models
    `amplitudes`, ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)``.
    Harmonic factors are applied by their harmonic transforms, kernel-based
    factors by their Gaussian process models `gps` with the covariance tables
    of their `kernels` (:class:`SpectralKernel` of the amplitude). The
    excitations are the blocks of :func:`excitation_blocks` (`blocks`): the
    main block and one block per combination of kernel-based factors at their
    zero mode. Spaces, amplitudes, kernels and GP models are dynamic pytree
    children, so pass the model into jitted functions as an argument (as
    NIFTy's optimizers do) rather than closing over it.
    """

    spaces: tuple = dataclasses.field(metadata=dict(static=False))
    amplitudes: tuple = dataclasses.field(metadata=dict(static=False))
    kernels: tuple = dataclasses.field(metadata=dict(static=False))
    gps: tuple = dataclasses.field(metadata=dict(static=False))

    def __init__(self, spaces, amplitudes, kernels, gps, zeromode, offset_mean, prefix, *, domain, init, mesh=None):
        self.spaces = tuple(spaces)
        self.amplitudes = tuple(amplitudes)
        self.kernels = tuple(kernels)
        self.gps = tuple(gps)
        self.zeromode = zeromode
        self.offset_mean = offset_mean
        self.prefix = str(prefix)
        self.blocks = tuple(excitation_blocks(self.spaces, self.prefix))
        self.xi_name = self.blocks[0][0]
        self.mesh = mesh
        if mesh is not None:  # excitations are drawn straight onto the mesh
            shardings = self.shardings
            init = {k: (partial(_init_sharded, v, shardings[k]) if k in shardings else v) for k, v in dict(init).items()}
        super().__init__(domain=domain, init=init)

    @property
    def shardings(self):
        """With a mesh: the sharding of every excitation block (a list for the
        per-level arrays of a multigrid factor); blocks without the sharded
        factor's axis and the amplitude parameters are replicated."""
        if self.mesh is None:
            return {}
        ax = self.mesh.axis_names[0]
        rep = NamedSharding(self.mesh, P())
        out = {}
        for name, Z, shapes in self.blocks:
            pos = 0
            specs = None
            for i, (space, gp) in enumerate(zip(self.spaces, self.gps)):
                if gp is not None and getattr(gp, "mesh", None) is not None and i not in Z:
                    gp_shardings = gp.shardings  # per excitation array of the factor
                    specs = [
                        NamedSharding(self.mesh, P(*([None] * pos), ax, *([None] * (len(sh) - pos - 1)))) if s is not None and s.spec and s.spec[0] is not None else rep
                        for s, sh in zip(gp_shardings, shapes)
                    ]
                    break
                pos += 0 if i in Z else len(space.excitation_shape) if gp is None else len(space.excitation_shapes[0])
            if specs is None:
                specs = [rep] * len(shapes)
            out[name] = specs[0] if len(shapes) == 1 else specs
        return out

    def shard(self, x):
        """Places the excitation blocks of a parameter dict on the mesh."""
        x = dict(x)
        for name, s in self.shardings.items():
            v = x[name]
            x[name] = [jax.device_put(np.asarray(a), si) for a, si in zip(v, s)] if isinstance(s, list) else jax.device_put(np.asarray(v), s)
        return x

    @property
    def sharding(self):
        """With a mesh: the sharding of the field (leading kernel axis on the mesh)."""
        if self.mesh is None:
            return None
        ax = self.mesh.axis_names[0]
        pos = 0
        for space, gp in zip(self.spaces, self.gps):
            if gp is not None and getattr(gp, "mesh", None) is not None:
                return NamedSharding(self.mesh, P(*([None] * pos), ax, *([None] * (len(self.target_shape) - pos - 1))))
            pos += len(space.shape)
        return NamedSharding(self.mesh, P())

    @property
    def target_shape(self):
        return sum((tuple(space.shape) for space in self.spaces), ())

    @property
    def target_grids(self):
        """Grids of the amplitude models."""
        return tuple(space.grid for space in self.spaces)

    @property
    def normalized_amplitudes(self):
        """Amplitudes with their non-zero modes divided by the zero mode."""

        def _mk_normed_amp(amp):  # Avoid late binding
            def normed_amplitude(p):
                return amp(p).at[1:].mul(1.0 / self.zeromode(p))

            return normed_amplitude

        return tuple(_mk_normed_amp(amp) for amp in self.amplitudes)

    def _apply_block(self, x, Z, amps, covs, zeromode):
        """The block's contribution: the factors not in `Z` applied along its
        axes, the factors in `Z` entering as the constant one (axes of size 1,
        broadcast by the sum over the blocks)."""
        pos = 0
        for i, (space, gp) in enumerate(zip(self.spaces, self.gps)):
            if i in Z:
                ins = lambda a: a.reshape(a.shape[:pos] + (1,) * len(space.shape) + a.shape[pos:])  # noqa: E731
                x = [ins(a) for a in x] if isinstance(x, list) else ins(x)
            elif gp is None:
                axes = tuple(range(pos, pos + len(space.excitation_shape)))
                f = lambda a: _apply_harmonic(space, a, axes, amps[i], zeromode)  # noqa: E731
                x = [f(a) for a in x] if isinstance(x, list) else f(x)
            else:
                x = _apply_kernel(gp, x, pos, covs[i], zeromode)
            pos += len(space.shape)
        return x

    def __call__(self, p):
        zeromode = self.zeromode(p)
        amps = [amp(p) for amp in self.amplitudes]
        covs = [None if k is None else k(p) for k in self.kernels]
        total = 0.0
        for name, Z, _ in self.blocks:
            total = total + self._apply_block(p[name], Z, amps, covs, zeromode)
        return self.offset_mean + zeromode * total


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

    def finalize(self, *, mesh=None) -> CorrelatedField:
        """Finishes off the model construction process and returns the
        constructed operator.

        With `mesh` (a one-dimensional ``jax.sharding.Mesh``) the kernel-based
        factor is generated on the mesh, sharded along its leading axis in
        contiguous blocks (:class:`~nifty.re.gaussian_process.ICRGP` and
        :class:`~nifty.re.gaussian_process.GraphGP` with ``mesh=``; graphgp
        graphs are built on the devices), and the excitations of the field are
        sharded alike (``shardings``, ``shard``, ``init`` of the returned
        field). If the points of a :class:`GraphGPSpace` are already a sharded
        ``jax.Array``, their mesh is used when `mesh` is None.

        The field is the outer product of the factors added with
        :meth:`add_fluctuations` (or :meth:`add_fluctuations_matern`),
        ``offset_mean + zeromode * prod_i (1 + a_i / zeromode)``, applied to a
        single array of excitations ``prefix + "xi"`` whose shape is the
        concatenation of the factors' excitation shapes (see :class:`Space`);
        the zero mode of a kernel-based factor is the leading entry along its
        axis. Kernel-based factors are generated by the Gaussian process
        models (:class:`~nifty.re.gaussian_process.ICRGP`,
        :class:`~nifty.re.gaussian_process.GraphGP`) built here from the
        spaces with a :class:`SpectralKernel` of their amplitudes. The
        returned :class:`CorrelatedField` has the attributes ``spaces``,
        ``amplitudes``, ``kernels``, ``gps``, ``target_grids`` (grids of the
        amplitude models) and ``normalized_amplitudes``.
        """
        if len(self._spaces) == 0:
            raise ValueError("add fluctuations before finalizing the model")
        # TODO: actually account for the dtype here
        for name, _, shapes in excitation_blocks(self._spaces, self._prefix):
            swd = [ShapeWithDtype(sh) for sh in shapes]
            self._parameter_tree[name] = swd[0] if len(swd) == 1 else swd
        if mesh is None:
            for i, space in enumerate(self._spaces):
                m = getattr(space, "mesh", None)
                if m is not None:
                    mesh = m
                    logger.info(
                        f"CorrelatedFieldMaker.finalize: mesh inferred from the sharded points of factor {i} "
                        f"({type(space).__name__}): {mesh.shape[mesh.axis_names[0]]} devices along '{mesh.axis_names[0]}'; the field is generated on this mesh"
                    )
                    break
        if mesh is not None and len(mesh.axis_names) != 1:
            raise ValueError("mesh must be one-dimensional")
        kernels, gps = [], []
        for space, amp in zip(self._spaces, self._fluctuations):
            if isinstance(space, KernelSpace):
                kernel = space.spectral_kernel(amp)
                kernels.append(kernel)
                gps.append(build_gp(space, kernel, mesh=mesh))
            else:
                kernels.append(None)
                gps.append(None)

        init = {
            k: partial(random_like, primals=v) for k, v in self._parameter_tree.items()
        }
        return CorrelatedField(
            self._spaces,
            self._fluctuations,
            kernels,
            gps,
            self.azm,
            self._offset_mean,
            self._prefix,
            domain=self._parameter_tree.copy(),
            init=init,
            mesh=mesh,
        )
