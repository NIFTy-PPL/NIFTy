# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank

import operator
from collections import namedtuple
from collections.abc import Mapping
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from jax import numpy as jnp
from jax import vmap

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
    um = jnp.unique(m_length)
    tol = uniqueness_rtol * um[-1]
    um = um[jnp.diff(jnp.append(um, 2 * um[-1])) > tol]
    # Group modes based on their length and store the result as power
    # distributor
    binbounds = 0.5 * (um[:-1] + um[1:])
    m_length_idx = jnp.searchsorted(binbounds, m_length)
    m_count = jnp.bincount(m_length_idx.ravel(), minlength=um.size)
    if jnp.any(m_count == 0) or um.shape != m_count.shape:
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
    mspc_distances = 1.0 / (jnp.array(shape) * jnp.array(distances))
    m_length = jnp.arange(shape[0], dtype=jnp.float64)
    m_length = jnp.minimum(m_length, shape[0] - m_length) * mspc_distances[0]
    if len(shape) != 1:
        m_length *= m_length
        for i in range(1, len(shape)):
            tmp = jnp.arange(shape[i], dtype=jnp.float64)
            tmp = jnp.minimum(tmp, shape[i] - tmp) * mspc_distances[i]
            tmp *= tmp
            m_length = jnp.expand_dims(m_length, axis=-1) + tmp
        m_length = jnp.sqrt(m_length)

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
    um = um.at[1:].set(jnp.log(um[1:]))
    um = um.at[1:].add(-um[1])
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
        distances = tuple(np.broadcast_to(distances, jnp.shape(shape)))

        totvol = jnp.prod(jnp.array(shape) * jnp.array(distances))
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
        (m_length_idx, m_length, m_count), (lmax, mmax, size) = (
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


def matern_amplitude(
    grid,
    scale: Callable,
    cutoff: Callable,
    loglogslope: Callable,
    renormalize_amplitude: bool,
    prefix: str = "",
    kind: str = "amplitude",
) -> Model:
    """Constructs a function computing the amplitude of a Matérn-kernel
    power spectrum.

    See
    :class:`nifty8.re.correlated_field.CorrelatedFieldMaker.add_fluctuations
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
    totvol = grid.total_volume
    mode_lengths = grid.harmonic_grid.mode_lengths
    mode_multiplicity = grid.harmonic_grid.mode_multiplicity

    scale = WrappedCall(scale, name=prefix + "scale")
    ptree = scale.domain.copy()
    cutoff = WrappedCall(cutoff, name=prefix + "cutoff")
    ptree.update(cutoff.domain)
    loglogslope = WrappedCall(loglogslope, name=prefix + "loglogslope")
    ptree.update(loglogslope.domain)

    def correlate(primals: Mapping) -> jnp.ndarray:
        scl = scale(primals)
        ctf = cutoff(primals)
        slp = loglogslope(primals)

        ln_spectrum = 0.25 * slp * jnp.log1p((mode_lengths / ctf) ** 2)

        spectrum = jnp.exp(ln_spectrum)

        norm = 1.0
        if renormalize_amplitude:
            logger.warning("Renormalize amplidude is not yet tested!")
            if kind.lower() == "amplitude":
                norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:] ** 4))
            elif kind.lower() == "power":
                norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:] ** 2))
            norm /= jnp.sqrt(totvol)  # Due to integral in harmonic space
        spectrum = scl * (jnp.sqrt(totvol) / norm) * spectrum
        spectrum = spectrum.at[0].set(totvol)
        if kind.lower() == "power":
            spectrum = jnp.sqrt(spectrum)
        elif kind.lower() != "amplitude":
            raise ValueError(f"invalid kind specified {kind!r}")
        return spectrum

    return Model(correlate, domain=ptree, init=partial(random_like, primals=ptree))


def non_parametric_amplitude(
    grid,
    fluctuations: Callable,
    loglogavgslope: Callable,
    flexibility: Optional[Callable] = None,
    asperity: Optional[Callable] = None,
    prefix: str = "",
    kind: str = "amplitude",
) -> Model:
    """Constructs a function computing the amplitude of a non-parametric power
    spectrum

    See
    :class:`nifty8.re.correlated_field.CorrelatedFieldMaker.add_fluctuations`
    for more details on the parameters.

    See also
    --------
    `Variable structures in M87* from space, time and frequency resolved
    interferometry`, Arras, Philipp and Frank, Philipp and Haim, Philipp
    and Knollmüller, Jakob and Leike, Reimar and Reinecke, Martin and
    Enßlin, Torsten, `<https://arxiv.org/abs/2002.05218>`_
    `<http://dx.doi.org/10.1038/s41550-021-01548-0>`_
    """
    totvol = grid.total_volume
    rel_log_mode_len = grid.harmonic_grid.relative_log_mode_lengths
    mode_multiplicity = grid.harmonic_grid.mode_multiplicity
    log_vol = grid.harmonic_grid.log_volume

    fluctuations = WrappedCall(
        fluctuations, name=prefix + "fluctuations", white_init=True
    )
    ptree = fluctuations.domain.copy()
    loglogavgslope = WrappedCall(
        loglogavgslope, name=prefix + "loglogavgslope", white_init=True
    )
    ptree.update(loglogavgslope.domain)
    if flexibility is not None and (log_vol.size > 0):
        flexibility = WrappedCall(
            flexibility, name=prefix + "flexibility", white_init=True
        )
        assert log_vol is not None
        assert rel_log_mode_len.ndim == log_vol.ndim == 1
        if asperity is not None:
            asperity = WrappedCall(asperity, name=prefix + "asperity", white_init=True)
        deviations = IntegratedWienerProcess(
            jnp.zeros((2,)),
            flexibility,
            log_vol,
            name=prefix + "spectrum",
            asperity=asperity,
        )
        ptree.update(deviations.domain)
    else:
        deviations = None

    def correlate(primals: Mapping) -> jnp.ndarray:
        flu = fluctuations(primals)
        slope = loglogavgslope(primals)
        slope *= rel_log_mode_len
        ln_spectrum = slope

        if deviations is not None:
            twolog = deviations(primals)
            # Prepend zeromode
            twolog = jnp.concatenate((jnp.zeros((1,)), twolog[:, 0]))
            ln_spectrum += _remove_slope(rel_log_mode_len, twolog)

        # Exponentiate and norm the power spectrum
        spectrum = jnp.exp(ln_spectrum)
        # Take the sqrt of the integral of the slope w/o fluctuations and
        # zero-mode while taking into account the multiplicity of each mode
        if kind.lower() == "amplitude":
            norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:] ** 2))
            norm /= jnp.sqrt(totvol)  # Due to integral in harmonic space
            amplitude = flu * (jnp.sqrt(totvol) / norm) * spectrum
        elif kind.lower() == "power":
            norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:]))
            norm /= jnp.sqrt(totvol)  # Due to integral in harmonic space
            amplitude = flu * (jnp.sqrt(totvol) / norm) * jnp.sqrt(spectrum)
        else:
            raise ValueError(f"invalid kind specified {kind!r}")
        amplitude = amplitude.at[0].set(totvol)
        return amplitude

    return Model(correlate, domain=ptree, init=partial(random_like, primals=ptree))


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
        self._target_grids = []
        self._parameter_tree = {}

        self._prefix = prefix

    def add_fluctuations(
        self,
        shape: Union[tuple, int],
        distances: Union[tuple, float],
        fluctuations: Union[tuple, Callable],
        loglogavgslope: Union[tuple, Callable],
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
        shape : tuple of int
            Shape of the position space grid
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
        grid = make_grid(shape, distances, harmonic_type)

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

        npa = non_parametric_amplitude(
            grid=grid,
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
        )
        self._fluctuations.append(npa)
        self._target_grids.append(grid)
        self._parameter_tree.update(npa.domain)

    def add_fluctuations_matern(
        self,
        shape: Union[tuple, int],
        distances: Union[tuple, float],
        scale: Union[tuple, Callable],
        cutoff: Union[tuple, Callable],
        loglogslope: Union[tuple, Callable],
        renormalize_amplitude: bool,
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
        shape : tuple of int
            Shape of the position space grid.
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
        grid = make_grid(shape, distances, harmonic_type)

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

        ma = matern_amplitude(
            grid=grid,
            scale=scale,
            cutoff=cutoff,
            loglogslope=loglogslope,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
            renormalize_amplitude=renormalize_amplitude,
        )
        self._fluctuations.append(ma)
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
        """
        harmonic_transforms = []
        excitation_shape = ()
        for sgrid in self._target_grids:
            sub_shp = None
            sub_shp = sgrid.harmonic_grid.shape
            excitation_shape += sub_shp
            n = len(excitation_shape)
            harmonic_dvol = 1.0 / sgrid.total_volume
            if isinstance(sgrid, RegularCartesianGrid):
                axes = tuple(range(n - len(sub_shp), n))
                # TODO: Generalize to complex
                trafo = partial(hartley, axes=axes)
            elif isinstance(sgrid, HEALPixGrid):
                axis = len(excitation_shape) - 1
                trafo = get_sht(
                    nside=sgrid.nside,
                    axis=axis,
                    lmax=sgrid.harmonic_grid.lmax,
                    mmax=sgrid.harmonic_grid.mmax,
                    nthreads=1,
                )
            harmonic_transforms.append((harmonic_dvol, trafo))

        # Register the parameters for the excitations in harmonic space
        # TODO: actually account for the dtype here
        pfx = self._prefix + "xi"
        self._parameter_tree[pfx] = ShapeWithDtype(excitation_shape)

        def outer_harmonic_transform(p):
            harmonic_dvol, ht = harmonic_transforms[0]
            outer = harmonic_dvol * ht(p)
            for harmonic_dvol, ht in harmonic_transforms[1:]:
                outer = harmonic_dvol * ht(outer)
            return outer

        def _mk_expanded_amp(amp, sub_dom):  # Avoid late binding
            def expanded_amp(p):
                return amp(p)[sub_dom.harmonic_grid.power_distributor]

            return expanded_amp

        expanded_amplitudes = []
        namps = self.get_normalized_amplitudes()
        for amp, sgrid in zip(namps, self._target_grids):
            expanded_amplitudes.append(_mk_expanded_amp(amp, sgrid))

        def outer_amplitude(p):
            outer = expanded_amplitudes[0](p)
            for amp in expanded_amplitudes[1:]:
                # NOTE, the order is important here and must match with the
                # excitations
                # TODO, use functions instead and utilize numpy's casting
                outer = jnp.tensordot(outer, amp(p), axes=0)
            return outer

        def correlated_field(p):
            ea = outer_amplitude(p)
            cf_h = self.azm(p) * ea * p[self._prefix + "xi"]
            return self._offset_mean + outer_harmonic_transform(cf_h)

        init = {
            k: partial(random_like, primals=v) for k, v in self._parameter_tree.items()
        }
        cf = Model(correlated_field, domain=self._parameter_tree.copy(), init=init)
        cf.normalized_amplitudes = namps
        cf.target_grids = tuple(self._target_grids)
        return cf
