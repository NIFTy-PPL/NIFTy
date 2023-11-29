# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections.abc import Mapping
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from jax import numpy as jnp

from .logger import logger
from .misc import wrap
from .model import Model, WrappedCall
from .num import lognormal_prior, normal_prior
from .tree_math import ShapeWithDtype, random_like


def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes=axes)
    return tmp.real + tmp.imag


def get_fourier_mode_distributor(
    shape: Union[tuple, int], distances: Union[tuple, float]
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
        distribute power from a power-space to the full harmonic domain.
    unique_mode_length : jnp.ndarray
        Unique length of Fourier modes.
    mode_multiplicity : jnp.ndarray
        Multiplicity for each unique Fourier mode length.
    """
    shape = (shape, ) if isinstance(shape, int) else tuple(shape)

    # Compute length of modes
    mspc_distances = 1. / (jnp.array(shape) * jnp.array(distances))
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

    # Construct an array of unique mode lengths
    uniqueness_rtol = 1e-12
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


def _make_domain(shape, distances, harmonic_domain_type):
    """Creates the domain attributes for the amplitude model"""
    shape = (shape, ) if isinstance(shape, int) else tuple(shape)
    distances = tuple(np.broadcast_to(distances, jnp.shape(shape)))

    totvol = jnp.prod(jnp.array(shape) * jnp.array(distances))
    # TODO: cache results such that only references are used afterwards
    domain = {
        "position_space_shape": shape,
        "position_space_total_volume": totvol,
        "position_space_distances": distances,
        "harmonic_domain_type": harmonic_domain_type.lower()
    }
    # Pre-compute lengths of modes and indices for distributing power
    if harmonic_domain_type.lower() == "fourier":
        domain["harmonic_space_shape"] = shape
        m_length_idx, um, m_count = get_fourier_mode_distributor(
            shape, distances
        )
        domain["power_distributor"] = m_length_idx
        domain["mode_multiplicity"] = m_count
        domain["mode_lengths"] = um

        um = um.at[1:].set(jnp.log(um[1:]))
        um = um.at[1:].add(-um[1])
        assert um[0] == 0.
        domain["relative_log_mode_lengths"] = um
        log_vol = um[2:] - um[1:-1]
        assert um.shape[0] - 2 == log_vol.shape[0]
        domain["log_volume"] = log_vol
    else:
        ve = f"invalid `harmonic_domain_type` {harmonic_domain_type!r}"
        raise ValueError(ve)
    return domain


def _twolog_integrate(log_vol, x):
    # Map the space to the one for the relative log-modes, i.e. pad the space
    # of the log volume
    twolog = jnp.empty((2 + log_vol.shape[0], ))
    twolog = twolog.at[0].set(0.)
    twolog = twolog.at[1].set(0.)

    twolog = twolog.at[2:].set(jnp.cumsum(x[1], axis=0))
    twolog = twolog.at[2:].set(
        (twolog[2:] + twolog[1:-1]) / 2. * log_vol + x[0]
    )
    twolog = twolog.at[2:].set(jnp.cumsum(twolog[2:], axis=0))
    return twolog


def _remove_slope(rel_log_mode_dist, x):
    sc = rel_log_mode_dist / rel_log_mode_dist[-1]
    return x - x[-1] * sc


def matern_amplitude(
    domain: Mapping,
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
    load vs. patient's age`, Guardiani, Matteo and Frank, Kostić Andrija
    and Edenhofer, Gordian and Roth, Jakob and Uhlmann, Berit and
    Enßlin, Torsten, `<https://arxiv.org/abs/2105.13483>`_
    `<https://doi.org/10.1371/journal.pone.0275011>`_
    """
    totvol = domain.get("position_space_total_volume", 1.)
    mode_lengths = domain["mode_lengths"]
    mode_multiplicity = domain["mode_multiplicity"]

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

        ln_spectrum = 0.25 * slp * jnp.log1p((mode_lengths / ctf)**2)

        spectrum = jnp.exp(ln_spectrum)

        norm = 1.
        if renormalize_amplitude:
            logger.warning("Renormalize amplidude is not yet tested!")
            if kind.lower() == "amplitude":
                norm = jnp.sqrt(
                    jnp.sum(mode_multiplicity[1:] * spectrum[1:]**4)
                )
            elif kind.lower() == "power":
                norm = jnp.sqrt(
                    jnp.sum(mode_multiplicity[1:] * spectrum[1:]**2)
                )
            norm /= jnp.sqrt(totvol)  # Due to integral in harmonic space
        spectrum = scl * (jnp.sqrt(totvol) / norm) * spectrum
        spectrum = spectrum.at[0].set(totvol)
        if kind.lower() == "power":
            spectrum = jnp.sqrt(spectrum)
        elif kind.lower() != "amplitude":
            raise ValueError(f"invalid kind specified {kind!r}")
        return spectrum

    return Model(
        correlate, domain=ptree, init=partial(random_like, primals=ptree)
    )


def non_parametric_amplitude(
    domain: Mapping,
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
    totvol = domain.get("position_space_total_volume", 1.)
    rel_log_mode_len = domain["relative_log_mode_lengths"]
    mode_multiplicity = domain["mode_multiplicity"]
    log_vol = domain.get("log_volume")

    fluctuations = WrappedCall(fluctuations, name=prefix + "fluctuations")
    ptree = fluctuations.domain.copy()
    loglogavgslope = WrappedCall(loglogavgslope, name=prefix + "loglogavgslope")
    ptree.update(loglogavgslope.domain)
    if flexibility is not None:
        flexibility = WrappedCall(flexibility, name=prefix + "flexibility")
        ptree.update(flexibility.domain)
        # Register the parameters for the spectrum
        assert log_vol is not None
        assert rel_log_mode_len.ndim == log_vol.ndim == 1
        ptree.update(
            {prefix + "spectrum": ShapeWithDtype((2, ) + log_vol.shape)}
        )
    if asperity is not None:
        asperity = WrappedCall(asperity, name=prefix + "asperity")
        ptree.update(asperity.domain)

    def correlate(primals: Mapping) -> jnp.ndarray:
        flu = fluctuations(primals)
        slope = loglogavgslope(primals)
        slope *= rel_log_mode_len
        ln_spectrum = slope

        if flexibility is not None:
            assert log_vol is not None
            xi_spc = primals[prefix + "spectrum"]
            flx = flexibility(primals)
            sig_flx = flx * jnp.sqrt(log_vol)
            sig_flx = jnp.broadcast_to(sig_flx, (2, ) + log_vol.shape)

            if asperity is None:
                shift = jnp.stack(
                    (log_vol / jnp.sqrt(12.), jnp.ones_like(log_vol)), axis=0
                )
                asp = shift * sig_flx * xi_spc
            else:
                asp = asperity(primals)
                shift = jnp.stack(
                    (log_vol**2 / 12., jnp.ones_like(log_vol)), axis=0
                )
                sig_asp = jnp.broadcast_to(
                    jnp.array([[asp], [0.]]), shift.shape
                )
                asp = jnp.sqrt(shift + sig_asp) * sig_flx * xi_spc

            twolog = _twolog_integrate(log_vol, asp)
            wo_slope = _remove_slope(rel_log_mode_len, twolog)
            ln_spectrum += wo_slope

        # Exponentiate and norm the power spectrum
        spectrum = jnp.exp(ln_spectrum)
        # Take the sqrt of the integral of the slope w/o fluctuations and
        # zero-mode while taking into account the multiplicity of each mode
        if kind.lower() == "amplitude":
            norm = jnp.sqrt(jnp.sum(mode_multiplicity[1:] * spectrum[1:]**2))
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

    return Model(
        correlate, domain=ptree, init=partial(random_like, primals=ptree)
    )


class CorrelatedFieldMaker():
    """Construction helper for hierarchical correlated field models.

    The correlated field models are parametrized by creating square roots of
    power spectrum operators ("amplitudes") via calls to
    :func:`add_fluctuations*` that act on the targeted field subdomains.
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
            Prefix to the names of the domains of the cf operator to be made.
            This determines the names of the operator domain.
        """
        self._azm = None
        self._offset_mean = None
        self._fluctuations = []
        self._target_subdomains = []
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
        harmonic_domain_type: str = "fourier",
        non_parametric_kind: str = "amplitude",
    ):
        """Adds a correlation structure to the to-be-made field.

        Correlations are described by their power spectrum and the subdomain on
        which they apply.

        Multiple calls to `add_fluctuations` are possible, in which case
        the constructed field will have the outer product of the individual
        power spectra as its global power spectrum.

        The parameters `fluctuations`, `flexibility`, `asperity` and
        `loglogavgslope` configure either the amplitude or the power
        spectrum model used on the target field subdomain of type
        `harmonic_domain_type`. It is assembled as the sum of a power
        law component (linear slope in log-log
        amplitude-frequency-space), a smooth varying component
        (integrated Wiener process) and a ragged component
        (un-integrated Wiener process).

        Parameters
        ----------
        shape : tuple of int
            Shape of the position space domain
        distances : tuple of float or float
            Distances in the position space domain
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
            Prefix of the power spectrum parameter domain names
        harmonic_domain_type : str
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
        domain = _make_domain(shape, distances, harmonic_domain_type)

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
            domain=domain,
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
        )
        self._fluctuations.append(npa)
        self._target_subdomains.append(domain)
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
        harmonic_domain_type: str = "fourier",
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
            Shape of the position space domain.
        distances : tuple of float or float
            Distances in the position space domain.
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
            Prefix of the power spectrum parameter domain names.
        harmonic_domain_type : str
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
        domain = _make_domain(shape, distances, harmonic_domain_type)

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
            domain=domain,
            scale=scale,
            cutoff=cutoff,
            loglogslope=loglogslope,
            prefix=self._prefix + prefix,
            kind=non_parametric_kind,
            renormalize_amplitude=renormalize_amplitude,
        )
        self._fluctuations.append(ma)
        self._target_subdomains.append(domain)
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
                return amp(p).at[1:].mul(1. / self.azm(p))

            return normed_amplitude

        return tuple(_mk_normed_amp(amp) for amp in self._fluctuations)

    @property
    def amplitude(self) -> Callable:
        """Returns the added fluctuation, i.e. un-normalized amplitude"""
        if len(self._fluctuations) > 1:
            s = (
                'If more than one spectrum is present in the model,'
                ' no unique set of amplitudes exist because only the'
                ' relative scale is determined.'
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
            return amp(p)**2

        return power

    def finalize(self) -> Model:
        """Finishes off the model construction process and returns the
        constructed operator.
        """
        harmonic_transforms = []
        excitation_shape = ()
        for sub_dom in self._target_subdomains:
            sub_shp = None
            sub_shp = sub_dom["harmonic_space_shape"]
            excitation_shape += sub_shp
            n = len(excitation_shape)
            axes = tuple(range(n - len(sub_shp), n))

            # TODO: Generalize to complex
            harmonic_dvol = 1. / sub_dom["position_space_total_volume"]
            harmonic_transforms.append(
                (harmonic_dvol, partial(hartley, axes=axes))
            )
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
                return amp(p)[sub_dom["power_distributor"]]

            return expanded_amp

        expanded_amplitudes = []
        namps = self.get_normalized_amplitudes()
        for amp, sub_dom in zip(namps, self._target_subdomains):
            expanded_amplitudes.append(_mk_expanded_amp(amp, sub_dom))

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
            k: partial(random_like, primals=v)
            for k, v in self._parameter_tree.items()
        }
        cf = Model(
            correlated_field, domain=self._parameter_tree.copy(), init=init
        )
        cf.normalized_amplitudes = namps
        return cf
