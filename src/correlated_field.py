from typing import Union, Optional
from collections.abc import Mapping

import sys
from jax import numpy as np
from jax import jit
from .sugar import ducktape
from .operator import ShapeWithDtype, normal_prior, lognormal_prior


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
    mode_length_idx : np.ndarray
        Index in power-space for every mode in harmonic-space. Can be used to
        distribute power from a power-space to the full harmonic domain.
    unique_mode_length : np.ndarray
        Unique length of Fourier modes.
    mode_multiplicity : np.ndarray
        Multiplicity for each unique Fourier mode length.
    """
    shape = tuple(shape)

    # Compute length of modes
    mspc_distances = 1. / (np.array(shape) * np.array(distances))
    m_length = np.arange(shape[0], dtype=np.float64)
    m_length = np.minimum(m_length, shape[0] - m_length) * mspc_distances[0]
    if len(shape) != 1:
        m_length *= m_length
        for i in range(1, len(shape)):
            tmp = np.arange(shape[i], dtype=np.float64)
            tmp = np.minimum(tmp, shape[i] - tmp) * mspc_distances[i]
            tmp *= tmp
            m_length = np.expand_dims(m_length, axis=-1) + tmp
        m_length = np.sqrt(m_length)

    # Construct an array of unique mode lengths
    uniqueness_rtol = 1e-12
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


def _twolog_integrate(log_vol, x):
    # Map the space to the one for the relative log-modes, i.e. pad the space
    # of the log volume
    twolog = np.empty((2 + log_vol.shape[0], ))
    twolog = twolog.at[0].set(0.)
    twolog = twolog.at[1].set(0.)

    twolog = twolog.at[2:].set(np.cumsum(x[1], axis=0))
    twolog = twolog.at[2:].set(
        (twolog[2:] + twolog[1:-1]) / 2. * log_vol + x[0]
    )
    twolog = twolog.at[2:].set(np.cumsum(twolog[2:], axis=0))
    return twolog


def _remove_slope(rel_log_mode_dist, x):
    sc = rel_log_mode_dist / rel_log_mode_dist[-1]
    return x - x[-1] * sc


def non_parametric_amplitude(
    domain: Mapping,
    fluctuations: callable,
    loglogavgslope: callable,
    flexibility: Optional[callable] = None,
    asperity: Optional[callable] = None,
    prefix: str = ""
):
    totvol = domain.get("position_space_total_volume", 1.)
    rel_log_mode_len = domain["relative_log_mode_lengths"]
    mode_multiplicity = domain["mode_multiplicity"]
    log_vol = domain.get("log_volume")

    ptree = {}
    fluctuations = ducktape(fluctuations, prefix + "_fluctuations")
    ptree[prefix + "_fluctuations"] = ShapeWithDtype(())
    loglogavgslope = ducktape(loglogavgslope, prefix + "_loglogavgslope")
    ptree[prefix + "_loglogavgslope"] = ShapeWithDtype(())
    if flexibility is not None:
        flexibility = ducktape(flexibility, prefix + "_flexibility")
        ptree[prefix + "_flexibility"] = ShapeWithDtype(())
        # Register the parameters for the spectrum
        assert rel_log_mode_len.ndim == log_vol.ndim == 1
        ptree[prefix + "_spectrum"] = ShapeWithDtype((2, ) + log_vol.shape)
    else:
        flexibility = None
    if asperity is not None:
        asperity = ducktape(asperity, prefix + "_asperity")
        ptree[prefix + "_asperity"] = ShapeWithDtype(())
    else:
        asperity = None

    def correlate(primals: Mapping) -> np.ndarray:
        flu = fluctuations(primals)
        slope = loglogavgslope(primals)
        slope *= rel_log_mode_len
        ln_amplitude = slope

        if flexibility is not None:
            xi_spc = primals.get(prefix + "_spectrum")
            flx = flexibility(primals)
            sig_flx = flx * np.sqrt(log_vol)
            sig_flx = np.broadcast_to(sig_flx, (2, ) + log_vol.shape)

            if asperity is None:
                shift = np.stack(
                    (log_vol / np.sqrt(12.), np.ones_like(log_vol)), axis=0
                )
                asp = shift * sig_flx * xi_spc
            else:
                asp = asperity(primals)
                shift = np.stack(
                    (log_vol**2 / 12., np.ones_like(log_vol)), axis=0
                )
                sig_asp = np.broadcast_to(np.array([[asp], [0.]]), shift.shape)
                asp = np.sqrt(shift + sig_asp) * sig_flx * xi_spc

            twolog = _twolog_integrate(log_vol, asp)
            wo_slope = _remove_slope(rel_log_mode_len, twolog)
            ln_amplitude += wo_slope

        # Exponentiate and norm the power spectrum
        amplitude = np.exp(ln_amplitude)
        # Take the sqrt of the integral of the slope w/o fluctuations and
        # zero-mode while taking into account the multiplicity of each mode
        norm = np.sqrt(np.sum(mode_multiplicity[1:] * amplitude[1:]**2))
        norm /= np.sqrt(totvol)  # Due to integral in harmonic space
        amplitude *= flu * (np.sqrt(totvol) / norm)
        amplitude = amplitude.at[0].set(totvol)
        return amplitude

    return correlate, ptree


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

    See the methods initialization, :func:`add_fluctuations` and
    :func:`finalize` for further usage information."""
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
        fluctuations: Union[tuple, callable],
        loglogavgslope: Union[tuple, callable],
        flexibility: Union[tuple, callable, None] = None,
        asperity: Union[tuple, callable, None] = None,
        prefix: str = "",
        harmonic_domain_type: str = "fourier",
    ):
        shape = tuple(shape)
        distances = tuple(np.broadcast_to(distances, np.shape(shape)))
        totvol = np.prod(np.array(shape) * np.array(distances))

        # Pre-compute lengths of modes and indices for distributing power
        # TODO: cache results such that only references are used afterwards
        domain = {
            "position_space_shape": shape,
            "position_space_total_volume": totvol,
            "position_space_distances": distances,
            "harmonic_domain_type": harmonic_domain_type.lower()
        }
        if harmonic_domain_type.lower() == "fourier":
            domain["harmonic_space_shape"] = shape
            m_length_idx, um, m_count = get_fourier_mode_distributor(
                shape, distances
            )
            domain["power_distributor"] = m_length_idx
            domain["mode_multiplicity"] = m_count

            # Transform the unique modes to log-space for the amplitude model
            um = um.at[1:].set(np.log(um[1:]))
            um = um.at[1:].add(-um[1])
            assert um[0] == 0.
            domain["relative_log_mode_lengths"] = um
            log_vol = um[2:] - um[1:-1]
            assert um.shape[0] - 2 == log_vol.shape[0]
            domain["log_volume"] = log_vol
        else:
            ve = f"invalid `harmonic_domain_type` {harmonic_domain_type!r}"
            raise ValueError(ve)

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

        npa, ptree = non_parametric_amplitude(
            domain=domain,
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
            prefix=self._prefix + "_" + prefix
        )
        self._fluctuations.append(npa)
        self._target_subdomains.append(domain)
        self._parameter_tree.update(ptree)

    def set_amplitude_total_offset(
        self, offset_mean: float, offset_std: Union[tuple, callable]
    ):
        """Sets the zero-mode for the combined amplitude operator

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the correlated field to be made.
        offset_std : tuple of float or callable
            Mean standard deviation and standard deviation of the standard
            deviation of the offset. No, this is not a word duplication.
        """
        if self._offset_mean is not None and self._azm is not None:
            msg = "Overwriting the previous mean offset and zero-mode"
            print(msg, file=sys.stderr)

        self._offset_mean = offset_mean
        zm = offset_std
        if not callable(zm):
            if len(offset_std) != 2:
                raise TypeError
            zm = lognormal_prior(*offset_std)

        self._azm = ducktape(zm, self._prefix + "_zeromode")
        self._parameter_tree[self._prefix + "_zeromode"] = ShapeWithDtype(())

    @property
    def amplitude_total_offset(self):
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
    def fluctuations(self):
        return tuple(self._fluctuations)

    def get_normalized_amplitudes(self):
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
    def amplitude(self):
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

    def finalize(self):
        """Finishes model construction process and returns the constructed
        operator.
        """
        harmonic_transforms = []
        excitation_shape = ()
        for sub_dom in self._target_subdomains:
            sub_shp = None
            sub_shp = sub_dom["harmonic_space_shape"]
            excitation_shape += sub_shp
            n = len(excitation_shape)
            axes = tuple(range(n - len(sub_shp), n))

            # TODO: Generalize to complex; Add dtype to parameter_tree?
            def ht_axs(p, axes=axes):  # Avoid late binding
                return hartley(p, axes=axes)

            harmonic_transforms.append(ht_axs)
        # Register the parameters for the excitations in harmonic space
        # TODO: actually account for the dtype here
        pfx = self._prefix + "_excitations"
        self._parameter_tree[pfx] = ShapeWithDtype(excitation_shape)

        def outer_harmonic_transform(p):
            outer = harmonic_transforms[0](p)
            for ht in harmonic_transforms[1:]:
                outer = ht(outer)
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
                outer = np.tensordot(outer, amp(p), axes=0)
            return outer

        def correlated_field(p):
            ea = outer_amplitude(p)
            cf_h = self.azm(p) * ea * p.get(self._prefix + "_excitations")
            return self._offset_mean + outer_harmonic_transform(cf_h)

        return correlated_field, self._parameter_tree.copy()
