from collections.abc import Mapping

import sys
from jax import numpy as np
from jax import jit
from .sugar import ducktape
from .operator import normal_prior, lognormal_prior


def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes=axes)
    return tmp.real + tmp.imag


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
    domain,
    fluctuations,
    loglogavgslope,
    flexibility=None,
    asperity=None,
    prefix=""
):
    totvol = domain.get("position_space_total_volume", 1.)
    harmonic_dom_type = domain["harmonic_domain_type"].lower()
    rel_log_mode_len = domain["relative_log_mode_lengths"]
    log_vol = domain.get("log_volume")

    ptree = {}
    fluctuations = ducktape(fluctuations, prefix + "_fluctuations")
    ptree[prefix + "_fluctuations"] = ()
    loglogavgslope = ducktape(loglogavgslope, prefix + "_loglogavgslope")
    ptree[prefix + "_loglogavgslope"] = ()
    if flexibility is not None:
        flexibility = ducktape(flexibility, prefix + "_flexibility")
        ptree[prefix + "_flexibility"] = ()
        # Register the parameters for the spectrum
        assert rel_log_mode_len.ndim == log_vol.ndim == 1
        ptree[prefix + "_spectrum"] = (2, ) + log_vol.shape
    else:
        flexibility = None
    if asperity is not None:
        asperity = ducktape(asperity, prefix + "_asperity")
        ptree[prefix + "_asperity"] = ()
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
        if harmonic_dom_type == "fourier":
            # TODO: Properly distributed the power for arbitrary domains
            # Take the sqrt of the integral of the slope w/o fluctuations and
            # zero-mode while taking into account the multiplicity of each mode
            norm = np.sqrt(2 * np.sum(amplitude[1:]**2) - amplitude[-1]**2)
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
    def __init__(self, prefix):
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
        shape,
        distances,
        fluctuations,
        loglogavgslope,
        flexibility=None,
        asperity=None,
        prefix="",
        harmonic_domain_type="fourier",
    ):
        shape = tuple(shape)
        totvol = np.prod(np.array(shape) * np.array(distances))
        if len(shape) > 1:
            ve = "Multi-dimensional amplitude operator not implemented (yet)"
            raise ValueError(ve)

        # Pre-compute lengths of modes and indices for distributing power
        # TODO: cache results such that only references are used afterwards
        # TODO: compute indices for distributing power
        domain = {
            "position_space_shape": shape,
            "position_space_total_volume": totvol,
            "position_space_distances": distances,
            "harmonic_domain_type": harmonic_domain_type.lower()
        }
        if harmonic_domain_type.lower() == "fourier":
            if len(shape) != 1:
                nie = f"Unsupported length of `shape`: {len(shape)}"
                raise ValueError(nie)
            lm = np.arange(shape[0] / 2 + 1., dtype=float)
            lm = lm.at[1:].set(np.log(lm[1:]))
            lm = lm.at[1:].add(-lm[1])
            # NOTE, the volume doesn't matter for either
            # `relative_log_mode_lengths` nor `log_volume` as relative logarithmic
            # quantities are volume-free
            domain["relative_log_mode_lengths"] = lm
            if flexibility is not None:
                domain["log_volume"] = lm[2:] - lm[1:-1]
                assert lm.shape[0] - 2 == domain["log_volume"].shape[0]

            # Compute length of modes
            ksp_dist = 1. / (np.array(shape) * np.array(distances))
            k_length = np.arange(shape[0], dtype=np.float64)
            k_length = np.minimum(k_length, shape[0] - k_length) * distances[0]
            if len(shape) != 1:
                k_length *= k_length
                for i in range(1, len(shape)):
                    tmp = np.arange(shape[i], dtype=np.float64)
                    tmp = np.minimum(tmp, shape[i] - tmp) * distances[i]
                    tmp *= tmp
                    k_length = np.expand_dims(k_length, axis=-1) + tmp
                k_length = np.sqrt(k_length)

            # Construct an array of unique mode lengths
            u_k_length = np.unique(k_length)
            tol = 1e-12 * u_k_length[-1]
            u_k_length = u_k_length[
                np.diff(np.r_[u_k_length, 2 * u_k_length[-1]]) > tol]

            # Group modes based on their length and store the result
            binbounds = 0.5 * (u_k_length[:-1] + u_k_length[1:])
            k_array_index = np.searchsorted(binbounds, k_length)
            domain['power_distributor'] = k_array_index
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

    def set_amplitude_total_offset(self, offset_mean, offset_std):
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
        self._parameter_tree[self._prefix + "_zeromode"] = ()

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
            if sub_dom["harmonic_domain_type"].lower() == "fourier":
                sub_shp = sub_dom["position_space_shape"]
            excitation_shape += sub_shp
            n = len(excitation_shape)
            axes = tuple(range(n - len(sub_shp), n))

            # TODO: Generalize to complex; Add dtype to parameter_tree?
            def ht_axs(p, axes=axes):  # Avoid late binding
                return hartley(p, axes=axes)

            harmonic_transforms.append(ht_axs)
        # Register the parameters for the excitations in harmonic space
        self._parameter_tree[self._prefix + "_excitations"] = excitation_shape

        def outer_harmonic_transform(p):
            outer = harmonic_transforms[0](p)
            for ht in harmonic_transforms[1:]:
                outer = ht(outer)
            return outer

        def _mk_expanded_amp(amp, harmonic_domain_type):  # Avoid late binding
            if harmonic_domain_type == "fourier":
                # TODO: Properly distributed the power for arbitrary domains
                def expanded_amp(p):
                    amp_at_p = amp(p)
                    # Every mode appears exactly two times, first ascending
                    # then descending.
                    return np.concatenate((amp_at_p, amp_at_p[-2:0:-1]))

            return expanded_amp

        expanded_amplitudes = []
        for amp, sub_dom in zip(
            self.get_normalized_amplitudes(), self._target_subdomains
        ):
            h_dom_type = sub_dom["harmonic_domain_type"].lower()
            expanded_amplitudes.append(_mk_expanded_amp(amp, h_dom_type))

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
