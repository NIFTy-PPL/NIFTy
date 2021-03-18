from jax import numpy as np
from .operator import normal_prior, lognormal_prior


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


def _remove_slope(rel_log_modes, x):
    sc = rel_log_modes / rel_log_modes[-1]
    return x - x[-1] * sc


class Amplitude():
    def __init__(
        self,
        shape,
        zeromode,
        fluctuations,
        loglogavgslope,
        flexibility=None,
        asperity=None,
        prefix="",
        harmonic_domain_type="fourier",
    ):
        self._shape = tuple(shape)
        self._zm = lognormal_prior(*zeromode)
        self._flu = lognormal_prior(*fluctuations)
        self._slp = normal_prior(*loglogavgslope)
        self._flx = lognormal_prior(
            *flexibility
        ) if flexibility is not None else None
        self._asp = lognormal_prior(*asperity) if asperity is not None else None
        self._prefix = prefix
        self._harmonic_dom_type = harmonic_domain_type.lower()

        if len(self._shape) > 1:
            ve = "Multi-dimensional amplitude operator not implemented (yet)"
            raise ValueError(ve)

        if self._harmonic_dom_type == "fourier":
            self._rel_log_modes = []
            self._log_vol = []
            for dim in self._shape:
                # TODO: add volume
                lm = np.arange(dim / 2 + 1., dtype=float)
                lm = lm.at[1:].set(np.log(lm[1:]))
                lm = lm.at[1:].add(-lm[1])
                # NOTE, the volume doesn't matter for either
                # `self._rel_log_modes` nor `self._log_vol` as relative
                # logarithmic quantities are volume-free
                self._rel_log_modes.append(lm)
                if self._flx is not None:
                    self._log_vol.append(lm[2:] - lm[1:-1])
                    assert self._rel_log_modes[-1].shape[
                        0] - 2 == self._log_vol[-1].shape[0]
        else:
            ve = f"invalid `harmonic_domain_type` {harmonic_domain_type!r}"
            raise ValueError(ve)

        # Poor man's domain
        self.tree_shape = {
            self._prefix + "_xizeromode": (),
            self._prefix + "_xifluctuations": (),
            self._prefix + "_xiloglogavgslope": (),
            self._prefix + "_xiexcitations": self._shape
        }
        if self._flx is not None:
            self.tree_shape[self._prefix + "_xiflexibility"] = ()
            # NOTE, entries of `self._rel_log_modes` are always one dimensional
            shp = (2, self._rel_log_modes[-1].shape[0] - 2)
            self.tree_shape[self._prefix + "_xispectrum"] = shp
        if self._asp is not None:
            self.tree_shape[self._prefix + "_xiasperity"] = ()

    def amplitude(self, primals):
        xi_zm = primals.get(self._prefix + "_xizeromode")
        xi_flu = primals.get(self._prefix + "_xifluctuations")
        xi_slp = primals.get(self._prefix + "_xiloglogavgslope")

        xi_flx = primals.get(self._prefix + "_xiflexibility")
        xi_asp = primals.get(self._prefix + "_xiasperity")
        xi_spc = primals.get(self._prefix + "_xispectrum")

        zm = self._zm(xi_zm)
        flu = self._flu(xi_flu)
        slope = self._slp(xi_slp)
        slope *= self._rel_log_modes[-1]
        ln_amplitude = slope

        if self._flx is not None:
            flx = self._flx(xi_flx)
            sig_flx = flx * np.sqrt(self._log_vol[-1])
            sig_flx = np.broadcast_to(sig_flx, (2, ) + self._log_vol[-1].shape)

            if self._asp is None:
                shift = np.stack(
                    (
                        self._log_vol[-1] / np.sqrt(12.),
                        np.ones_like(self._log_vol[-1])
                    ),
                    axis=0
                )
                asp = shift * sig_flx * xi_spc
            else:
                asp = self._asp(xi_asp)
                shift = np.stack(
                    (
                        self._log_vol[-1]**2 / 12.,
                        np.ones_like(self._log_vol[-1])
                    ),
                    axis=0
                )
                sig_asp = np.broadcast_to(np.array([[asp], [0.]]), shift.shape)
                asp = np.sqrt(shift + sig_asp) * sig_flx * xi_spc

            twolog = _twolog_integrate(self._log_vol[-1], asp)
            wo_slope = _remove_slope(self._rel_log_modes[-1], twolog)
            ln_amplitude += wo_slope

        # Exponentiate and norm the power spectrum
        amplitude = np.exp(ln_amplitude)
        if self._harmonic_dom_type == "fourier":
            # Take the sqrt of the integral of the slope w/o fluctuations and
            # zero-mode while taking into account the multiplicity of each mode
            norm = np.sqrt(2 * np.sum(amplitude[1:]**2) - amplitude[-1]**2)
        amplitude *= flu / norm
        amplitude = amplitude.at[0].set(zm)
        return amplitude

    def __call__(self, primals):
        xi_excitation = primals.get(self._prefix + "_xiexcitations")

        amplitude = self.amplitude(primals)
        if self._harmonic_dom_type == "fourier":
            # Every mode appears exactly two times, first ascending then
            # descending Save a little on the computational side by mirroring
            # the ascending part
            # NOTE, it would be possible to put this into an operator and use
            # index_mul
            harmonic_sqrt_power = np.concatenate(
                (amplitude, amplitude[-2:0:-1])
            )
        return harmonic_sqrt_power * xi_excitation
