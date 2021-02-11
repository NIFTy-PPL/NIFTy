from jax import numpy as np

class Amplitude():
    def __init__(
        self,
        shape,
        zeromode,
        fluctuations,
        loglogavgslope,
        prefix,
        harmonic_domain_type,
    ):
        self._shape = tuple(shape)
        self._zm_m, self._zm_std = zeromode  # TODO: moment-match
        self._fl_m, self._fl_std = fluctuations  # TODO: moment-match
        self._slope_m, self._slope_std = loglogavgslope
        self._prefix = prefix
        self._harmonic_dom_type = harmonic_domain_type.lower()

        if len(self._shape) > 1:
            ve = "Multi-dimensional amplitude operator not implemented (yet)"
            raise ValueError(ve)

        if self._harmonic_dom_type == "fourier":
            self._rel_log_modes = []
            for dim in self._shape:
                lm = np.arange(dim / 2 + 1., dtype=float)
                lm = index_update(lm, slice(1, None), np.log(lm[1:]))
                self._rel_log_modes.append(lm)
        else:
            ve = f"invalid `harmonic_domain_type` {harmonic_domain_type!r}"
            raise ValueError(ve)

    def __call__(self, primals):
        xi_zm = primals.get(self._prefix + "_xizeromode")
        xi_fl = primals.get(self._prefix + "_xifluctuations")
        xi_slope = primals.get(self._prefix + "_xiloglogslope")
        xi_excitation = primals.get(self._prefix + "_xiexcitations")

        zm = self._zm_m + self._zm_std * xi_zm  # TODO: moment-match
        fl = self._fl_m + self._fl_std * xi_fl  # TODO: moment-match
        slope = self._slope_m + self._slope_std * xi_slope
        amplitude = np.exp(slope * self._rel_log_modes[-1])
        if self._harmonic_dom_type == "fourier":
            # Take the sqrt of the integral of the slope w/o fluctuations and
            # zero-mode while taking into account the multiplicity of each mode
            norm = np.sqrt(2 * np.sum(amplitude[1:]**2) - amplitude[-1]**2)
        amplitude *= fl / norm
        amplitude = index_update(amplitude, 0, zm)
        self._debug_amplitude = amplitude
        if self._harmonic_dom_type == "fourier":
            # Every mode appears exactly two times, first ascending then descending
            # Save a little on the computational side by mirroring the ascending part
            # NOTE, it would be possible to use index_mul and slice cleverly
            harmonic_sqrt_power = np.concatenate(
                (amplitude, amplitude[-2:0:-1])
            )
        return harmonic_sqrt_power * xi_excitation


