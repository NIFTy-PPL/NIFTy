from ..model import Model
from ..gauss_markov import GaussMarkovProcess

import jax.numpy as jnp


class FrequencyDeviations(Model):
    def __init__(
        self,
        three_d_frequency_deviations: GaussMarkovProcess,  # TODO: Add 3d correlated field
        frequencies: tuple[float],
        reference_frequency: int
    ):
        self.three_d_frequency_deviations = three_d_frequency_deviations

        frequencies_relative = jnp.array(
            frequencies - frequencies[reference_frequency])[:, None, None]
        frequencies_denomenator = 1 / jnp.sum(frequencies_relative**2)

        def deviations_call(p):
            dev = three_d_frequency_deviations(p)
            dev = dev - dev[reference_frequency]

            # m = sum_l(gmp(l) * (l-l0)) / sum_l((l-l0)**2)
            dev_slope = (jnp.sum(dev*frequencies_relative, axis=0)
                         * frequencies_denomenator)

            return dev - dev_slope

        super().__init__(
            deviations_call, init=three_d_frequency_deviations.init)
