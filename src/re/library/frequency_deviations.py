from typing import Union
from numpy.typing import ArrayLike

from ..model import Model
from ..gauss_markov import GaussMarkovProcess

import jax.numpy as jnp


class FrequencyDeviations(Model):
    def __init__(
        self,
        frequency_deviations_model: GaussMarkovProcess,  # TODO: Add 3d correlated field
        frequencies: Union[tuple[float], ArrayLike],
        reference_frequency_index: int
    ):
        self.three_d_frequency_deviations = frequency_deviations_model

        relative_freqs = jnp.array(
            frequencies - frequencies[reference_frequency_index])[:, None, None]
        frequencies_denominator = 1 / jnp.sum(relative_freqs**2)

        def deviations_call(p):
            dev = frequency_deviations_model(p)
            dev = dev - dev[reference_frequency_index]

            # m = sum_l(gmp(l) * (l-l0)) / sum_l((l-l0)**2)
            dev_slope = (jnp.sum(dev*relative_freqs, axis=0)
                         * frequencies_denominator)

            return dev - dev_slope

        super().__init__(
            deviations_call, init=frequency_deviations_model.init)
