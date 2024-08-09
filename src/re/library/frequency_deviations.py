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
        self.frequency_deviations = frequency_deviations_model

        slicing_tuple = (
            (slice(None),) +
            (None,) * len(self.frequency_deviations.target.shape[1:])
        )
        relative_freqs = jnp.array(
            frequencies - frequencies[reference_frequency_index])[slicing_tuple]
        frequencies_denominator = 1 / jnp.sum(relative_freqs**2)

        def deviations_call(p):
            dev = self.frequency_deviations(p)
            dev = dev - dev[reference_frequency_index]

            # m = sum_l(gmp(l) * (l-l0)) / sum_l((l-l0)**2)
            dev_slope = (jnp.sum(dev*relative_freqs, axis=0)
                         * frequencies_denominator)

            return dev - dev_slope

        super().__init__(
            deviations_call, init=frequency_deviations_model.init)
