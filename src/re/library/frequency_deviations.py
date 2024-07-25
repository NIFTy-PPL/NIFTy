import nifty8.re as jft
from nifty8.re import GaussMarkovProcess
from nifty8.re import VModel

import jax.numpy as jnp


class FrequencyDeviations(jft.Model):
    def __init__(
        self,
        three_d_frequency_deviations: VModel,  # TODO: Add 3d correlated field
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


def build_frequency_devations_from_1d_process(
    process: GaussMarkovProcess,
    shape_2d: tuple[int],
    frequencies: tuple[float],
    reference_frequency: int,
):

    tmp = VModel(process, shape_2d[0]*shape_2d[1], out_axes=1)
    mapped_process = jft.Model(
        lambda x: tmp(x).reshape((len(frequencies), shape_2d[0], shape_2d[1])),
        init=tmp.init)

    return FrequencyDeviations(
        mapped_process,
        frequencies,
        reference_frequency)


def build_frequency_devations_from_correlated_field():
    raise NotImplementedError
