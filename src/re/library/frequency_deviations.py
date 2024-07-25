from ..gauss_markov import GaussMarkovProcess
from ..model import VModel, Model

import jax.numpy as jnp

# from jax import vmap
# from functools import reduce, partial
# from ..tree_math.vector_math import ShapeWithDtype
# from ..tree_math.forest_math import random_like
# from ..model import Initializer
# from ..tree_math.vector import Vector


# class MappedModel(Model):
#     """Maps a model to a higher dimensional space."""

#     def __init__(self, model, mapped_key, shape, first_axis=True):
#         """Intitializes the mapping class.

#         Parameters:
#         ----------
#         model: nifty.re.Model most probable a Correlated Field Model or a
#             Gauss-Markov Process
#         mapped_key: string, dictionary key for input dimension which is
#             going to be mapped.
#         shape: tuple, number of copies in each dim. Size of the
#         first_axis: if True prepends the number of copies
#             else they will be appended
#         """
#         self._model = model
#         ndof = reduce(lambda x, y: x * y, shape)
#         keys = model.domain.keys()
#         if mapped_key not in keys:
#             raise ValueError

#         xi_dom = model.domain[mapped_key]
#         if first_axis:
#             new_primals = ShapeWithDtype(
#                 (ndof,) + xi_dom.shape, xi_dom.dtype)
#             axs = 0
#             self._out_axs = 0
#             self._shape = shape + model.target.shape
#         else:
#             new_primals = ShapeWithDtype(
#                 xi_dom.shape + (ndof,), xi_dom.dtype)
#             axs = -1
#             self._out_axs = 1
#             self._shape = model.target.shape + shape

#         new_domain = model.domain.copy()
#         new_domain[mapped_key] = new_primals

#         xiinit = partial(random_like, primals=new_primals)

#         init = model.init
#         init = {k: init[k] if k != mapped_key else xiinit for k in keys}

#         self._axs = ({k: axs if k == mapped_key else None for k in keys},)
#         super().__init__(domain=new_domain, init=Initializer(init))

#     def __call__(self, x):
#         x = x.tree if isinstance(x, Vector) else x
#         return (vmap(self._model, in_axes=self._axs,
#                      out_axes=self._out_axs)(x)).reshape(self._shape)


class FrequencyDeviations(Model):
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

    # mapped_process = MappedModel(
    #     process, process.name, shape=shape_2d, first_axis=False)

    tmp = VModel(process, shape_2d[0]*shape_2d[1], out_axes=1)
    mapped_process = Model(
        lambda x: tmp(x).reshape((len(frequencies), shape_2d[0], shape_2d[1])),
        init=tmp.init)

    return FrequencyDeviations(
        mapped_process,
        frequencies,
        reference_frequency)


def build_frequency_devations_from_correlated_field():
    raise NotImplementedError
