# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Margret Westerkamp, Vincent Eberle, Philipp Frank

from functools import partial, reduce
import jax
import jax.numpy as jnp
from .model import Model, Initializer
from .tree_math.vector_math import ShapeWithDtype
from .tree_math.forest_math import random_like
from .tree_math.vector import Vector


class MappedModel(Model):
    """Maps a model to a higher dimensional space."""

    def __init__(self, model, mapped_key, shape, first_axis=True):
        """Intitializes the mapping class.

        Parameters:
        ----------
        model: nifty.re.Model most probable a Correlated Field Model or a
            Gauss-Markov Process
        mapped_key: string, dictionary key for input dimension which is
            going to be mapped.
        shape: tuple, number of copies in each dim. Size of the
        first_axis: if True prepends the number of copies
            else they will be appended
        """
        self._model = model
        ndof = reduce(lambda x, y: x * y, shape)
        keys = model.domain.keys()
        if mapped_key not in keys:
            raise ValueError

        xi_dom = model.domain[mapped_key]
        if first_axis:
            new_primals = ShapeWithDtype((ndof,) + xi_dom.shape, xi_dom.dtype)
            axs = 0
            self._out_axs = 0
            self._shape = shape + model.target.shape
        else:
            new_primals = ShapeWithDtype(xi_dom.shape + (ndof,), xi_dom.dtype)
            axs = -1
            self._out_axs = 1
            self._shape = model.target.shape + shape

        new_domain = model.domain.copy()
        new_domain[mapped_key] = new_primals

        xiinit = partial(random_like, primals=new_primals)

        init = model.init
        init = {k: init[k] if k != mapped_key else xiinit for k in keys}

        self._axs = ({k: axs if k == mapped_key else None for k in keys},)
        super().__init__(domain=new_domain, init=Initializer(init))

    def __call__(self, x):
        x = x.tree if isinstance(x, Vector) else x
        return (jax.vmap(self._model, in_axes=self._axs,
                         out_axes=self._out_axs)(x)).reshape(self._shape)


class GeneralModel(Model):
    """General Sky Model, plugging together several components."""

    def __init__(self, dict_of_fields={}):
        """Initializes the general sky model.

        keys for the dictionary:
        -----------------------
        spatial: typically 2D Model for spatial log flux(x),
            where x is the spatial vector.
        freq_plaw: jubik0.build_power_law or other 3D model.
        freq_dev: additional flux(frequency / energy) dependent process.
            often deviations from freq_plaw.

        Parameters:
        ----------
        dict of fields: the respective keys and the
            nifty.re.models as values"""
        self._available_fields = dict_of_fields

    def build_model(self):
        """#NOTE Docstring."""
        def add_functions(f1, f2):
            def function(x):
                return f1(x) + f2(x)
            return function

        if 'spatial' not in self._available_fields.keys() or self._available_fields['spatial'] is None:
            raise NotImplementedError
        else:
            spatial = self._available_fields['spatial']
            func = spatial
            domain = spatial.domain
            if 'freq_plaw' in self._available_fields.keys() and self._available_fields['freq_plaw'] is not None:
                plaw = self._available_fields['freq_plaw']
                func = add_functions(func, plaw)
                domain = domain | plaw.domain
            if 'freq_dev' in self._available_fields.keys() and self._available_fields['freq_dev'] is not None:
                dev = self._available_fields['freq_dev']

                def extract_keys(a, domain):
                    b = {key: a[key] for key in domain}
                    return b

                def extracted_dev(op):
                    def callable_dev(x):
                        return op(extract_keys(x, op.domain))
                    return callable_dev

                func = add_functions(func, extracted_dev(dev))
                domain = domain | dev.domain
            if 'pol' in self._available_fields.keys() and self._available_fields['pol'] is not None:
                raise NotImplementedError
            if 'time' in self._available_fields.keys() and self._available_fields['time'] is not None:
                raise NotImplementedError
            res_func = lambda x: func(x) if len(func(x).shape) == 3 else jnp.reshape(func(x),
                                                                                     (1,) + func(x).shape)
            res = Model(res_func, domain=domain)
        return res


def build_power_law(freqs, alph):
    """Models a logarithm of a power law.
    Building bloc for e.g. a multifrequency model

    Parameters:
    -----------
    freqs: log(frequencies) or log (energies) # FIXME use it like this in our code
    alpha: 2D spectral index map

    returns:
    --------
    logarithmic powerlaw, meaning a linear function
    with slope alpha evaluated at freqs.

    """
    if isinstance(alph, Model):
        res = lambda x: jnp.outer(freqs, alph(x)).reshape(freqs.shape + alph.target.shape)
    elif isinstance(alph, float):
        # FIXME not working at the moment
        res = jnp.outer(freqs, alph).reshape(freqs.shape)
    return Model(res, domain=alph.domain)
