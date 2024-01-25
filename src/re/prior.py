# Copyright(C) 2013-2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from .model import WrappedCall
from .num import (
    invgamma_prior, laplace_prior, lognormal_prior, normal_prior, uniform_prior
)

_doc_shared = """name : hashable, optional
            Name within the new `input` on which `call` acts.
        shape : tuple or tree-like structure of ShapeWithDtype
            Shape of the latent parameter(s) that are transformed to the desired
            distribution. This can also be an arbitrary shape-dtype structure in
            which case `dtype` is ignored. Defaults to a scalar.
        dtype : dtype
            Data type of the latent parameter(s) that are transformed to the
            desired distribution."""


def _format_doc(func):
    func.__doc__ = func.__doc__.format(_doc_shared=_doc_shared)
    return func


class LaplacePrior(WrappedCall):
    @_format_doc
    def __init__(self, alpha, **kwargs):
        """Transforms standard normally distributed random variables to a
        Laplace distribution.

        Parameters
        ----------
        alpha : tree-like structure with arithmetics
            Scale parameter.
        {_doc_shared}
        """
        self.alpha = alpha
        call = laplace_prior(self.alpha)
        super().__init__(call, white_init=True, **kwargs)


class NormalPrior(WrappedCall):
    @_format_doc
    def __init__(self, mean, std, **kwargs):
        """Transforms standard normally distributed random variables to a
        normal distribution.

        Parameters
        ----------
        mean : tree-like structure with arithmetics
            Mean of the normal distribution.
        std : tree-like structure with arithmetics
            Standard deviation of the normal distribution.
        {_doc_shared}
        """
        self.mean = mean
        self.std = std
        call = normal_prior(self.mean, self.std)
        super().__init__(call, white_init=True, **kwargs)


class LogNormalPrior(WrappedCall):
    @_format_doc
    def __init__(self, mean, std, **kwargs):
        """Transforms standard normally distributed random variables to a
        log-normal distribution.

        Parameters
        ----------
        mean : tree-like structure with arithmetics
            Mean of the log-normal distribution.
        std : tree-like structure with arithmetics
            Standard deviation of the log-normal distribution.
        {_doc_shared}
        """
        self.mean = mean
        self.std = std
        call = lognormal_prior(self.mean, self.std)
        super().__init__(call, white_init=True, **kwargs)


class UniformPrior(WrappedCall):
    @_format_doc
    def __init__(self, a_min, a_max, **kwargs):
        """Transforms standard normally distributed random variables to a
        uniform distribution.

        Parameters
        ----------
        a_min : tree-like structure with arithmetics
            Minimum value.
        a_max : tree-like structure with arithmetics
            Maximum value.
        {_doc_shared}
        """
        self.low = self.a_min = a_min
        self.high = self.a_max = a_max
        call = uniform_prior(self.a_min, self.a_max)
        super().__init__(call, white_init=True, **kwargs)


class InvGammaPrior(WrappedCall):
    @_format_doc
    def __init__(self, a, scale, loc=0., step=1e-2, **kwargs):
        """Transforms standard normally distributed random variables to an
        inverse gamma distribution.

        Parameters
        ----------
        a : float
            Shape parameter.
        scale : float
            Scale parameter.
        loc : float
            Location parameter.
        step : float
            Step size for numerical integration.
        {_doc_shared}

        Notes
        -----
        Broadcasting over tree-like structure is not yet implemented. Please
        file an issue if you need this feature.
        """
        self.a = a
        self.scale = scale
        self.loc = loc
        self.step = step
        call = invgamma_prior(self.a, self.scale, self.loc, self.step)
        super().__init__(call, white_init=True, **kwargs)
