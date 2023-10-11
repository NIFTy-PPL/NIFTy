# Copyright(C) 2013-2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from .model import WrappedCall
from .num import *


class LaplacePrior(WrappedCall):
    def __init__(self, alpha, **kwargs):
        """Transforms standard normally distributed random variables to a
        Laplace distribution.

        Parameters
        ----------
        alpha : float
            Scale parameter.
        name : hashable
            Name of the latent parameter that transformed to a Laplace
            distribution.
        shape : tuple
            Shape of the latent parameter that transformed to a Laplace
            distribution.
        dtype : dtype
            Data type of the latent parameter that transformed to a Laplace
            distribution.
        """
        self.alpha = alpha
        call = laplace_prior(self.alpha)
        super().__init__(call, white_domain=True, **kwargs)


class NormalPrior(WrappedCall):
    def __init__(self, mean, std, **kwargs):
        """Transforms standard normally distributed random variables to a
        normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the normal distribution.
        std : float
            Standard deviation of the normal distribution.
        name : hashable
            Name of the latent parameter that transformed to a normal
            distribution.
        shape : tuple
            Shape of the latent parameter that transformed to a normal
            distribution.
        dtype : dtype
            Data type of the latent parameter that transformed to a normal
            distribution.
        """
        self.mean = mean
        self.std = std
        call = normal_prior(self.mean, self.std)
        super().__init__(call, white_domain=True, **kwargs)


class LogNormalPrior(WrappedCall):
    def __init__(self, mean, std, **kwargs):
        """Transforms standard normally distributed random variables to a
        log-normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the log-normal distribution.
        std : float
            Standard deviation of the log-normal distribution.
        name : hashable
            Name of the latent parameter that transformed to a log-normal
            distribution.
        shape : tuple
            Shape of the latent parameter that transformed to a log-normal
            distribution.
        dtype : dtype
            Data type of the latent parameter that transformed to a log-normal
            distribution.
        """
        self.mean = mean
        self.std = std
        call = lognormal_prior(self.mean, self.std)
        super().__init__(call, white_domain=True, **kwargs)


class UniformPrior(WrappedCall):
    def __init__(self, a_min, a_max, **kwargs):
        """Transforms standard normally distributed random variables to a
        uniform distribution.

        Parameters
        ----------
        a_min : float
            Minimum value.
        a_max : float
            Maximum value.
        name : hashable
            Name of the latent parameter that transformed to a uniform
            distribution.
        shape : tuple
            Shape of the latent parameter that transformed to a uniform
            distribution.
        dtype : dtype
            Data type of the latent parameter that transformed to a uniform
            distribution.
        """
        self.low = self.a_min = a_min
        self.high = self.a_max = a_max
        call = uniform_prior(self.a_max, self.a_max)
        super().__init__(call, white_domain=True, **kwargs)


class InvGammaPrior(WrappedCall):
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
        name : hashable
            Name of the latent parameter that transformed to an inverse gamma
            distribution.
        shape : tuple
            Shape of the latent parameter that transformed to an inverse gamma
            distribution.
        dtype : dtype
            Data type of the latent parameter that transformed to an inverse
            gamma distribution.
        """
        self.a = a
        self.scale = scale
        self.loc = loc
        self.step = step
        call = invgamma_prior(self.a, self.scale, self.loc, self.step)
        super().__init__(call, white_domain=True, **kwargs)
