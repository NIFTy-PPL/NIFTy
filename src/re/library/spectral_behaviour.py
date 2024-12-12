# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors:  Julian Rüstig

from abc import ABC, abstractmethod

import numpy as np
# from numpy.typing import ArrayLike

import jax.numpy as jnp
from jax.typing import ArrayLike

from ..model import Model


class LogSpectralBehavior(Model, ABC):
    @property
    @abstractmethod
    def relative_log_frequencies(self):
        '''The spectral axis is fixed to be the first index of the output of
        the model. The `relative_log_frequencies` cast the output arrays of the
        models to conform with this convention by following the numpy casting
        conventions.'''
        pass

    @abstractmethod
    def mean(self, p) -> list[ArrayLike]:
        pass

    @abstractmethod
    def fluctuations(self, p) -> list[ArrayLike]:
        pass

    @abstractmethod
    def fluctuations_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def mean_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def remove_degeneracy_of_spectral_deviations(
        self,
        deviations: ArrayLike
    ) -> ArrayLike:
        pass


class SingleLogSpectralBehavior(LogSpectralBehavior, ABC):
    '''Abstract base class for a model with single spectral behavior. 
    Hence, mean is a single number and fluctations is an Array. '''
    @property
    @abstractmethod
    def relative_log_frequencies(self):
        '''The spectral axis is fixed to be the first index of the output of
        the model. The `relative_log_frequencies` cast the output arrays of the
        models to conform with this convention by following the numpy casting
        conventions.'''
        pass

    @abstractmethod
    def mean(self, p) -> ArrayLike:
        pass

    @abstractmethod
    def fluctuations(self, p) -> ArrayLike:
        pass

    @abstractmethod
    def fluctuations_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def mean_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def remove_degeneracy_of_spectral_deviations(
        self,
        deviations: ArrayLike
    ) -> ArrayLike:
        pass


class SpectralIndex(SingleLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        mean: Model,
        fluctuations: Model,
        reference_frequency_index: int,
    ):
        # The spectral axis is fixed to be the first index of the output of the
        # model. The slicing tuple takes care of the numpy slicing convention.
        slicing_tuple = (
            (slice(None),) + (None,)*len(fluctuations.target.shape))
        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._mean = mean
        self._fluctuations = fluctuations

        self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        super().__init__(domain=self._mean.domain | self._fluctuations.domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return self._mean(p)

    def fluctuations(self, p) -> ArrayLike:
        return self._fluctuations(p)

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        return self.fluctuations(p) * self.relative_log_frequencies

    def mean_with_frequencies(self, p) -> ArrayLike:
        return self.mean(p) * self.relative_log_frequencies

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        dev_slope = (
            jnp.sum(deviations * self.relative_log_frequencies, axis=0) *
            self._denominator)
        return deviations - dev_slope * self.relative_log_frequencies

    def __call__(self, p):
        # FIXME : only needed for instantiation of the model. However, maybe
        # there is a better way !!!
        # Since the apply is not needed
        return None
