# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors:  Julian Rüstig

from abc import ABC, abstractmethod
from functools import reduce

import jax.numpy as jnp
from jax.typing import ArrayLike

from .scaled_excitations import ScaledExcitations

from ..model import Model
from ..tree_math.vector import Vector


class HarmonicLogSpectralBehavior(Model, ABC):
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


class SingleHarmonicLogSpectralBehavior(HarmonicLogSpectralBehavior, ABC):
    '''Abstract base class for a model with single spectral behavior.
    Hence, mean is a single number and fluctations is an Array.

    Note
    ----
    This can be used to speed up some calculations in the
    `CorrelatedMultiFrequencySky` class.
    '''
    pass


class SpectralIndex(SingleHarmonicLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        mean: Model,
        spectral_scaled_excitations: ScaledExcitations,
        reference_frequency_index: int,
    ):
        '''Spectral Index spectral behavior. Special case of the
        `SpectralPolynomial`. However, since it has a single parameter, it can
        be used to speed up evaluations.

        .. math::
            \\log F(\\nu, A, ) = A^{spec.}
            * \\left\\{ \\mathrm{fluc.}^A \\xi^A \\nu  \\right\\} 
            + \\mathrm{mean}^A \\nu

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies. The relative frequencies are calculated
            internally. See `reference_frequency_index`.
        mean: Model
            The means of the polynomial model. See equation above.
        spectral_scaled_excitations: Model
            The fluctuations already applied to the xis. Hence, the
            combination of \\mathrm{fluc.}^A \\xi^A.
        reference_frequency_index: int
            The index of the reference frequency. Used in order to calculate
            the relative frequencies.
        '''
        # NOTE : The model assumes that spectral axis is fixed to be the first
        # index of the output model. The slicing tuple matches the shape of the
        # relative_log_frequencies using the numpy slicing convention.
        slicing_tuple = (
            (slice(None),) + (None,)*len(spectral_scaled_excitations.target.shape))
        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._mean = mean
        self.spectral_scaled_excitations = spectral_scaled_excitations

        self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        super().__init__(domain=self._mean.domain | self.spectral_scaled_excitations.domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return self._mean(p)

    def fluctuations(self, p) -> ArrayLike:
        return self.spectral_scaled_excitations(p)

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
        # NOTE : only needed for instantiation of the model. However, maybe
        # there is a better way !!!
        # Since the apply is not needed
        return None


class SpectralPolynomial(HarmonicLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        means: list[Model],
        fluctuations: list[Model],
        reference_frequency_index: int,
    ):
        '''Polynomial spectral behavior. The spectral index is a special case
        of this model.

        .. math::
            \\log F(\\nu, A, B, ...) = A^{spec.}
            * \\left\\{ \\mathrm{fluc.}^A \\xi^A \\nu + \\mathrm{fluc.}^B \\xi^B \\nu^2 + ... \\right\\} 
            + \\mathrm{mean}^A \\nu + \\mathrm{mean}^B \\nu^2 + ...

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies. The relative frequencies are calculated
            internally. See `reference_frequency_index`.
        means: Model
            The means of the polynomial model. See equation above.
        fluctuations: Model
            The fluctuations, already applied to the xis. Hence, the
            combination of \\mathrm{fluc.}^A \\xi^A.
        reference_frequency_index: int
            The index of the reference frequency. Used in order to calculate
            the relative frequencies.
        '''

        # NOTE : The sizes between the means and the fluctuations must match.
        assert isinstance(means, list)
        assert isinstance(fluctuations, list)
        assert len(means) == len(fluctuations)
        fluctuations_target_shape = fluctuations[0].target.shape
        for fluc in fluctuations:
            assert fluc.target.shape == fluctuations_target_shape

        # NOTE : The model assumes that spectral axis is fixed to be the first
        # index of the output model. The slicing tuple matches the shape of the
        # relative_log_frequencies using the numpy slicing convention.
        slicing_tuple = (
            (slice(None),) + (None,)*len(fluctuations_target_shape))
        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._means = means
        self._fluctuations = fluctuations
        # self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        domain = reduce(
            lambda a, b: a | b,
            [(m.domain.tree if isinstance(m.domain, Vector) else m.domain)
             for m in self._means+self._fluctuations]
        )

        super().__init__(domain=domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return [m(p) for m in self._means]

    def fluctuations(self, p) -> ArrayLike:
        return [f(p) for f in self._fluctuations]

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        '''Implements the fluctuations part. See above.

        .. math::
            \\mathrm{fluc.}^A\\xi^A\\nu + \\mathrm{fluc.}^B\\xi^B\\nu^2 + ...
        '''

        values = jnp.array([f(p)*self.relative_log_frequencies**(i+1)
                            for i, f in enumerate(self._fluctuations)])

        # NOTE: The 0th-axis contains the polynomial values, hence the sum
        # over 0th-axis returns the polynomial value at the different log-
        # frequencies.
        return jnp.sum(values, axis=0)

    def mean_with_frequencies(self, p) -> ArrayLike:
        '''Implements the means part. See above.

        .. math::
            \\mathrm{mean}^A \\nu + \\mathrm{mean}^B \\nu^2 + ...
        '''

        values = jnp.array([f(p)*self.relative_log_frequencies**(i+1)
                            for i, f in enumerate(self._means)])

        return jnp.sum(values, axis=0)

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        # FIXME:: This needs to be implemented !!!
        raise NotImplementedError

        dev_slope = (
            jnp.sum(deviations * self.relative_log_frequencies, axis=0) *
            self._denominator)
        return deviations - dev_slope * self.relative_log_frequencies

    def __call__(self, p):
        # NOTE : only needed for instantiation of the model. However, maybe
        # there is a better way !!!
        # Since the apply is not needed
        return None
