# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig

from .mf_model_utils import _build_distribution_or_default

from ..model import Model
from ..num.stats_distributions import lognormal_prior, normal_prior


class ScaledExcitations(Model):
    def __init__(
        self,
        fluctuations: Model,
        latent_space_xi: Model
    ):
        self.fluctuations = fluctuations
        self.latent_space_xi = latent_space_xi
        super().__init__(domain=fluctuations.domain | latent_space_xi.domain)

    def __call__(self, p):
        return self.fluctuations(p) * self.latent_space_xi(p)


def build_scaled_excitations(
    prefix: str,
    fluctuation_settings: dict,
    shape: tuple[int],
) -> ScaledExcitations:
    fluctuations = _build_distribution_or_default(
        fluctuation_settings,
        f'{prefix}_fluctuations',
        lognormal_prior
    )
    latent_space_xi = _build_distribution_or_default(
        (0.0, 1.0),
        f'{prefix}_xi',
        normal_prior,
        shape=shape)
    return ScaledExcitations(fluctuations, latent_space_xi)

