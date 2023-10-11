# Copyright(C) 2013-2021 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from .lanczos import (
    lanczos_tridiag, stochastic_logdet_from_lanczos, stochastic_lq_logdet
)
from .stats_distributions import (
    interpolator, invgamma_invprior, laplace_prior, invgamma_prior,
    lognormal_invprior, lognormal_moments, lognormal_prior, normal_invprior,
    normal_prior, uniform_prior
)
from .unique import amend_unique, amend_unique_, unique
