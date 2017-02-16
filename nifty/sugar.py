# -*- coding: utf-8 -*-

from nifty import PowerSpace,\
                  Field,\
                  DiagonalOperator,\
                  FFTOperator

__all__ = ['create_power_operator']


def create_power_operator(domain, power_spectrum, distribution_strategy='not'):
    if not domain.harmonic:
        fft = FFTOperator(domain)
        domain = fft.target[0]

    power_domain = PowerSpace(domain,
                              distribution_strategy=distribution_strategy)

    fp = Field(power_domain,
               val=power_spectrum,
               distribution_strategy=distribution_strategy)

    f = fp.power_synthesize(std=0)

    power_operator = DiagonalOperator(domain, diagonal=f)

    return power_operator
